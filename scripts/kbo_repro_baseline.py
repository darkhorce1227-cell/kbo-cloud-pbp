"""Frozen, market-free, phase-coherent baseline and one-block ablations.

No production activation. See research/repro_baseline/protocol.json.
Only prior calendar dates enter history; monthly coefficients use earlier months.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import scipy
from scipy.stats import nbinom
import sklearn
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
HITS = {"single", "double", "triple", "home_run"}
WALKS = {"walk", "intent_walk", "hit_by_pitch"}
KS = {"strikeout", "strikeout_double_play"}
PHASES = ("f5", "late")


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_sources(paths, cfg):
    all_frames, manifest = [], []
    for path in paths:
        digest = sha_file(path)
        accepted = {v: int(k) for k, v in cfg["source_sha256"].items()}
        if digest not in accepted:
            raise ValueError(f"Unregistered source hash: {path}: {digest}")
        year = accepted[digest]
        x = pd.read_csv(path, low_memory=False, dtype={"game_pk": str, "pitcher": str, "batter": str})
        x["game_date"] = pd.to_datetime(x.game_date).dt.normalize()
        if set(x.game_date.dt.year) != {year}:
            raise ValueError("Source year mismatch")
        if x.duplicated(["game_pk", "at_bat_number", "pitch_number"]).any():
            raise ValueError("Duplicate stable pitch keys")
        original = len(x)
        x = x[x.game_date <= pd.Timestamp(cfg["season_end"][str(year)])].copy()
        manifest.append({"file": Path(path).name, "sha256": digest, "year": year,
                         "source_rows": original, "included_rows": len(x),
                         "included_games": int(x.game_pk.nunique()),
                         "calendar_cutoff": cfg["season_end"][str(year)]})
        all_frames.append(x)
    out = pd.concat(all_frames, ignore_index=True)
    if out.duplicated(["game_pk", "at_bat_number", "pitch_number"]).any():
        raise ValueError("Duplicate observations across source files")
    return out, manifest


def prepare_games(raw):
    """Retain final PA state, score labels, and explicitly labelled identity proxies."""
    x = raw.copy()
    x["game_date"] = pd.to_datetime(x.game_date).dt.normalize()
    x = x.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"], kind="stable")
    pa = x.drop_duplicates(["game_pk", "at_bat_number"], keep="last").copy()
    pa["batting_team"] = np.where(pa.inning_topbot.eq("top"), pa.away_team, pa.home_team)
    pa["pitching_team"] = np.where(pa.inning_topbot.eq("top"), pa.home_team, pa.away_team)
    ev = pa.events.fillna("")
    # Stat vector: PA, H, TB, BB+HBP, K, HR. Used only after date batch closes.
    pa["h"] = ev.isin(HITS).astype(int)
    pa["tb"] = ev.map({"single": 1, "double": 2, "triple": 3, "home_run": 4}).fillna(0)
    pa["bb"] = ev.isin(WALKS).astype(int)
    pa["k"] = ev.isin(KS).astype(int)
    pa["hr"] = ev.eq("home_run").astype(int)
    out, exclusions = [], []
    for pk, g in pa.groupby("game_pk", sort=False):
        g = g.sort_values("at_bat_number", kind="stable")
        f = g[g.inning <= 5]
        completed_f5 = (g.inning > 5).any()
        if not completed_f5 and len(f) and "post_outs" in g:
            z = f.iloc[-1]
            completed_f5 = z.inning == 5 and z.inning_topbot == "bot" and z.post_outs == 3
        if not completed_f5:
            exclusions.append({"game_pk": str(pk), "reason": "F5_completion_unverified"})
            continue
        last, fifth = g.iloc[-1], f.iloc[-1]
        scores = {side: [int(fifth[f"post_{side}_score"]), int(last[f"post_{side}_score"])]
                  for side in ("away", "home")}
        if any(z[1] < z[0] or z[0] < 0 for z in scores.values()):
            raise ValueError(f"Score decrease in {pk}")
        teams = {"away": str(last.away_team), "home": str(last.home_team)}
        first_starters, starters, relief, lineups, hitters = {}, {}, {}, {}, {}
        for side, team in teams.items():
            pitching = g[g.pitching_team.astype(str).eq(team)]
            starter = str(pitching.pitcher.iloc[0])
            first_starters[team] = starter
            sp = pitching[pitching.pitcher.astype(str).eq(starter)]
            rp = pitching[~pitching.pitcher.astype(str).eq(starter)]
            stats = lambda q: np.r_[len(q), q[["h", "tb", "bb", "k", "hr"]].sum().to_numpy(float)]
            if len(rp):
                entry = rp.iloc[0]
                outs = (int(entry.inning) - 1) * 3 + int(entry.outs_when_up)
            elif "post_outs" in sp:
                outs = int(((sp.inning - 1) * 3 + sp.post_outs).max())
            else:
                outs = None
            if outs is not None and not 0 <= outs <= 45:
                raise ValueError(f"Invalid starter outs in {pk}")
            starters[team] = {"id": starter, "stats": stats(sp), "outs": outs}
            relief[team] = stats(rp)
            batting = g[g.batting_team.astype(str).eq(team)]
            nine = batting.drop_duplicates("batter", keep="first").head(9)
            lineups[team] = nine.batter.astype(str).tolist() if len(nine) == 9 else None
            hitters[team] = {str(p): stats(q) for p, q in batting.groupby("batter")}
        out.append({"game_pk": str(pk), "date": pd.Timestamp(last.game_date), "teams": teams,
                    "scores": scores, "starter_ids": first_starters, "starters": starters,
                    "relief": relief, "lineups": lineups, "hitters": hitters})
    return sorted(out, key=lambda z: (z["date"], z["game_pk"])), exclusions


def _sum_vectors(records):
    return np.sum([r[1] for r in records], axis=0) if records else np.zeros(6)


def build_features(games, cfg):
    """Date batch barrier prevents target and same-day outcome contamination."""
    team_history = defaultdict(list)
    pitcher_history = defaultdict(list)
    pitcher_outs = defaultdict(list)
    batter_history = defaultdict(list)
    bp_history = defaultdict(list)
    league_events = defaultdict(lambda: np.zeros(6))
    league_starter_outs = defaultdict(list)
    rows = []
    groups = defaultdict(list)
    for g in games:
        groups[g["date"]].append(g)
    for date, day in sorted(groups.items()):
        year = date.year
        prior = [r for (yr, _), hist in team_history.items() if yr == year for r in hist]
        # Priors are fixed defaults only before any season observation.
        lg_event = league_events[year]
        lg_kbb = (lg_event[4] - lg_event[3] + 120 * .10) / (lg_event[0] + 120)
        lg_hr = (lg_event[5] + 120 * .03) / (lg_event[0] + 120)
        lg_ob = (lg_event[1] + lg_event[3] + 120 * .32) / (lg_event[0] + 120)
        lg_power = (lg_event[2] - lg_event[1] + 120 * .13) / (lg_event[0] + 120)
        lg_outs = float(np.mean(league_starter_outs[year])) if league_starter_outs[year] else 15.
        cutoff = date - pd.Timedelta(days=365)
        for g in day:
            for side, team in g["teams"].items():
                other = "home" if side == "away" else "away"
                opponent = g["teams"][other]
                spid = g["starter_ids"][opponent]  # opposing, never own-team starter
                sph = [r for r in pitcher_history[spid] if r[0] >= cutoff]
                v = _sum_vectors(sph)
                pseudo = cfg["starter_prior_bf"]
                kbb = (v[4] - v[3] + pseudo * lg_kbb) / (v[0] + pseudo)
                hr = (v[5] + pseudo * lg_hr) / (v[0] + pseudo)
                oh = [r[1] for r in pitcher_outs[spid] if r[0] >= cutoff][-5:]
                ps = cfg["starter_prior_starts"]
                eouts = (sum(oh) + ps * lg_outs) / (len(oh) + ps)
                bph = bp_history[(year, opponent)]
                bpv = _sum_vectors(bph)
                pb = cfg["relief_prior_bf"]
                bpkbb = (bpv[4] - bpv[3] + pb * lg_kbb) / (bpv[0] + pb)
                bphr = (bpv[5] + pb * lg_hr) / (bpv[0] + pb)
                recent_bf = sum(r[1][0] for r in bph if r[0] >= date - pd.Timedelta(days=3))
                ids = g["lineups"][team]
                # Normal lineup benchmark = same team's prior season PA mix.
                team_events = np.zeros(6)
                for h in team_history[(year, team)]:
                    team_events += h["events"]
                normal_ob = (team_events[1] + team_events[3] + 120 * lg_ob) / (team_events[0] + 120)
                normal_power = (team_events[2] - team_events[1] + 120 * lg_power) / (team_events[0] + 120)
                lineup_ob, lineup_power = [], []
                for batter in ids or []:
                    bh = [r for r in batter_history[batter] if r[0] >= cutoff]
                    z = _sum_vectors(bh)
                    ph = cfg["batter_prior_pa"]
                    lineup_ob.append((z[1] + z[3] + ph * lg_ob) / (z[0] + ph))
                    lineup_power.append((z[2] - z[1] + ph * lg_power) / (z[0] + ph))
                for pi, phase in enumerate(PHASES):
                    fallback = (2.5, 2.0)[pi]
                    lg = float(np.mean([r[phase][0] for r in prior])) if prior else fallback
                    recent = [r for r in prior if r["date"] >= date - pd.Timedelta(days=cfg["league_recent_days"])]
                    po = cfg["league_prior_team_games"]
                    offset = (sum(r[phase][0] for r in recent) + po * lg) / (len(recent) + po)
                    mine, opp = team_history[(year, team)], team_history[(year, opponent)]
                    pt = cfg["team_prior_games"]
                    off = (sum(r[phase][0] for r in mine) + pt * lg) / (len(mine) + pt)
                    deff = (sum(r[phase][1] for r in opp) + pt * lg) / (len(opp) + pt)
                    short = []
                    for days in (7, 14):
                        hh = [r for r in mine if r["date"] >= date - pd.Timedelta(days=days)]
                        short.append((sum(r[phase][0] for r in hh) + pt * off) / (len(hh) + pt))
                    yy = g["scores"][side]
                    target = yy[0] if phase == "f5" else yy[1] - yy[0]
                    source_dates = [r["date"] for r in prior]
                    source_dates += [r[0] for r in sph]
                    max_date = max(source_dates).date().isoformat() if source_dates else None
                    rows.append({"game_pk": g["game_pk"], "date": date, "team": team,
                                 "opponent": opponent, "opposing_starter": spid,
                                 "is_home": int(side == "home"), "phase": phase,
                                 "target": target, "offset": offset,
                                 "off_long_log": math.log(off / lg),
                                 "opp_def_long_log": math.log(deff / lg),
                                 "opp_sp_kbb_delta": kbb - lg_kbb,
                                 "opp_sp_hr_delta": hr - lg_hr,
                                 "opp_sp_outs_delta": (eouts - lg_outs) / 3.,
                                 "off7_delta": math.log(short[0] / off),
                                 "off14_delta": math.log(short[1] / off),
                                 "opp_bp_kbb_delta": bpkbb - lg_kbb,
                                 "opp_bp_hr_delta": bphr - lg_hr,
                                 "opp_bp_recent3_log_bf": math.log1p(recent_bf),
                                 "lineup_onbase_delta": float(np.mean(lineup_ob) - normal_ob) if ids else 0.,
                                 "lineup_power_delta": float(np.mean(lineup_power) - normal_power) if ids else 0.,
                                 "lineup_proxy_available": bool(ids), "prior_sp_bf": v[0],
                                 "prior_sp_starts": len(oh), "history_max_date": max_date})
        # NO writes to history are permitted above this barrier.
        for g in day:
            for side, team in g["teams"].items():
                other = "home" if side == "away" else "away"
                a, h = g["scores"][side], g["scores"][other]
                stats = sum(g["hitters"][team].values(), np.zeros(6))
                team_history[(year, team)].append({"date": date, "f5": [a[0], h[0]],
                                                   "late": [a[1] - a[0], h[1] - h[0]], "events": stats})
                sp = g["starters"][team]
                pitcher_history[sp["id"]].append((date, sp["stats"]))
                if sp["outs"] is not None:
                    pitcher_outs[sp["id"]].append((date, sp["outs"]))
                    league_starter_outs[year].append(sp["outs"])
                bp_history[(year, team)].append((date, g["relief"][team]))
                for batter, vv in g["hitters"][team].items():
                    batter_history[batter].append((date, vv))
                league_events[year] += stats
    f = pd.DataFrame(rows)
    if len(f):
        present = f.history_max_date.notna()
        if not (pd.to_datetime(f.loc[present, "history_max_date"]) < f.loc[present, "date"]).all():
            raise ValueError("Feature history is not strictly prior-date")
    return f


def nb_pmf(mu, alpha, tail=1e-12):
    if not np.isfinite(mu) or mu <= 0:
        raise ValueError("Nonpositive/nonfinite model mean")
    size, p = 1 / alpha, 1 / (1 + alpha * mu)
    stop = int(nbinom.ppf(1 - tail, size, p)) + 1
    if stop > 4096:
        raise ValueError("Excessive PMF support; inspect model instead of clipping")
    out = nbinom.pmf(np.arange(stop), size, p)
    if 1 - out.sum() > tail * 2:
        raise ValueError("PMF omitted mass exceeds tolerance")
    return out


def score_distribution(away, home):
    joint = away[:, None] * home[None, :]
    aa, hh = np.indices(joint.shape)
    total, margin = aa + hh, hh - aa
    mass = float(joint.sum())
    if abs(mass - 1) > 1e-9:
        raise ValueError("Joint distribution does not sum to one")
    return {"away": float(joint[margin < 0].sum()), "draw": float(joint[margin == 0].sum()),
            "home": float(joint[margin > 0].sum()),
            "home_minus1p5": float(joint[margin >= 2].sum()),
            "under4p5": float(joint[total <= 4].sum()),
            "under8p5": float(joint[total <= 8].sum()),
            "win1loss": [float(joint[margin <= -2].sum()), float(joint[np.abs(margin) <= 1].sum()),
                         float(joint[margin >= 2].sum())]}


def probability_losses(prob, label):
    q = np.asarray(prob, float)
    if not np.isfinite(q).all() or q.min() < -1e-10 or abs(q.sum() - 1) > 1e-8:
        raise ValueError("Invalid outcome probability vector")
    truth = np.eye(len(q))[label]
    return -math.log(max(float(q[label]), 1e-15)), float(np.sum((q - truth) ** 2))


def evaluate_game(mu, actual, cfg):
    # Full team counts are a convolution: F5 + nonnegative late, on every path.
    pmf5 = [nb_pmf(mu["f5"][i], cfg["nb_alpha_f5"], cfg["pmf_tail_tolerance"]) for i in (0, 1)]
    pmfl = [nb_pmf(mu["late"][i], cfg["nb_alpha_late"], cfg["pmf_tail_tolerance"]) for i in (0, 1)]
    pmff = [np.convolve(pmf5[i], pmfl[i]) for i in (0, 1)]
    if any(mu["f5"][i] + mu["late"][i] < mu["f5"][i] for i in (0, 1)):
        raise ValueError("Phase-inconsistent mean")
    f5, full = score_distribution(*pmf5), score_distribution(*pmff)
    a5, h5 = actual["f5"]
    af, hf = actual["full"]
    label = int(h5 > a5) * 2 if h5 != a5 else 1
    ll5, bs5 = probability_losses([f5["away"], f5["draw"], f5["home"]], label)
    ans = {"f5_wdl_logloss": ll5, "f5_wdl_brier": bs5,
           "f5_wdl_hit": float(np.argmax([f5["away"], f5["draw"], f5["home"]]) == label)}
    if af != hf:
        den = full["away"] + full["home"]
        ll, bs = probability_losses([full["away"] / den, full["home"] / den], int(hf > af))
        ans.update(full_decided_logloss=ll, full_decided_brier=bs,
                   full_decided_hit=float((full["home"] > full["away"]) == (hf > af)))
    for name, pr, y, line in [("f5_total", f5, a5 + h5, 4.5), ("full_total", full, af + hf, 8.5)]:
        u = pr["under4p5" if line == 4.5 else "under8p5"]
        ll, bs = probability_losses([u, 1 - u], int(y > line))
        ans.update({name + "_logloss": ll, name + "_brier": bs})
    for name, pr, margin in [("f5_hcap", f5, h5 - a5), ("full_hcap", full, hf - af)]:
        p = pr["home_minus1p5"]
        ll, bs = probability_losses([1 - p, p], int(margin >= 2))
        ans.update({name + "_logloss": ll, name + "_brier": bs})
    ll, bs = probability_losses(full["win1loss"], 0 if hf - af <= -2 else 2 if hf - af >= 2 else 1)
    ans.update(full_win1loss_logloss=ll, full_win1loss_brier=bs)
    for name, pp, yy in [("f5", pmf5, [a5, h5]), ("full", pmff, [af, hf])]:
        ans[name + "_count_nll"] = float(np.mean([-math.log(max(float(p[y]) if y < len(p) else 0., 1e-15))
                                                  for p, y in zip(pp, yy)]))
        means = mu["f5"] if name == "f5" else np.asarray(mu["f5"]) + mu["late"]
        ans[name + "_mean_bias"] = float(np.mean(means) - np.mean(yy))
    ans.update(f5_p_away=f5["away"], f5_p_draw=f5["draw"], f5_p_home=f5["home"],
               full_p_home_decided=full["home"] / (full["away"] + full["home"]))
    return ans


def model_columns(cfg):
    base = cfg["baselines"]["baseline_team"]
    return {**cfg["baselines"], **{k: base + v for k, v in cfg["additions_to_baseline_team_individually"].items()}}


def monthly_forecasts(frame, cfg):
    f = frame.copy()
    f["month"] = f.date.dt.to_period("M")
    f = f[f.date >= pd.to_datetime(f.date.dt.year.astype(str) + "-" + cfg["training_start_month_day"])]
    logs, artifacts, coverage = [], {}, []
    for month in sorted(f.month.unique()):
        tr = f[f.date < month.start_time]
        te = f[f.month == month]
        ntrain = tr.game_pk.nunique()
        if ntrain < cfg["minimum_training_games"]:
            coverage.append({"month": str(month), "status": "warmup", "prior_training_games": ntrain})
            continue
        if tr.date.max() >= te.date.min():
            raise ValueError("Training reaches forecast date")
        coverage.append({"month": str(month), "status": "evaluated", "train_games": ntrain,
                         "test_games": int(te.game_pk.nunique())})
        for name, features in model_columns(cfg).items():
            for phase in PHASES:
                train, test = tr[tr.phase == phase], te[te.phase == phase]
                scaler = StandardScaler()
                xt = scaler.fit_transform(train[features].to_numpy(float))
                xx = scaler.transform(test[features].to_numpy(float))
                model = PoissonRegressor(alpha=cfg["ridge_alpha"], max_iter=1000, tol=1e-9)
                # Weighted scaled target is exactly the Poisson objective with log(offset).
                model.fit(xt, train.target.to_numpy(float) / train.offset, sample_weight=train.offset)
                pred = test.offset.to_numpy(float) * model.predict(xx)
                if not np.isfinite(pred).all() or pred.min() <= 0:
                    raise ValueError("Invalid count mean")
                z = test[["game_pk", "date", "team", "is_home", "phase", "target"]].copy()
                z["mu"], z["model"], z["train_max_date"] = pred, name, str(train.date.max().date())
                logs.append(z)
                artifacts[f"{month}/{name}/{phase}"] = {"features": features,
                    "intercept": float(model.intercept_), "coef": model.coef_.tolist(),
                    "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
                    "train_max_date": str(train.date.max().date()), "n_train_games": ntrain}
        print(f"MONTH {month}: train {ntrain} games, forecast {te.game_pk.nunique()} games", flush=True)
    if not logs:
        raise ValueError("No eligible evaluation months")
    return pd.concat(logs, ignore_index=True), artifacts, coverage


def game_predictions(logs, cfg):
    rows = []
    for (pk, name), d in logs.groupby(["game_pk", "model"], sort=True):
        mu, actual = {}, {}
        if len(d) != 4:
            raise ValueError("Each model/game needs both teams and phases")
        for phase in PHASES:
            v = d[d.phase == phase].sort_values("is_home")
            mu[phase] = v.mu.to_numpy(float)
            actual[phase] = v.target.to_numpy(int)
        actual["full"] = actual["f5"] + actual["late"]
        result = evaluate_game(mu, actual, cfg)
        result.update(game_pk=pk, model=name, date=str(d.date.iloc[0].date()),
                      f5_away_mu=mu["f5"][0], f5_home_mu=mu["f5"][1],
                      late_away_mu=mu["late"][0], late_home_mu=mu["late"][1],
                      full_away_mu=sum(mu[p][0] for p in PHASES),
                      full_home_mu=sum(mu[p][1] for p in PHASES),
                      f5_away_actual=actual["f5"][0], f5_home_actual=actual["f5"][1],
                      full_away_actual=actual["full"][0], full_home_actual=actual["full"][1],
                      train_max_date=d.train_max_date.iloc[0])
        rows.append(result)
    out = pd.DataFrame(rows)
    if out.duplicated(["game_pk", "model"]).any():
        raise ValueError("Duplicate forecast game/model")
    model_sets = [set(g.game_pk) for _, g in out.groupby("model")]
    if any(v != model_sets[0] for v in model_sets):
        raise ValueError("Ablations evaluated on different games")
    return out


def paired_ci(deltas, dates, cfg, comparisons=1):
    """Resample dates as clusters, weighting by their original game counts."""
    q = pd.DataFrame({"delta": deltas, "date": dates}).groupby("date").delta.agg(["sum", "count"])
    rng = np.random.default_rng(cfg["seed"])
    idx = rng.integers(0, len(q), size=(cfg["bootstrap_repetitions"], len(q)))
    bootstrap = q["sum"].to_numpy()[idx].sum(axis=1) / q["count"].to_numpy()[idx].sum(axis=1)
    alpha = .05 / max(1, comparisons)
    return {"delta": float(np.mean(deltas)), "ci95": np.quantile(bootstrap, [.025, .975]).tolist(),
            "familywise_ci": np.quantile(bootstrap, [alpha / 2, 1 - alpha / 2]).tolist(),
            "date_clusters": len(q), "n_games": len(deltas)}


def summarize(pred, cfg):
    metrics = [x for x in pred.columns if x.endswith(("logloss", "brier", "count_nll", "mean_bias", "_hit"))]
    out = {"aggregate": {}, "by_year": {}, "by_month": {}, "paired_vs_team": {}}
    for name, g in pred.groupby("model"):
        out["aggregate"][name] = {"games": len(g), "decided_games": int(g.full_decided_logloss.notna().sum()), **{k: float(g[k].mean()) for k in metrics}}
        for year, yy in g.groupby(g.date.str[:4]):
            out["by_year"].setdefault(year, {})[name] = {"games": len(yy), **{k: float(yy[k].mean()) for k in metrics}}
        for month, yy in g.groupby(g.date.str[:7]):
            out["by_month"].setdefault(month, {})[name] = {"games": len(yy), **{k: float(yy[k].mean()) for k in metrics}}
    base = pred[pred.model == "baseline_team"].set_index("game_pk")
    primary_family = len(cfg["additions_to_baseline_team_individually"]) * len(cfg["primary_metrics"])
    for name, g in pred.groupby("model"):
        if name == "baseline_team":
            continue
        q = g.set_index("game_pk").join(base[metrics], rsuffix="_base", validate="one_to_one")
        out["paired_vs_team"][name] = {}
        for key in metrics:
            qq = q[[key, key + "_base", "date"]].dropna()
            res = paired_ci((qq[key] - qq[key + "_base"]).to_numpy(), qq.date.to_numpy(), cfg,
                            primary_family if key in cfg["primary_metrics"] else 1)
            res["primary_familywise_comparisons"] = primary_family if key in cfg["primary_metrics"] else None
            out["paired_vs_team"][name][key] = res
    return out


def render_report(metrics, manifest, coverage, cfg, protocol_hash):
    a = metrics["aggregate"]
    title = "# 재현 가능한 KBO 기준 모형과 개별 변수 추가 실험\n"
    lines = [title, f"모형 {cfg['version']} / 상태: 회고적 연구, Champion 승격 없음.",
             f"프로토콜 SHA256: \x60{protocol_hash}\x60", "",
             "같은 경기에서 팀 기준 모형에 각 정보 묶음을 독립적으로 추가했다. 낮은 log loss가 좋다.", "",
             "|모형|경기 수 F5/Full승패|F5 3방향 log loss|Full 조건부 승패 log loss|F5 득점 NLL|Full 득점 NLL|",
             "|---|---:|---:|---:|---:|---:|"]
    for name, v in a.items():
        lines.append(f"|{name}|{v['games']}/{v['decided_games']}|{v['f5_wdl_logloss']:.5f}|{v['full_decided_logloss']:.5f}|{v['f5_count_nll']:.5f}|{v['full_count_nll']:.5f}|")
    lines += ["", "## 팀 기준 대비 차이", "", "음수는 개선이다. 날짜 군집 95% 구간과 주 지표 전체 비교에 대한 Bonferroni 구간을 함께 보존했다.", "",
              "|추가 변수|F5 Δ log loss (95% 구간)|Full Δ log loss (95% 구간)|", "|---|---:|---:|"]
    for name, v in metrics["paired_vs_team"].items():
        cols = []
        for k in cfg["primary_metrics"]:
            z = v[k]
            cols.append(f"{z['delta']:+.5f} [{z['ci95'][0]:+.5f}, {z['ci95'][1]:+.5f}]")
        lines.append(f"|{name}|{'|'.join(cols)}|")
    lines += ["", "## 해석", "", "- 모든 모델은 같은 경기와 같은 월별 학습 마감에서 비교했다.",
              "- Full 승패는 실제 무승부를 제외한 조건부 두 방향이다. F5는 무승부를 포함한 세 방향이다.",
              "- F5와 후반은 비음수 득점 분포이고 Full은 두 분포의 합성이다. 각 팀 Full 평균은 항상 F5 이상이다.",
              "- 이미 연구자가 관찰한 역사 자료를 사용했다. 새 독립 전향 성능이나 당시 실제 운용 성적이 아니다.",
              "- 좋은 결과가 나온 변수들을 사후 조합하지 않았다. 실제 베팅 가격·수익률을 계산하지 않았다.",
              "- starter/lineup identity는 경기 후 PBP 재구성이다. 독립적으로 보존된 발표 시각이 없으므로 실전 사전예측 자격을 주장하지 않는다.",
              "- 2024·2025는 9월 30일까지, 2026은 10월 12일까지다. 모든 시즌 전체 경기를 포함한다는 의미는 아니다.",
              "- 차기 검증은 다음 미관찰 시즌의 실제 발표 라인업·선발·입력 시각을 동결해 진행한다. 기존 BP30 게이트는 별개로 유지한다.", "", "## 입력", "", "\x60\x60\x60json", json.dumps(manifest, ensure_ascii=False, indent=2), "\x60\x60\x60", "",
              "## 월별 범위", "", "\x60\x60\x60json", json.dumps(coverage, ensure_ascii=False, indent=2), "\x60\x60\x60", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--protocol", default=str(ROOT / "research/repro_baseline/protocol.json"))
    p.add_argument("--out", required=True)
    args = p.parse_args()
    config_path, out = Path(args.protocol), Path(args.out)
    cfg = json.loads(config_path.read_text())
    protocol_hash = sha_file(config_path)
    raw, manifest = load_sources(args.inputs, cfg)
    games, exclusions = prepare_games(raw)
    print(f"SCORES games={len(games)} exclusions={len(exclusions)}", flush=True)
    features = build_features(games, cfg)
    print(f"FEATURES rows={len(features)}", flush=True)
    logs, fits, coverage = monthly_forecasts(features, cfg)
    pred = game_predictions(logs, cfg)
    metrics = summarize(pred, cfg)
    out.mkdir(parents=True, exist_ok=True)
    features.to_csv(out / "features.csv.gz", index=False)
    pred.to_csv(out / "predictions.csv", index=False, float_format="%.10g")
    run_manifest = {"protocol_sha256": protocol_hash, "sources": manifest, "exclusions": exclusions,
                    "coverage": coverage, "versions": {"python": platform.python_version(), "numpy": np.__version__,
                    "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
                    "status": cfg["status"], "eligible_games": int(pred.game_pk.nunique()),
                    "model_count": int(pred.model.nunique()), "official_activation": False}
    for file, data in [("metrics.json", metrics), ("fitted_models.json", fits), ("manifest.json", run_manifest)]:
        (out / file).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (out / "REPORT.md").write_text(render_report(metrics, manifest, coverage, cfg, protocol_hash))
    print("BASELINE_RESULT=" + json.dumps({"games": run_manifest["eligible_games"], "aggregate": metrics["aggregate"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
