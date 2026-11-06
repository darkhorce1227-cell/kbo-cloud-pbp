"""Time-ordered offense/platoon research; no changes to any operational model.

Inputs are retrospective PBP first-nine/first-pitcher proxies. Current official
profiles provide STATIC handedness only, never historical roster availability.
All statistical state is updated after the whole target date is predicted.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from scipy.stats import nbinom, poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "research/offense-platoon-v2/study_plan.json"
KEY = ["game_pk", "at_bat_number", "pitch_number"]
EVENT_NAMES = ["onbase", "tb", "k"]
BASE_FEATURES = [
    "off_long", "off30", "opp_long", "opp30", "is_home", "park",
    "opponent_onbase", "opponent_tb", "opponent_k", "sp_expected_ip",
]
DELTA_FEATURES = ["lineup_" + event for event in EVENT_NAMES]
PLATOON_FEATURES = ["batter_platoon_" + event for event in EVENT_NAMES] + [
    "pitcher_specialist_" + event for event in EVENT_NAMES
]
MODELS = {
    "context": BASE_FEATURES,
    "lineup_delta": BASE_FEATURES + DELTA_FEATURES,
    "lineup_platoon": BASE_FEATURES + DELTA_FEATURES + PLATOON_FEATURES,
}


def player_id(value) -> str:
    return str(int(value))


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sources(paths: list[Path], plan: dict):
    frames, audit = [], []
    for path in paths:
        year = next((str(y) for y in (2024, 2025, 2026) if str(y) in path.name), None)
        if year is None:
            raise ValueError("Source filename must identify season: " + path.name)
        actual_sha = file_sha(path)
        if actual_sha != plan["source_sha256"][year]:
            raise ValueError("Source hash mismatch for season " + year)
        x = pd.read_csv(path)
        if x.duplicated(KEY).any():
            raise ValueError("Duplicate stable pitch keys in " + path.name)
        raw_rows, raw_games = len(x), x.game_pk.nunique()
        x["game_pk"] = x.game_pk.astype(str)
        x["game_date"] = pd.to_datetime(x.game_date).dt.normalize()
        x = x[x.game_date.between(year + "-01-01", plan["regular_season_end"][year])].copy()
        frames.append(x)
        audit.append({
            "file": path.name, "sha256": actual_sha, "raw_rows": raw_rows,
            "raw_games": int(raw_games), "regular_rows": len(x),
            "regular_games": int(x.game_pk.nunique()), "duplicate_stable_keys": 0,
        })
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(KEY).any():
        raise ValueError("Duplicate stable keys across files")
    return combined.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"], kind="stable"), audit


def prepare_games(raw: pd.DataFrame):
    raw = raw.copy()
    raw["inning_topbot"] = raw.inning_topbot.astype(str).str.lower().replace({"bot": "bottom"})
    if not raw.inning_topbot.isin(["top", "bottom"]).all():
        raise ValueError("Unknown inning half encoding")
    first_pitchers = {}
    for (pk, half), q in raw.groupby(["game_pk", "inning_topbot"], sort=False):
        first_pitchers[(str(pk), str(half))] = player_id(q.iloc[0].pitcher)
    pa = raw.drop_duplicates(["game_pk", "at_bat_number"], keep="last").copy()
    pa["batting_team"] = np.where(pa.inning_topbot.eq("top"), pa.away_team, pa.home_team)
    pa["pitching_team"] = np.where(pa.inning_topbot.eq("top"), pa.home_team, pa.away_team)
    pa["batter_id"] = pa.batter.map(player_id)
    pa["pitcher_id"] = pa.pitcher.map(player_id)
    event = pa.events.fillna("").astype(str)
    pa["onbase"] = event.isin(["single", "double", "triple", "home_run", "walk", "intent_walk", "hit_by_pitch"]).astype(float)
    pa["tb"] = event.map({"single": 1, "double": 2, "triple": 3, "home_run": 4}).fillna(0).astype(float)
    pa["k"] = event.isin(["strikeout", "strikeout_double_play"]).astype(float)
    pa["outs_made"] = (pa.post_outs - pa.outs_when_up).clip(lower=0, upper=3).fillna(0)
    games, failures = [], []
    for pk, q in pa.groupby("game_pk", sort=False):
        q = q.sort_values("at_bat_number", kind="stable")
        early = q[q.inning <= 5]
        reason = None
        if early.empty or early.inning.max() != 5:
            reason = "NO_COMPLETE_F5"
        elif not ((q.inning >= 6).any() or ((early.inning.eq(5) & early.inning_topbot.eq("bottom") & early.post_outs.eq(3)).any())):
            reason = "FIFTH_COMPLETION_NOT_PROVED"
        score_cols = ["post_away_score", "post_home_score"]
        if q[score_cols].isna().any().any() or (q[score_cols].diff().dropna() < 0).any().any():
            reason = "SCOREBOARD_NONMONOTONE_OR_MISSING"
        if reason:
            failures.append({"game_pk": str(pk), "reason": reason})
            continue
        score5 = early.iloc[-1][score_cols].to_numpy(float)
        score_full = q.iloc[-1][score_cols].to_numpy(float)
        if (score_full < score5).any() or (score5 < 0).any():
            raise ValueError("Negative score increment: " + str(pk))
        away, home = str(q.away_team.iloc[0]), str(q.home_team.iloc[0])
        lineups = {}
        for team in (away, home):
            first_nine = q[q.batting_team.eq(team)].drop_duplicates("batter_id", keep="first").head(9)
            lineups[team] = first_nine.batter_id.tolist() if len(first_nine) == 9 and first_nine.inning.max() <= 4 else None
        games.append({
            "pk": str(pk), "date": pd.Timestamp(q.game_date.iloc[0]), "away": away, "home": home,
            "f5": score5, "late": score_full - score5, "full": score_full,
            "lineups": lineups,
            "starters": {home: first_pitchers[(str(pk), "top")], away: first_pitchers[(str(pk), "bottom")]},
            "pa": q,
        })
    games.sort(key=lambda g: (g["date"], g["pk"]))
    return games, {"valid_score_games": len(games), "score_exclusions": failures,
                   "missing_first_nine_games": sum(any(v is None for v in g["lineups"].values()) for g in games)}


def rates(counts: np.ndarray, prior: np.ndarray, pseudo: float) -> np.ndarray:
    return (counts[1:] + pseudo * prior) / (counts[0] + pseudo)


def relative_prior(overall: np.ndarray, cell: np.ndarray, marginal: np.ndarray) -> np.ndarray:
    """League matchup odds shift for bounded events, rate ratio for total bases."""
    a, b, c = np.clip(overall, 1e-6, 1 - 1e-6), np.clip(cell, 1e-6, 1 - 1e-6), np.clip(marginal, 1e-6, 1 - 1e-6)
    odds = (a / (1 - a)) * (b / (1 - b)) / (c / (1 - c))
    out = odds / (1 + odds)
    out[1] = overall[1] * cell[1] / max(marginal[1], 1e-6)
    return out


class History:
    def __init__(self, plan: dict, hands: dict):
        self.plan, self.hands = plan, hands
        self.bat = defaultdict(lambda: np.zeros(4))
        self.bat_hand = defaultdict(lambda: np.zeros(4))
        self.pit = defaultdict(lambda: np.zeros(4))
        self.pit_side = defaultdict(lambda: np.zeros(4))
        self.league = np.zeros(4)
        self.league_hand = defaultdict(lambda: np.zeros(4))
        self.league_side = defaultdict(lambda: np.zeros(4))
        self.league_cell = defaultdict(lambda: np.zeros(4))
        self.stands = defaultdict(Counter)
        self.starts = defaultdict(list)
        self.team = defaultdict(list)
        self.bp = defaultdict(list)
        self.max_date = None

    def hand(self, pitcher: str) -> str:
        r = self.hands.get(pitcher, {})
        return r.get("throws", "UNKNOWN") if r.get("status") == "VERIFIED_PROFILE" else "UNKNOWN"

    def side(self, batter: str, pitcher_hand: str) -> str | None:
        cnt = self.stands.get(batter, {})
        if not cnt:
            return None
        if cnt.get("L", 0) >= 5 and cnt.get("R", 0) >= 5:
            return "R" if pitcher_hand == "L" else "L"
        return max(cnt, key=cnt.get)

    def league_rates(self) -> np.ndarray:
        return rates(self.league, np.array([.33, .38, .20]), 120)

    def batter_rates(self, batter: str) -> np.ndarray:
        return rates(self.bat[batter], self.league_rates(), self.plan["overall_player_pseudo_pa"])

    def pitcher_rates(self, pitcher: str) -> np.ndarray:
        return rates(self.pit[pitcher], self.league_rates(), self.plan["overall_player_pseudo_pa"])

    def batter_match(self, batter: str, hand: str) -> np.ndarray:
        overall = self.batter_rates(batter)
        side = self.side(batter, hand)
        if hand not in ("L", "R") or side not in ("L", "R"):
            return overall
        lg = self.league_rates()
        marginal = rates(self.league_side[side], lg, self.plan["league_split_pseudo_pa"])
        cell = rates(self.league_cell[(hand, side)], marginal, self.plan["league_split_pseudo_pa"])
        prior = relative_prior(overall, cell, marginal)
        return rates(self.bat_hand[(batter, hand)], prior, self.plan["split_pseudo_pa"])

    def pitcher_residual(self, pitcher: str, side: str | None) -> np.ndarray:
        hand = self.hand(pitcher)
        if hand not in ("L", "R") or side not in ("L", "R"):
            return np.zeros(3)
        overall = self.pitcher_rates(pitcher)
        lg = self.league_rates()
        marginal = rates(self.league_hand[hand], lg, self.plan["league_split_pseudo_pa"])
        cell = rates(self.league_cell[(hand, side)], marginal, self.plan["league_split_pseudo_pa"])
        prior = relative_prior(overall, cell, marginal)
        # The league matchup component is already in batter_match; retain only
        # this pitcher's deviation from that prior to prevent double counting.
        split = rates(self.pit_side[(pitcher, side)], prior, self.plan["split_pseudo_pa"])
        return split - prior

    def close_day(self, games: list[dict]):
        for g in games:
            date, year = g["date"], int(g["date"].year)
            for index, team, opp in ((0, g["away"], g["home"]), (1, g["home"], g["away"])):
                self.team[(year, team)].append({
                    "date": date, "lineup": g["lineups"][team], "is_home": index,
                    "f5_for": g["f5"][index], "f5_against": g["f5"][1-index],
                    "late_for": g["late"][index], "late_against": g["late"][1-index],
                    "full_total": float(g["full"].sum()),
                })
            q = g["pa"]
            for r in q.itertuples(index=False):
                v = np.array([1., r.onbase, r.tb, r.k])
                batter, pitcher, side = r.batter_id, r.pitcher_id, str(r.stand)
                hand = self.hand(pitcher)
                self.bat[batter] += v
                self.pit[pitcher] += v
                self.league += v
                if side in ("L", "R"):
                    self.stands[batter][side] += 1
                    self.pit_side[(pitcher, side)] += v
                    self.league_side[side] += v
                if hand in ("L", "R"):
                    self.bat_hand[(batter, hand)] += v
                    self.league_hand[hand] += v
                    if side in ("L", "R"):
                        self.league_cell[(hand, side)] += v
            for team in (g["away"], g["home"]):
                qq = q[q.pitching_team.eq(team)]
                starter = g["starters"][team]
                sp = qq[qq.pitcher_id.eq(starter)]
                self.starts[starter].append(float(sp.outs_made.sum()) / 3.)
                bp = qq[~qq.pitcher_id.eq(starter)]
                v = np.r_[len(bp), bp[EVENT_NAMES].sum().to_numpy(float)]
                hc = Counter(self.hand(p) for p in bp.pitcher_id)
                self.bp[(year, team)].append({"date": date, "v": v, "hands": hc})
        self.max_date = games[0]["date"]


def build_rows(games: list[dict], plan: dict, hands: dict) -> pd.DataFrame:
    hist = History(plan, hands)
    weights = np.asarray(plan["slot_weights"], float)
    weights /= weights.sum()
    rows = []
    by_date = defaultdict(list)
    for g in games:
        by_date[g["date"]].append(g)
    for date, day in sorted(by_date.items()):
        year = int(date.year)
        recent_start = date - pd.Timedelta(days=plan["league_recent_days"])
        all_entries = [e for (y, _), es in hist.team.items() if y == year for e in es]
        lg_long, lg30 = {}, {}
        for phase in ("f5", "late"):
            values = [e[phase + "_for"] for e in all_entries]
            recent = [e[phase + "_for"] for e in all_entries if e["date"] >= recent_start]
            lg_long[phase] = float(np.mean(values)) if values else 2.4
            pseudo = plan["league_recent_pseudo_team_games"]
            lg30[phase] = (sum(recent) + pseudo * lg_long[phase]) / (len(recent) + pseudo)
        year_starts = [ip for ips in hist.starts.values() for ip in ips]
        league_ip = float(np.mean(year_starts)) if year_starts else 5.

        @lru_cache(None)
        def hitter(batter):
            return hist.batter_rates(batter)

        @lru_cache(None)
        def hitter_match(batter, hand):
            return hist.batter_match(batter, hand)

        for g in day:
            if any(ids is None for ids in g["lineups"].values()):
                continue
            assert hist.max_date is None or hist.max_date < date
            for index, team, opponent in ((0, g["away"], g["home"]), (1, g["home"], g["away"])):
                my = hist.team[(year, team)]
                opp = hist.team[(year, opponent)]
                lineup = g["lineups"][team]
                current = np.average([hitter(b) for b in lineup], axis=0, weights=weights)
                past_nines = [e["lineup"] for e in my if e["lineup"] is not None][-plan["normal_lineup_last_games"]:]
                normal = np.mean([np.average([hitter(b) for b in nine], weights=weights, axis=0) for nine in past_nines], axis=0) if past_nines else current
                lineup_delta = current - normal
                starter = g["starters"][opponent]
                sp_hand = hist.hand(starter)
                starts = hist.starts[starter]
                ps = plan["starter_pseudo_starts"]
                eip = (sum(starts) + ps * league_ip) / (len(starts) + ps)
                sp_rates = hist.pitcher_rates(starter)
                bp_history = [e for e in hist.bp[(year, opponent)] if e["date"] >= recent_start]
                bp_counts = sum((e["v"] for e in bp_history), np.zeros(4))
                bp_rates = rates(bp_counts, hist.league_rates(), plan["overall_player_pseudo_pa"])
                bp_hands = Counter()
                for entry in bp_history:
                    bp_hands.update(entry["hands"])
                known_hands = hist.league_hand["L"][0] + hist.league_hand["R"][0]
                league_left = hist.league_hand["L"][0] / known_hands if known_hands else .30
                p_left = (bp_hands["L"] + plan["bp_hand_pseudo_bf"] * league_left) / (bp_hands["L"] + bp_hands["R"] + plan["bp_hand_pseudo_bf"])
                split_l = np.average([hitter_match(b, "L") - hitter(b) for b in lineup], weights=weights, axis=0)
                split_r = np.average([hitter_match(b, "R") - hitter(b) for b in lineup], weights=weights, axis=0)
                sp_split = split_l if sp_hand == "L" else split_r if sp_hand == "R" else np.zeros(3)
                bp_split = p_left * split_l + (1 - p_left) * split_r
                specialist = np.average([hist.pitcher_residual(starter, hist.side(b, sp_hand)) for b in lineup], weights=weights, axis=0)
                home_history = [e for e in hist.team[(year, g["home"])] if e["is_home"]]
                league_full = lg_long["f5"] + lg_long["late"]
                park = (sum(e["full_total"] for e in home_history) + 2 * plan["park_pseudo_games"] * league_full) / (2 * (len(home_history) + plan["park_pseudo_games"]) * league_full)
                for phase in ("f5", "late"):
                    lg = max(lg30[phase], 1e-6)
                    def ratio(entries, kind, recent_only=False):
                        ent = [e for e in entries if not recent_only or e["date"] >= recent_start]
                        pseudo = plan["team_pseudo_games"]
                        return (sum(e[phase + "_" + kind] for e in ent) + pseudo * lg) / ((len(ent) + pseudo) * lg)
                    exposure = min(eip / 5., 1.) if phase == "f5" else min(max(eip - 5., 0.) / 4., 1.)
                    opp_events = exposure * sp_rates + (1 - exposure) * bp_rates
                    platoon = exposure * sp_split + (1 - exposure) * bp_split
                    r = {
                        "game_pk": g["pk"], "game_date": date, "year": year,
                        "team": team, "opponent": opponent, "is_home": index,
                        "phase": phase, "target": float(g[phase][index]), "offset": lg,
                        "max_feature_date": hist.max_date,
                        "off_long": ratio(my, "for"), "off30": ratio(my, "for", True),
                        "opp_long": ratio(opp, "against"), "opp30": ratio(opp, "against", True),
                        "park": park, "sp_expected_ip": eip, "sp_exposure": exposure,
                        "opposing_starter": starter, "opposing_throws": sp_hand,
                        "batter_history_pa": sum(hist.bat[b][0] for b in lineup),
                        "pitcher_split_pa": sum(hist.pit_side[(starter, s)][0] for s in ("L", "R")),
                        "normal_nines": len(past_nines),
                    }
                    for j, event in enumerate(EVENT_NAMES):
                        r["opponent_" + event] = float(opp_events[j])
                        r["lineup_" + event] = float(lineup_delta[j])
                        r["batter_platoon_" + event] = float(platoon[j])
                        r["pitcher_specialist_" + event] = float(exposure * specialist[j])
                    rows.append(r)
        hist.close_day(day)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    if not (frame.max_feature_date.isna() | (frame.max_feature_date < frame.game_date)).all():
        raise ValueError("Feature timestamp leakage")
    if not frame.groupby(["game_pk", "phase"]).size().eq(2).all():
        raise ValueError("Unpaired game rows")
    if not np.isfinite(frame[BASE_FEATURES + DELTA_FEATURES + PLATOON_FEATURES + ["offset"]].to_numpy(float)).all():
        raise ValueError("Nonfinite model input")
    return frame


def fit_phase(train: pd.DataFrame, test: pd.DataFrame, features: list[str], alpha: float):
    scaler = StandardScaler().fit(train[features])
    model = PoissonRegressor(alpha=alpha, max_iter=1500, tol=1e-9)
    # Rate + exposure weight equals a Poisson count likelihood with log offset.
    model.fit(scaler.transform(train[features]), train.target / train.offset, sample_weight=train.offset)
    mu = model.predict(scaler.transform(test[features])) * test.offset.to_numpy(float)
    if not np.isfinite(mu).all() or (mu <= 0).any():
        raise ValueError("Invalid learned run means")
    artifact = {"features": features, "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(), "intercept": float(model.intercept_),
                "coef": model.coef_.tolist(), "alpha": alpha}
    return mu, artifact


def nb_pmf(mu: np.ndarray, alpha: float) -> np.ndarray:
    r = 1. / alpha
    p = r / (r + mu)
    maximum = int(np.max(nbinom.ppf(1 - 1e-11, r, p)))
    out = nbinom.pmf(np.arange(maximum + 1)[None, :], r, p[:, None])
    if np.max(1 - out.sum(axis=1)) > 2e-11:
        raise ValueError("PMF truncation error")
    return out


def phase_distributions(f5_mu: np.ndarray, late_mu: np.ndarray, plan: dict):
    f5 = nb_pmf(f5_mu, plan["alpha_nb2_f5"])
    late = nb_pmf(late_mu, plan["alpha_nb2_late_research_only"])
    full = fftconvolve(f5, late, axes=1)
    if full.min() < -1e-13:
        raise ValueError("Negative convolution probability")
    full = np.maximum(full, 0)  # FFT roundoff only, never a run-mean repair.
    if np.max(1 - full.sum(axis=1)) > 5e-11:
        raise ValueError("Full score PMF truncation error")
    return f5, full


def wdl_from_pmf(away: np.ndarray, home: np.ndarray) -> np.ndarray:
    if away.shape != home.shape:
        raise ValueError("Different score supports")
    away_win = np.sum(away * (np.cumsum(home, axis=1) - home), axis=1)
    draw = np.sum(away * home, axis=1)
    home_win = np.sum(home * (np.cumsum(away, axis=1) - away), axis=1)
    probs = np.c_[away_win, draw, home_win]
    if np.max(np.abs(probs.sum(axis=1) - 1)) > 1e-9:
        raise ValueError("WDL probabilities do not sum to one")
    return probs


def paired_ci(delta: np.ndarray, indices: np.ndarray) -> dict:
    means = delta[indices].mean(axis=1)
    return {"mean_delta": float(delta.mean()),
            "ci95": np.quantile(means, [.025, .975]).tolist(),
            "n_games": len(delta), "unit": "game", "negative_is_better": True}


def evaluate(frame: pd.DataFrame, plan: dict, output: Path):
    result = {
        "version": plan["version"], "status": plan["status"],
        "evidence_class": plan["evidence_class"], "study_plan_sha256": file_sha(PLAN_PATH),
        "count_distribution": "NB2 F5 + independent nonnegative NB2 late increments; Full=convolution",
        "full_wdl_status": "FINAL_SCORE_DIAGNOSTIC_NOT_OFFICIAL_GAME_ENDING_RULE_MODEL",
        "official_champion_comparison": "UNAVAILABLE_NO_MATCHED_REPRODUCIBLE_CHAMPION",
        "folds": [],
    }
    all_predictions, all_games = [], []
    fitted = {}
    for fold in plan["folds"]:
        year = fold["test_year"]
        tr = frame[frame.year.isin(fold["train_years"]) & (frame.game_date.dt.strftime("%m-%d") >= plan["first_evaluation_month_day"])]
        te = frame[frame.year.eq(year) & (frame.game_date.dt.strftime("%m-%d") >= plan["first_evaluation_month_day"])]
        if tr.empty or te.empty:
            continue
        if tr.game_date.max() >= te.game_date.min():
            raise ValueError("Train/test date overlap")
        te = te.sort_values(["game_pk", "is_home", "phase"], kind="stable").copy()
        keyrows = te[te.phase.eq("f5")].sort_values(["game_pk", "is_home"])
        keys = keyrows.game_pk.unique().tolist()
        n_games = len(keys)
        if len(keyrows) != 2 * n_games:
            raise ValueError("Unmatched heldout team scores")
        rng = np.random.default_rng(plan["bootstrap"]["seed"] + year)
        indices = rng.integers(0, n_games, size=(plan["bootstrap"]["resamples"], n_games))
        summary = {
            "train_years": fold["train_years"], "test_year": year,
            "train_games": int(tr.game_pk.nunique()), "test_games": n_games,
            "train_max_date": str(tr.game_date.max().date()),
            "test_min_date": str(te.game_date.min().date()),
            "unknown_opponent_hand_teamgames": int(keyrows.opposing_throws.eq("UNKNOWN").sum()),
            "models": {}, "paired_differences": {},
        }
        per_game_losses = {}
        for name, features in MODELS.items():
            phase_mu = {}
            for phase in ("f5", "late"):
                a = tr[tr.phase.eq(phase)]
                b = te[te.phase.eq(phase)].sort_values(["game_pk", "is_home"])
                mu, artifact = fit_phase(a, b, features, plan["alpha_poisson_ridge"])
                phase_mu[phase] = mu
                fitted[f"{year}/{name}/{phase}"] = artifact
                pp = b[["game_pk", "game_date", "team", "is_home", "phase", "target"]].copy()
                pp["model"], pp["test_year"], pp["mu"] = name, year, mu
                all_predictions.append(pp)
            f5_mu, late_mu = phase_mu["f5"], phase_mu["late"]
            f5_true = keyrows.target.to_numpy(int)
            late_true = te[te.phase.eq("late")].sort_values(["game_pk", "is_home"]).target.to_numpy(int)
            full_mu, full_true = f5_mu + late_mu, f5_true + late_true
            if (full_mu < f5_mu).any():
                raise ValueError("DISTRIBUTION_INCONSISTENT")
            p5, pfull = phase_distributions(f5_mu, late_mu, plan)
            stats, game_loss = {}, {}
            for scope, mu, y, pmf in (("f5", f5_mu, f5_true, p5), ("full", full_mu, full_true, pfull)):
                # Means may be fitted with Poisson while score PMFs use the
                # fixed research dispersion. Report both proper count scores.
                p_obs = pmf[np.arange(len(y)), y]
                count_nll = -np.log(np.maximum(p_obs, 1e-300))
                poisson_nll = -poisson.logpmf(y, mu)
                pair_scores = y.reshape(-1, 2)
                probabilities = wdl_from_pmf(pmf[0::2], pmf[1::2])
                actual = np.where(pair_scores[:, 1] > pair_scores[:, 0], 2, np.where(pair_scores[:, 1] < pair_scores[:, 0], 0, 1))
                logloss = -np.log(np.maximum(probabilities[np.arange(n_games), actual], 1e-300))
                brier = np.mean((probabilities - np.eye(3)[actual]) ** 2, axis=1)
                stats[scope] = {
                    "count_nll": float(count_nll.mean()), "poisson_nll": float(poisson_nll.mean()),
                    "score_mae": float(np.abs(y-mu).mean()), "actual_mean": float(y.mean()),
                    "predicted_mean": float(mu.mean()), "actual_over_predicted": float(y.sum()/mu.sum()),
                    "wdl_logloss": float(logloss.mean()), "wdl_brier_sum_over_3": float(brier.mean()),
                    "actual_draw_rate": float((actual == 1).mean()), "predicted_draw_rate": float(probabilities[:, 1].mean()),
                }
                game_loss[scope + "_count_nll"] = count_nll.reshape(-1, 2).mean(axis=1)
                game_loss[scope + "_poisson_nll"] = poisson_nll.reshape(-1, 2).mean(axis=1)
                game_loss[scope + "_wdl_logloss"] = logloss
                game_loss[scope + "_wdl_brier"] = brier
                for i, pk in enumerate(keys):
                    all_games.append({
                        "game_pk": pk, "test_year": year, "model": name, "scope": scope,
                        "p_away": probabilities[i, 0], "p_draw": probabilities[i, 1], "p_home": probabilities[i, 2],
                        "actual_class": int(actual[i]), "wdl_logloss": logloss[i], "wdl_brier_sum_over_3": brier[i],
                        "count_nll": game_loss[scope + "_count_nll"][i],
                    })
            summary["models"][name] = stats
            per_game_losses[name] = game_loss
        for candidate, reference in (("lineup_delta", "context"), ("lineup_platoon", "context"), ("lineup_platoon", "lineup_delta")):
            summary["paired_differences"][candidate + "_minus_" + reference] = {
                metric: paired_ci(per_game_losses[candidate][metric] - per_game_losses[reference][metric], indices)
                for metric in per_game_losses[reference]
            }
        result["folds"].append(summary)
        print(json.dumps({"test_year": year, "test_games": n_games,
                          "models": summary["models"], "paired": summary["paired_differences"]}, ensure_ascii=False), flush=True)
    result["decision"] = "RESEARCH_ONLY_NO_PRODUCTION_PROMOTION_FROM_DEVELOPMENT_REPLAY"
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (output / "fitted_models.json").write_text(json.dumps(fitted, ensure_ascii=False, indent=2) + "\n")
    pd.concat(all_predictions, ignore_index=True).to_csv(output / "team_predictions.csv.gz", index=False)
    pd.DataFrame(all_games).to_csv(output / "game_probabilities.csv.gz", index=False)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--hands", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(PLAN_PATH.read_text())
    hand_document = json.loads(args.hands.read_text())
    raw, source_audit = load_sources(args.source, plan)
    games, game_audit = prepare_games(raw)
    print(json.dumps({"source_audit": source_audit, "game_audit": game_audit}, ensure_ascii=False), flush=True)
    frame = build_rows(games, plan, hand_document["players"])
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "features.csv.gz", index=False)
    audit = {
        "source_audit": source_audit, "game_audit": game_audit,
        "hand_map_sha256": file_sha(args.hands), "code_sha256": file_sha(Path(__file__)),
        "study_plan_sha256": file_sha(PLAN_PATH), "feature_rows": len(frame),
        "feature_sha256": file_sha(args.output / "features.csv.gz"),
        "strict_before_date": bool((frame.max_feature_date.isna() | (frame.max_feature_date < frame.game_date)).all()),
        "lineup_and_starter_source": "RETROSPECTIVE_PBP_PROXY_NOT_PREANNOUNCEMENT",
        "future_holdout_used": False,
    }
    (args.output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    evaluate(frame, plan, args.output)
    print("OFFENSE_PLATOON_V2_DONE=" + str(args.output), flush=True)


if __name__ == "__main__":
    main()
