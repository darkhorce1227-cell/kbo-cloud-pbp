from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import build_mu_v1_2_features as base
import run_mu_v1_2_pipeline as runner

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "research" / "mu-v1.3" / "preprocessing.json"
MODEL_PATH = ROOT / "research" / "mu-v1.3" / "model_spec.json"
METRICS_PATH = ROOT / "research" / "mu-v1.3" / "metrics.json"

SIMPLE_F5 = ["off_f5_long_ratio", "off_f5_30d_ratio", "park_factor", "is_home"]
CORE_F5 = [
    "off_f5_long_ratio", "off_f5_30d_ratio",
    "sp_kbb_long", "sp_kbb_recent5", "sp_hr_long", "sp_hr_recent5",
    "sp_h_long", "sp_h_recent5", "sp_ip_long", "sp_ip_recent5",
    "opp_bp_ra9_long_rel", "opp_bp_ra9_30d_rel", "park_factor", "is_home",
]
SIMPLE_LATE = ["off_late_long_ratio", "off_late_30d_ratio", "park_factor", "is_home"]
CORE_LATE = [
    "off_late_long_ratio", "off_late_30d_ratio", "opp_bp_ra9_long_rel",
    "opp_bp_ra9_30d_rel", "sp_ip_long", "sp_ip_recent5", "park_factor", "is_home",
]


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def load_all(last_year: int):
    base.prepare_pa = runner.prepare_pa_fixed
    raw = base.load_seasons(last_year)
    pa = base.prepare_pa(raw)
    return base.game_tables(pa)


def _safe_rate(num: float, den: float, fallback: float) -> float:
    if den and np.isfinite(den) and den > 0:
        return float(num / den)
    return float(fallback)


def build_features(
    tg: pd.DataFrame,
    st: pd.DataFrame,
    bp: pd.DataFrame,
    *,
    d: pd.Timestamp,
    team: str,
    opponent: str,
    opposing_starter: Any,
    is_home: int,
    cfg: dict,
    park_factor_override: float | None = None,
) -> dict[str, float]:
    hist_start = pd.Timestamp(cfg["history_start"])
    sh = cfg["shrink"]
    pre = tg[(tg.game_date < d) & (tg.game_date >= hist_start)]
    if pre.empty:
        raise RuntimeError(f"no league history before {d.date()}")
    pre30 = pre[pre.game_date >= d - pd.Timedelta(days=int(cfg["recent_days"]))]

    lf5_long = float(pre.runs_f5.mean())
    llate_long = float(pre.runs_late.mean())
    lf5_30 = float(pre30.runs_f5.mean()) if len(pre30) else lf5_long
    llate_30 = float(pre30.runs_late.mean()) if len(pre30) else llate_long

    my = pre[pre.team.astype(str).eq(str(team))]
    my30 = pre30[pre30.team.astype(str).eq(str(team))]

    def shrunk_team(x: pd.DataFrame, metric: str, league: float, pseudo_games: float) -> float:
        val = (float(x[metric].sum()) + pseudo_games * league) / (float(len(x)) + pseudo_games)
        return _safe_rate(val, league, 1.0)

    out: dict[str, float] = {
        "off_f5_long_ratio": shrunk_team(my, "runs_f5", lf5_long, float(sh["off_f5_long_pseudo_games"])),
        "off_f5_30d_ratio": shrunk_team(my30, "runs_f5", lf5_30, float(sh["off_f5_30d_pseudo_games"])),
        "off_late_long_ratio": shrunk_team(my, "runs_late", llate_long, float(sh["off_late_long_pseudo_games"])),
        "off_late_30d_ratio": shrunk_team(my30, "runs_late", llate_30, float(sh["off_late_30d_pseudo_games"])),
        "league_f5_30d": lf5_30,
        "league_late_30d": llate_30,
        "is_home": float(is_home),
    }

    sp_pre = st[(st.game_date < d) & (st.game_date >= hist_start)]
    sp_mine = sp_pre[sp_pre.starter.astype(str).eq(str(opposing_starter))].sort_values("game_date")
    sp_recent = sp_mine.tail(int(cfg["starter_recent_starts"]))
    league_pa = float(sp_pre.pa.sum())
    league_kbb = _safe_rate(float(sp_pre.k.sum() - sp_pre.bb.sum()), league_pa, 0.0)
    league_h = _safe_rate(float(sp_pre.h.sum()), league_pa, 0.235)
    league_hr = _safe_rate(float(sp_pre.hr.sum()), league_pa, 0.023)
    league_ip = float(sp_pre.ip.mean()) if len(sp_pre) else 5.0

    def shrunk_rate(x: pd.DataFrame, numerator: str, league: float, pseudo_pa: float) -> float:
        den = float(x.pa.sum())
        if numerator == "kbb":
            num = float(x.k.sum() - x.bb.sum())
        else:
            num = float(x[numerator].sum())
        return (num + pseudo_pa * league) / (den + pseudo_pa) if den + pseudo_pa > 0 else league

    def shrunk_ip(x: pd.DataFrame, pseudo_starts: float) -> float:
        return (float(x.ip.sum()) + pseudo_starts * league_ip) / (float(len(x)) + pseudo_starts)

    out.update({
        "sp_kbb_long": shrunk_rate(sp_mine, "kbb", league_kbb, float(sh["sp_kbb_long_pseudo_pa"])),
        "sp_kbb_recent5": shrunk_rate(sp_recent, "kbb", league_kbb, float(sh["sp_kbb_recent5_pseudo_pa"])),
        "sp_h_long": shrunk_rate(sp_mine, "h", league_h, float(sh["sp_h_long_pseudo_pa"])),
        "sp_h_recent5": shrunk_rate(sp_recent, "h", league_h, float(sh["sp_h_recent5_pseudo_pa"])),
        "sp_hr_long": shrunk_rate(sp_mine, "hr", league_hr, float(sh["sp_hr_long_pseudo_pa"])),
        "sp_hr_recent5": shrunk_rate(sp_recent, "hr", league_hr, float(sh["sp_hr_recent5_pseudo_pa"])),
        "sp_ip_long": shrunk_ip(sp_mine, float(sh["sp_ip_long_pseudo_starts"])),
        "sp_ip_recent5": shrunk_ip(sp_recent, float(sh["sp_ip_recent5_pseudo_starts"])),
    })

    bp_pre = bp[(bp.game_date < d) & (bp.game_date >= hist_start)]
    bp30 = bp_pre[bp_pre.game_date >= d - pd.Timedelta(days=int(cfg["recent_days"]))]
    opp = bp_pre[bp_pre.pitching_team.astype(str).eq(str(opponent))]
    opp30 = bp30[bp30.pitching_team.astype(str).eq(str(opponent))]

    def ra9(x: pd.DataFrame, fallback: float) -> float:
        outs = float(x.bp_outs.sum())
        return float(x.bp_runs.sum() * 27.0 / outs) if outs > 0 else fallback

    league_bp = ra9(bp_pre, 4.5)
    league_bp30 = ra9(bp30, league_bp)

    def bp_rel(x: pd.DataFrame, league: float, pseudo_outs: float) -> float:
        outs = float(x.bp_outs.sum())
        runs = float(x.bp_runs.sum())
        team_ra9 = (runs * 27.0 + pseudo_outs * league) / (outs + pseudo_outs)
        return _safe_rate(team_ra9, league, 1.0)

    out["opp_bp_ra9_long_rel"] = bp_rel(opp, league_bp, float(sh["bp_long_pseudo_outs"]))
    out["opp_bp_ra9_30d_rel"] = bp_rel(opp30, league_bp30, float(sh["bp_30d_pseudo_outs"]))

    if park_factor_override is not None:
        out["park_factor"] = float(park_factor_override)
    else:
        home_team = team if is_home else opponent
        park_games = pre[(pre.team.astype(str).eq(str(home_team))) & (pre.is_home == 1)]
        league_full = float(pre.runs_full.mean())
        ids = park_games.game_pk.unique()
        pg = pre[pre.game_pk.isin(ids)]
        pseudo = float(sh["park_pseudo_games"])
        n_games = float(len(ids))
        if n_games <= 0 or league_full <= 0:
            out["park_factor"] = 1.0
        else:
            park_team_rate = (float(pg.runs_full.sum()) + 2.0 * pseudo * league_full) / (2.0 * (n_games + pseudo))
            out["park_factor"] = park_team_rate / league_full
    return out


def make_rows(tg: pd.DataFrame, st: pd.DataFrame, bp: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, cfg: dict) -> pd.DataFrame:
    st_map = st.set_index(["game_pk", "pitching_team"])["starter"].to_dict()
    rows = []
    for r in tg[(tg.game_date >= start) & (tg.game_date <= end)].sort_values(["game_date", "game_pk", "is_home"]).itertuples(index=False):
        opp_starter = st_map.get((r.game_pk, r.opponent))
        if opp_starter is None:
            continue
        f = build_features(
            tg, st, bp, d=r.game_date, team=r.team, opponent=r.opponent,
            opposing_starter=opp_starter, is_home=int(r.is_home), cfg=cfg,
        )
        f.update({
            "game_pk": str(r.game_pk), "game_date": str(r.game_date.date()),
            "team": str(r.team), "opponent": str(r.opponent), "is_home": float(r.is_home),
            "runs_f5": float(r.runs_f5), "runs_late": float(r.runs_late), "runs_full": float(r.runs_full),
        })
        rows.append(f)
    return pd.DataFrame(rows)


def fit_poisson_ridge(df: pd.DataFrame, features: list[str], target: str, offset_col: str, alpha: float) -> dict:
    Xraw = df[features].to_numpy(float)
    y = df[target].to_numpy(float)
    offset = np.clip(df[offset_col].to_numpy(float), 1e-6, None)
    mean = Xraw.mean(axis=0)
    scale = Xraw.std(axis=0, ddof=0)
    scale = np.where(scale < 1e-10, 1.0, scale)
    X = (Xraw - mean) / scale
    n, p = X.shape
    D = np.column_stack([np.ones(n), X])
    theta = np.zeros(p + 1)
    theta[0] = math.log(max(y.mean(), 1e-6) / max(offset.mean(), 1e-6))
    pen = np.zeros(p + 1)
    pen[1:] = float(alpha)
    for _ in range(100):
        eta = np.log(offset) + D @ theta
        eta = np.clip(eta, -20, 20)
        mu = np.exp(eta)
        grad = D.T @ (y - mu) - pen * theta
        info = D.T @ (D * mu[:, None]) + np.diag(pen)
        try:
            delta = np.linalg.solve(info + np.eye(p + 1) * 1e-9, grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(info + np.eye(p + 1) * 1e-7, grad, rcond=None)[0]
        theta += delta
        if float(np.max(np.abs(delta))) < 1e-9:
            break
    return {
        "features": features,
        "target": target,
        "offset": offset_col,
        "ridge_alpha": float(alpha),
        "scaler_mean": {k: float(v) for k, v in zip(features, mean)},
        "scaler_scale": {k: float(v) for k, v in zip(features, scale)},
        "params": {"const": float(theta[0]), **{k: float(v) for k, v in zip(features, theta[1:])}},
    }


def raw_component(comp: dict, f: dict[str, float]) -> float:
    v = float(comp["params"]["const"])
    for k in comp["features"]:
        z = (float(f[k]) - float(comp["scaler_mean"][k])) / float(comp["scaler_scale"][k])
        v += float(comp["params"][k]) * z
    return float(f[comp["offset"]]) * math.exp(max(-20.0, min(20.0, v)))


def score(model: dict, away: dict[str, float], home: dict[str, float]) -> dict[str, float]:
    sf_a, sf_h = raw_component(model["components"]["simple_f5"], away), raw_component(model["components"]["simple_f5"], home)
    cf_a, cf_h = raw_component(model["components"]["core_f5"], away), raw_component(model["components"]["core_f5"], home)
    f5_total = sf_a + sf_h
    g5 = float(model["gamma"]["f5"])
    sa = cf_a ** g5 / (cf_a ** g5 + cf_h ** g5)
    f5a, f5h = f5_total * sa, f5_total * (1.0 - sa)
    sl_a, sl_h = raw_component(model["components"]["simple_late"], away), raw_component(model["components"]["simple_late"], home)
    cl_a, cl_h = raw_component(model["components"]["core_late"], away), raw_component(model["components"]["core_late"], home)
    full_total = sf_a + sf_h + sl_a + sl_h
    cfa, cfh = cf_a + cl_a, cf_h + cl_h
    gf = float(model["gamma"]["full"])
    sfa = cfa ** gf / (cfa ** gf + cfh ** gf)
    return {
        "f5_away": f5a, "f5_home": f5h, "f5_total": f5_total,
        "full_away": full_total * sfa, "full_home": full_total * (1.0 - sfa), "full_total": full_total,
    }


def nb_logpmf(k: int, mu: float, alpha: float) -> float:
    mu = max(mu, 1e-9)
    r = 1.0 / alpha
    p = r / (r + mu)
    return math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1) + r * math.log(p) + k * math.log(1.0 - p)


def wdl_probs(away_mu: float, home_mu: float, alpha: float, max_runs: int = 30) -> np.ndarray:
    a = np.array([math.exp(nb_logpmf(k, away_mu, alpha)) for k in range(max_runs + 1)])
    h = np.array([math.exp(nb_logpmf(k, home_mu, alpha)) for k in range(max_runs + 1)])
    a /= a.sum(); h /= h.sum()
    away_win = float(sum(a[i] * h[j] for i in range(max_runs + 1) for j in range(max_runs + 1) if i > j))
    draw = float(np.dot(a, h))
    home_win = max(0.0, 1.0 - away_win - draw)
    return np.array([away_win, draw, home_win], dtype=float)


def evaluate(rows: pd.DataFrame, model: dict) -> dict:
    by_game = {g: x.sort_values("is_home") for g, x in rows.groupby("game_pk") if len(x) == 2}
    out = {"f5": {"abs": [], "nll": [], "brier": [], "ll": [], "hit": []}, "full": {"abs": [], "nll": [], "brier": [], "ll": [], "hit": []}}
    for _, x in by_game.items():
        away = x.iloc[0].to_dict(); home = x.iloc[1].to_dict()
        if int(away["is_home"]) != 0:
            away, home = home, away
        pred = score(model, away, home)
        for kind, alpha, actual_cols, pred_keys in (
            ("f5", float(model["alphas"]["f5"]), ("runs_f5", "runs_f5"), ("f5_away", "f5_home")),
            ("full", float(model["alphas"]["full"]), ("runs_full", "runs_full"), ("full_away", "full_home")),
        ):
            ya, yh = int(round(float(away[actual_cols[0]]))), int(round(float(home[actual_cols[1]])))
            ma, mh = float(pred[pred_keys[0]]), float(pred[pred_keys[1]])
            out[kind]["abs"].extend([abs(ya - ma), abs(yh - mh)])
            out[kind]["nll"].extend([-nb_logpmf(ya, ma, alpha), -nb_logpmf(yh, mh, alpha)])
            probs = wdl_probs(ma, mh, alpha)
            idx = 0 if ya > yh else 1 if ya == yh else 2
            yvec = np.zeros(3); yvec[idx] = 1.0
            out[kind]["brier"].append(float(np.mean((probs - yvec) ** 2)))
            out[kind]["ll"].append(float(-math.log(max(probs[idx], 1e-12))))
            if ya != yh:
                pick = 0 if probs[0] > probs[2] else 2
                out[kind]["hit"].append(1.0 if pick == idx else 0.0)
    return {
        k: {
            "team_score_mae": float(np.mean(v["abs"])),
            "nb_nll": float(np.mean(v["nll"])),
            "wdl_brier_mean3": float(np.mean(v["brier"])),
            "wdl_logloss": float(np.mean(v["ll"])),
            "decided_side_hit": float(np.mean(v["hit"])) if v["hit"] else None,
            "games": int(len(v["ll"])),
        }
        for k, v in out.items()
    }


def train() -> dict:
    cfg = config()
    tg, st, bp = load_all(2026)
    train_start = pd.Timestamp(cfg["train_start"])
    train_end = pd.Timestamp(cfg["train_end"])
    train_rows = make_rows(tg, st, bp, train_start, train_end, cfg)
    mcfg = cfg["model"]
    model = {
        "version": cfg["version"],
        "status": "RESEARCH_CANDIDATE_NOT_CHAMPION",
        "preprocessing": "research/mu-v1.3/preprocessing.json",
        "train": f"{cfg['train_start']} through {cfg['train_end']}",
        "alphas": {"f5": float(mcfg["f5_nb_alpha"]), "full": float(mcfg["full_nb_alpha"])},
        "gamma": {"f5": float(mcfg["f5_gamma"]), "full": float(mcfg["full_gamma"])},
        "components": {
            "simple_f5": fit_poisson_ridge(train_rows, SIMPLE_F5, "runs_f5", "league_f5_30d", float(mcfg["simple_f5_ridge_alpha"])),
            "core_f5": fit_poisson_ridge(train_rows, CORE_F5, "runs_f5", "league_f5_30d", float(mcfg["core_f5_ridge_alpha"])),
            "simple_late": fit_poisson_ridge(train_rows, SIMPLE_LATE, "runs_late", "league_late_30d", float(mcfg["simple_late_ridge_alpha"])),
            "core_late": fit_poisson_ridge(train_rows, CORE_LATE, "runs_late", "league_late_30d", float(mcfg["core_late_ridge_alpha"])),
        },
    }
    MODEL_PATH.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    oos_start = pd.Timestamp("2026-01-01")
    oos_end = pd.Timestamp(cfg["oos_end"])
    oos_rows = make_rows(tg, st, bp, oos_start, oos_end, cfg)
    metrics = {
        "version": cfg["version"],
        "train_team_games": int(len(train_rows)),
        "train_games": int(train_rows.game_pk.nunique()),
        "oos_team_games": int(len(oos_rows)),
        "oos_games": int(oos_rows.game_pk.nunique()),
        "oos_window": f"2026-01-01 through {cfg['oos_end']}",
        "metrics": evaluate(oos_rows, model),
        "comparison_note": "No 2026 outcome was used to choose preprocessing, ridge alpha, gamma, or NB alpha. v1.3 is a new reproducible candidate, not an exact reconstruction of v1.2.",
    }
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"model": model, "metrics": metrics}


def build_shadow(input_path: Path, out_path: Path) -> dict:
    cfg = config()
    if not MODEL_PATH.exists():
        raise RuntimeError("mu-v1.3 model_spec.json missing; run train first")
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    inp = json.loads(input_path.read_text(encoding="utf-8"))
    d = pd.Timestamp(inp["target_date"]).normalize()
    tg, st, bp = load_all(d.year)
    games = []
    for g in inp.get("games", []):
        af = build_features(tg, st, bp, d=d, team=g["away_team"], opponent=g["home_team"], opposing_starter=g["home_starter"], is_home=0, cfg=cfg, park_factor_override=g.get("park_factor"))
        hf = build_features(tg, st, bp, d=d, team=g["home_team"], opponent=g["away_team"], opposing_starter=g["away_starter"], is_home=1, cfg=cfg, park_factor_override=g.get("park_factor"))
        games.append({
            "game_pk": g.get("game_pk"), "away_team": g["away_team"], "home_team": g["home_team"],
            "away_starter": g["away_starter"], "home_starter": g["home_starter"],
            "away_features": af, "home_features": hf, "mechanical_mu": score(model, af, hf),
        })
    payload = {
        "status": "ACTIVE", "model": cfg["version"], "champion_status": "NOT_CHAMPION",
        "target_date": str(d.date()), "input_games": len(games), "games": games,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train")
    s = sub.add_parser("shadow")
    s.add_argument("--input", default="state/mechanical_mu_input.json")
    s.add_argument("--out", default="state/mechanical_mu_shadow_v1_3.json")
    args = ap.parse_args()
    payload = train() if args.cmd == "train" else build_shadow(ROOT / args.input, ROOT / args.out)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
