#!/usr/bin/env python3
"""Leakage-free calibration study for KBO postseason Expected-IP tail probabilities.

Research only. For each target postseason season Y, probabilities are built only
from MATCHED residuals in seasons < Y.  The strict residual is:

    residual = actual_postseason_ip - same-season_player_rs_starter_avg_ip

Tail model for a target row with RS starter average r and threshold t:

    P(actual_ip >= t) = P(residual >= t-r)

using a Jeffreys-smoothed empirical CDF: (k + 0.5) / (n + 1).
For the early-hook event actual_ip < 4, the analogous lower-tail count is used.

Benchmarks:
- expanding postseason climatology for the event (also prior seasons only)
- Gaussian residual model using prior residual mean/std

No live activation is performed by this script.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "research" / "postseason_expected_ip" / "ps_rs_starter_avg_ip.csv"
DEFAULT_OUTPUT_DIR = ROOT / "research" / "postseason_expected_ip"
EVENTS = {
    "IP_GE_5": (">=", 5.0),
    "IP_GE_6": (">=", 6.0),
    "IP_LT_4": ("<", 4.0),
}
EPS = 1e-12


def jeffreys(k: int, n: int) -> float:
    return (float(k) + 0.5) / (float(n) + 1.0)


def outcome(actual_ip: float, op: str, threshold: float) -> int:
    if op == ">=":
        return int(actual_ip >= threshold - 1e-12)
    if op == "<":
        return int(actual_ip < threshold - 1e-12)
    raise ValueError(op)


def empirical_tail(residuals: np.ndarray, cutoff: float, op: str) -> float:
    if op == ">=":
        k = int(np.sum(residuals >= cutoff - 1e-12))
    elif op == "<":
        k = int(np.sum(residuals < cutoff - 1e-12))
    else:
        raise ValueError(op)
    return jeffreys(k, len(residuals))


def gaussian_tail(mean: float, sd: float, cutoff: float, op: str) -> float:
    if sd <= 0 or not math.isfinite(sd):
        return float("nan")
    z = (cutoff - mean) / sd
    if op == ">=":
        return float(1.0 - norm.cdf(z))
    if op == "<":
        return float(norm.cdf(z))
    raise ValueError(op)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    q = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def reliability_bins(frame: pd.DataFrame, event_name: str, model_col: str) -> list[dict]:
    sub = frame[frame["event"] == event_name].copy()
    bins = np.linspace(0.0, 1.0, 6)
    # right edge 1.0 included in final bin
    sub["bin"] = pd.cut(sub[model_col], bins=bins, include_lowest=True, right=True, duplicates="drop")
    out: list[dict] = []
    for interval, g in sub.groupby("bin", observed=True):
        if len(g) == 0:
            continue
        out.append({
            "event": event_name,
            "model": model_col,
            "bin": str(interval),
            "n": int(len(g)),
            "mean_probability": float(g[model_col].mean()),
            "observed_rate": float(g["outcome"].mean()),
            "abs_gap": float(abs(g[model_col].mean() - g["outcome"].mean())),
        })
    return out


def ece_from_bins(bins: list[dict], total_n: int) -> float:
    if total_n <= 0:
        return float("nan")
    return float(sum(x["n"] / total_n * x["abs_gap"] for x in bins))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--eval-start", type=int, default=2017)
    ap.add_argument("--eval-end", type=int, default=2025)
    ap.add_argument("--curve-min-rs-ip", type=float, default=3.0)
    ap.add_argument("--curve-max-rs-ip", type=float, default=7.5)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    required = {"season", "game_pk", "team", "starter_id", "starter_name", "actual_ip", "rs_link_status", "rs_starter_avg_ip", "delta_ps_minus_rs_ip"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    matched = df[df["rs_link_status"] == "MATCHED"].copy()
    matched["actual_ip"] = pd.to_numeric(matched["actual_ip"], errors="raise")
    matched["rs_starter_avg_ip"] = pd.to_numeric(matched["rs_starter_avg_ip"], errors="raise")
    matched["delta_ps_minus_rs_ip"] = pd.to_numeric(matched["delta_ps_minus_rs_ip"], errors="raise")

    eval_all = df[(df["season"] >= args.eval_start) & (df["season"] <= args.eval_end)].copy()
    eval_matched = matched[(matched["season"] >= args.eval_start) & (matched["season"] <= args.eval_end)].copy()

    rows: list[dict] = []
    leakage_failures = 0
    for _, r in eval_matched.sort_values(["season", "game_pk", "team"]).iterrows():
        y = int(r["season"])
        hist = matched[matched["season"] < y].copy()
        if hist.empty:
            raise ValueError(f"no historical residuals for target season {y}")
        if int(hist["season"].max()) >= y:
            leakage_failures += 1
        residuals = hist["delta_ps_minus_rs_ip"].to_numpy(float)
        mu = float(np.mean(residuals))
        sd = float(np.std(residuals, ddof=1))
        for event_name, (op, threshold) in EVENTS.items():
            cutoff = threshold - float(r["rs_starter_avg_ip"])
            p_emp = empirical_tail(residuals, cutoff, op)
            hist_outcomes = np.array([outcome(v, op, threshold) for v in hist["actual_ip"].to_numpy(float)], dtype=int)
            p_clim = jeffreys(int(hist_outcomes.sum()), len(hist_outcomes))
            p_gauss = gaussian_tail(mu, sd, cutoff, op)
            rows.append({
                "season": y,
                "game_date": r.get("game_date", ""),
                "game_pk": r["game_pk"],
                "team": r["team"],
                "starter_id": str(r["starter_id"]),
                "starter_name": r["starter_name"],
                "actual_ip": float(r["actual_ip"]),
                "rs_starter_avg_ip": float(r["rs_starter_avg_ip"]),
                "event": event_name,
                "operator": op,
                "threshold_ip": threshold,
                "outcome": outcome(float(r["actual_ip"]), op, threshold),
                "prior_start_season": int(hist["season"].min()),
                "prior_end_season": int(hist["season"].max()),
                "prior_residual_n": int(len(hist)),
                "prior_residual_mean": mu,
                "prior_residual_sd": sd,
                "residual_cutoff": cutoff,
                "p_empirical_residual": p_emp,
                "p_gaussian_residual": p_gauss,
                "p_prior_climatology": p_clim,
            })

    pred = pd.DataFrame(rows)
    if pred.empty:
        raise ValueError("no evaluation predictions")

    metrics: list[dict] = []
    reliability: list[dict] = []
    models = ["p_empirical_residual", "p_gaussian_residual", "p_prior_climatology"]
    for event_name in EVENTS:
        sub = pred[pred["event"] == event_name].copy()
        y = sub["outcome"].to_numpy(float)
        event_rate = float(y.mean())
        for model in models:
            p = sub[model].to_numpy(float)
            bs = brier(y, p)
            ll = logloss(y, p)
            bins = reliability_bins(pred, event_name, model)
            reliability.extend(bins)
            ece = ece_from_bins(bins, len(sub))
            clim_bs = None
            if model != "p_prior_climatology":
                clim_p = sub["p_prior_climatology"].to_numpy(float)
                clim_bs = brier(y, clim_p)
            metrics.append({
                "event": event_name,
                "model": model,
                "n": int(len(sub)),
                "event_rate": event_rate,
                "mean_probability": float(np.mean(p)),
                "calibration_in_the_large": float(np.mean(p) - event_rate),
                "brier": bs,
                "log_loss": ll,
                "ece_5bin": ece,
                "brier_skill_vs_prior_climatology": (None if clim_bs is None or clim_bs <= 0 else float(1.0 - bs / clim_bs)),
            })

    metrics_df = pd.DataFrame(metrics)
    rel_df = pd.DataFrame(reliability)

    # Season-level directional audit: empirical residual Brier vs expanding climatology.
    season_rows: list[dict] = []
    for event_name in EVENTS:
        for season, sub in pred[pred["event"] == event_name].groupby("season"):
            y = sub["outcome"].to_numpy(float)
            b_emp = brier(y, sub["p_empirical_residual"].to_numpy(float))
            b_clim = brier(y, sub["p_prior_climatology"].to_numpy(float))
            season_rows.append({
                "event": event_name,
                "season": int(season),
                "n": int(len(sub)),
                "event_rate": float(y.mean()),
                "empirical_brier": b_emp,
                "climatology_brier": b_clim,
                "empirical_brier_lower": bool(b_emp < b_clim),
                "brier_delta_empirical_minus_climatology": float(b_emp - b_clim),
            })
    season_df = pd.DataFrame(season_rows)

    # 2026 research curve from all 2016-2025 strict matched residuals (284 rows).
    hist_2026 = matched[matched["season"] <= 2025].copy()
    residuals_2026 = hist_2026["delta_ps_minus_rs_ip"].to_numpy(float)
    mu_2026 = float(np.mean(residuals_2026))
    sd_2026 = float(np.std(residuals_2026, ddof=1))
    grid_outs_min = int(round(args.curve_min_rs_ip * 3))
    grid_outs_max = int(round(args.curve_max_rs_ip * 3))
    curve_rows: list[dict] = []
    for outs in range(grid_outs_min, grid_outs_max + 1):
        rs_ip = outs / 3.0
        row = {"rs_starter_avg_ip": rs_ip, "residual_n": int(len(residuals_2026)), "residual_mean": mu_2026, "residual_sd": sd_2026}
        for event_name, (op, threshold) in EVENTS.items():
            cutoff = threshold - rs_ip
            row[f"p_{event_name.lower()}_empirical"] = empirical_tail(residuals_2026, cutoff, op)
            row[f"p_{event_name.lower()}_gaussian"] = gaussian_tail(mu_2026, sd_2026, cutoff, op)
        curve_rows.append(row)
    curve_df = pd.DataFrame(curve_rows)

    empirical_metrics = metrics_df[metrics_df["model"] == "p_empirical_residual"].set_index("event")
    by_event = {}
    for event_name in EVENTS:
        m = empirical_metrics.loc[event_name]
        wins = season_df[season_df["event"] == event_name]["empirical_brier_lower"]
        by_event[event_name] = {
            "n": int(m["n"]),
            "event_rate": float(m["event_rate"]),
            "mean_probability": float(m["mean_probability"]),
            "calibration_in_the_large": float(m["calibration_in_the_large"]),
            "brier": float(m["brier"]),
            "log_loss": float(m["log_loss"]),
            "ece_5bin": float(m["ece_5bin"]),
            "brier_skill_vs_prior_climatology": float(m["brier_skill_vs_prior_climatology"]),
            "season_brier_wins_vs_climatology": int(wins.sum()),
            "season_count": int(len(wins)),
        }

    checks = {
        "eval_total_rows_258": int(len(eval_all)) == 258,
        "eval_matched_rows_256": int(len(eval_matched)) == 256,
        "eval_unavailable_rows_2": int(len(eval_all) - len(eval_matched)) == 2,
        "prediction_rows_768": int(len(pred)) == 256 * len(EVENTS),
        "events_exact": set(pred["event"].unique()) == set(EVENTS),
        "all_probabilities_strictly_inside_0_1": bool(((pred[models] > 0) & (pred[models] < 1)).all().all()),
        "no_target_or_future_leakage": leakage_failures == 0 and bool((pred["prior_end_season"] < pred["season"]).all()),
        "first_eval_fold_prior_n_28": bool((pred[pred["season"] == args.eval_start]["prior_residual_n"] == 28).all()),
        "2026_residual_n_284": int(len(residuals_2026)) == 284,
        "2026_residual_mean_matches_canonical": abs(mu_2026 - (-0.7598198773356355)) < 1e-12,
    }

    # Research activation gate is deliberately conservative; it is a conclusion,
    # not an integrity check. We require positive Brier skill and tolerable mean
    # calibration gap for all three operational events before recommending live use.
    activation_tests = {
        event_name: {
            "positive_brier_skill_vs_climatology": by_event[event_name]["brier_skill_vs_prior_climatology"] > 0,
            "abs_calibration_in_the_large_le_0_05": abs(by_event[event_name]["calibration_in_the_large"]) <= 0.05,
            "ece_5bin_le_0_10": by_event[event_name]["ece_5bin"] <= 0.10,
        }
        for event_name in EVENTS
    }
    live_tail_activation_recommended = all(all(v.values()) for v in activation_tests.values())

    validation = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "study": "PS_EXPECTED_IP_TAIL_PROBABILITY_WALK_FORWARD_V1",
        "scope": "RESEARCH_ONLY_NO_LIVE_ACTIVATION",
        "definition": {
            "residual": "actual_ps_ip - same-season_player_rs_starter_avg_ip",
            "empirical_probability": "Jeffreys-smoothed expanding prior-season residual CDF",
            "events": {k: {"operator": v[0], "threshold_ip": v[1]} for k, v in EVENTS.items()},
            "benchmarks": ["expanding_prior_postseason_climatology", "Gaussian_prior_residual_distribution"],
        },
        "counts": {
            "eval_total": int(len(eval_all)),
            "eval_matched": int(len(eval_matched)),
            "eval_unavailable": int(len(eval_all) - len(eval_matched)),
            "prediction_event_rows": int(len(pred)),
            "strict_2026_residual_n": int(len(residuals_2026)),
        },
        "strict_2026_residual_distribution": {
            "mean": mu_2026,
            "sd": sd_2026,
            "q05": float(np.quantile(residuals_2026, 0.05)),
            "q10": float(np.quantile(residuals_2026, 0.10)),
            "q25": float(np.quantile(residuals_2026, 0.25)),
            "median": float(np.quantile(residuals_2026, 0.50)),
            "q75": float(np.quantile(residuals_2026, 0.75)),
            "q90": float(np.quantile(residuals_2026, 0.90)),
            "q95": float(np.quantile(residuals_2026, 0.95)),
        },
        "empirical_model_by_event": by_event,
        "activation_tests": activation_tests,
        "live_tail_activation_recommended": live_tail_activation_recommended,
        "checks": checks,
        "usage_rule": "Do not enable live tail probabilities unless calibration gate passes; current role/pitch cap/injury/rest/manager leash remain higher priority regardless.",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.output_dir / "ps_expected_ip_tail_walk_forward_predictions_2017_2025.csv", index=False)
    metrics_df.to_csv(args.output_dir / "ps_expected_ip_tail_walk_forward_metrics.csv", index=False)
    rel_df.to_csv(args.output_dir / "ps_expected_ip_tail_reliability_bins.csv", index=False)
    season_df.to_csv(args.output_dir / "ps_expected_ip_tail_metrics_by_season.csv", index=False)
    curve_df.to_csv(args.output_dir / "ps_expected_ip_tail_probability_curve_2026.csv", index=False)
    with (args.output_dir / "ps_expected_ip_tail_validation.json").open("w", encoding="utf-8") as f:
        json.dump(validation, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    print(json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True))
    if validation["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
