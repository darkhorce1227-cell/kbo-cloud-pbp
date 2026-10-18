#!/usr/bin/env python3
"""Research-only 2026 KBO PS Expected-IP tail-probability challenger.

The residual distribution is the 284 reproducible 2016-2025 postseason
starter deltas. The validation artifact is the independent 2017-2025
walk-forward study, never training on the target or future postseason.
All outputs are Challenger-only; nothing here selects a bet or overrides
current role, workload, or Champion Expected IP.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from forecast_ps_expected_ip_mean import build_forecast, load_canonical

ROOT = Path(__file__).resolve().parents[1]
RS_PS = ROOT / "research/postseason_expected_ip/ps_rs_starter_avg_ip.csv"
VALIDATION = ROOT / "research/postseason_expected_ip/ps_expected_ip_tail_validation.json"
EVENTS = {"IP_GE_5": (">=", 5.0), "IP_GE_6": (">=", 6.0), "IP_LT_4": ("<", 4.0)}
TOL = 1e-11


def load_verified_residuals(
    rows_path: Path = RS_PS, validation_path: Path = VALIDATION
) -> tuple[list[float], dict]:
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "PASS":
        raise ValueError("Tail calibration study is not validated")
    if not validation.get("live_tail_activation_recommended"):
        raise ValueError("Tail calibration activation gate did not pass")
    checks = validation.get("checks", {})
    if not checks or not all(checks.values()):
        raise ValueError("Tail study integrity checks not all PASS")
    if set(validation.get("activation_tests", {})) != set(EVENTS):
        raise ValueError("Unexpected calibrated events")
    if any(not all(s.values()) for s in validation["activation_tests"].values()):
        raise ValueError("Event calibration gate did not pass")

    residuals = []
    with rows_path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            yr = int(r["season"])
            if yr > 2025 or yr < 2016:
                raise ValueError(f"Unexpected historical season: {yr}")
            if r["rs_link_status"] != "MATCHED":
                if r["delta_ps_minus_rs_ip"] or r["rs_starter_avg_ip"]:
                    raise ValueError("UNAVAILABLE row contains inferred baseline/delta")
                continue
            actual = float(r["actual_ip"])
            baseline = float(r["rs_starter_avg_ip"])
            delta = float(r["delta_ps_minus_rs_ip"])
            if not all(map(math.isfinite, [actual, baseline, delta])):
                raise ValueError("Non-finite residual source")
            if not math.isclose(actual - baseline, delta, abs_tol=TOL):
                raise ValueError("Residual formula mismatch")
            residuals.append(delta)

    if len(residuals) != 284:
        raise ValueError(f"Strict residual count not 284: {len(residuals)}")
    if len(residuals) != validation["counts"]["strict_2026_residual_n"]:
        raise ValueError("Tail validation residual count mismatch")
    mean = sum(residuals) / len(residuals)
    if not math.isclose(
        mean, validation["strict_2026_residual_distribution"]["mean"],
        rel_tol=0, abs_tol=TOL
    ):
        raise ValueError("Tail validation residual mean mismatch")
    return residuals, validation


def probability(residuals: list[float], cutoff: float, op: str) -> float:
    if op == ">=":
        n = sum(x >= cutoff - 1e-12 for x in residuals)
    elif op == "<":
        n = sum(x < cutoff - 1e-12 for x in residuals)
    else:
        raise ValueError(op)
    return (n + 0.5) / (len(residuals) + 1.0)


def build_tail_forecast(
    rs_ip: float,
    rs_starts: int,
    status: dict,
    contract: dict,
    residuals: list[float],
    validation: dict,
    *,
    role: str | None = None,
    round_name: str | None = None,
    champion_ip: float | None = None,
    pitch_cap: int | None = None,
    injury_news: bool = False,
    short_rest: bool = False,
    manager_leash: str | None = None,
) -> dict:
    mean_result = build_forecast(
        rs_ip=rs_ip, rs_starts=rs_starts, status=status, contract=contract,
        role=role, round_name=round_name, champion_ip=champion_ip,
        pitch_cap=pitch_cap, injury_news=injury_news,
        short_rest=short_rest, manager_leash=manager_leash,
    )
    if not residuals:
        raise ValueError("No historical residuals")
    if mean_result["status"] == "UNAVAILABLE":
        event_probabilities = {event: None for event in EVENTS}
    else:
        event_probabilities = {
            event: probability(residuals, threshold-rs_ip, op)
            for event, (op, threshold) in EVENTS.items()
        }
        if event_probabilities["IP_GE_5"] < event_probabilities["IP_GE_6"]:
            raise ValueError("Tail probability monotonicity violated")

    return {
        "model": "PS_EXPECTED_IP_EMPIRICAL_TAIL_2026_V1",
        "status": mean_result["status"],
        "mean_challenger": mean_result,
        "historical_residual_n": len(residuals),
        "event_probabilities": event_probabilities,
        "historical_calibration": validation["empirical_model_by_event"],
        "operational_selected_probabilities": None,
        "usage_scope": "RESEARCH_CHALLENGER_ONLY",
        "calibration_scope": "2017-2025 prior-season walk-forward, 256 matched starts",
        "safety": "Current role, pitch cap, injury, rest, manager leash and Champion Expected IP outrank historical tail probabilities; no automated bet or size.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rs-ip", type=float, required=True)
    ap.add_argument("--rs-starts", type=int, required=True)
    ap.add_argument("--role", default="UNKNOWN")
    ap.add_argument("--round", dest="round_name", default=None)
    ap.add_argument("--champion-ip", type=float, default=None)
    ap.add_argument("--pitch-cap", type=int, default=None)
    ap.add_argument("--injury-news", action="store_true")
    ap.add_argument("--short-rest", action="store_true")
    ap.add_argument("--manager-leash", choices=["normal","tight","long","unknown"], default="unknown")
    ap.add_argument("--rows-file", type=Path, default=RS_PS)
    ap.add_argument("--validation-file", type=Path, default=VALIDATION)
    args = ap.parse_args()
    status, contract = load_canonical()
    residuals, validation = load_verified_residuals(args.rows_file, args.validation_file)
    result = build_tail_forecast(
        rs_ip=args.rs_ip, rs_starts=args.rs_starts,
        status=status, contract=contract,
        residuals=residuals, validation=validation,
        role=args.role, round_name=args.round_name,
        champion_ip=args.champion_ip, pitch_cap=args.pitch_cap,
        injury_news=args.injury_news, short_rest=args.short_rest,
        manager_leash=args.manager_leash,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
