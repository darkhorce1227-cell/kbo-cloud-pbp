#!/usr/bin/env python3
"""Reproducible mean-only KBO postseason Expected-IP research challenger.

This intentionally does NOT reconstruct the historical residual vector or tail
probabilities.  It only applies the frozen 2026 mean shift preserved in
research/postseason_expected_ip/frozen_2026_prior.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIOR = ROOT / "research" / "postseason_expected_ip" / "frozen_2026_prior.json"


def load_prior(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("status") != "CHALLENGER_ONLY_MEAN_REPRODUCIBLE":
        raise ValueError(f"unexpected prior status: {data.get('status')}")
    shift = data.get("research_basis", {}).get("mean_shift_ip")
    if not isinstance(shift, (int, float)):
        raise ValueError("mean_shift_ip missing from prior")
    return data


def build_forecast(rs_ip: float, prior: dict, role: str | None, round_name: str | None) -> dict:
    if not (0.0 <= rs_ip <= 9.0):
        raise ValueError("--rs-ip must be decimal innings in [0, 9]")

    shift = float(prior["research_basis"]["mean_shift_ip"])
    challenger = max(0.0, min(9.0, rs_ip + shift))
    warnings: list[str] = []

    if rs_ip < 5.0:
        warnings.append("LOW_SUPPORT_RS_IP_LT5: historical subgroup uncertainty was materially larger")
    if round_name and round_name.lower() in {"wd", "wc", "wildcard", "wild_card"}:
        warnings.append("LOW_SUPPORT_WILDCARD: historical WC subgroup improvement was uncertain")
    if role and role.upper() not in {"ESTABLISHED_STARTER", "NEWLY_ESTABLISHED_STARTER"}:
        warnings.append("SPECIAL_ROLE_CONTEXT_REQUIRED: do not auto-apply historical mean shift")

    return {
        "model": "PS_EXPECTED_IP_MEAN_SHIFT_2026",
        "status": "CHALLENGER_ONLY",
        "rs_starter_avg_ip": round(rs_ip, 7),
        "historical_ps_mean_shift_ip": shift,
        "challenger_mean_ip": round(challenger, 7),
        "role": role,
        "round": round_name,
        "warnings": warnings,
        "reproducibility": {
            "mean_forecast": "REPRODUCIBLE",
            "tail_probabilities": "UNAVAILABLE_MISSING_286_RESIDUAL_VECTOR",
            "walk_forward_refit": "UNAVAILABLE_MISSING_ORIGINAL_APPEARANCE_TABLE",
        },
        "usage_rule": "Research challenger only; never silently override Champion EIP or current role/news/pitch-cap evidence.",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rs-ip", type=float, required=True, help="Decimal innings, e.g. 5 2/3 = 5.6666667 (not 5.2)")
    p.add_argument("--role", default=None)
    p.add_argument("--round", dest="round_name", default=None)
    p.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    args = p.parse_args()

    prior = load_prior(args.prior)
    result = build_forecast(args.rs_ip, prior, args.role, args.round_name)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
