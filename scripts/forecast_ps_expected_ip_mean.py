#!/usr/bin/env python3
"""Canonical strict KBO postseason Expected-IP research challenger.

This helper applies only the reproducible strict historical mean shift. It does
not invent a value when the pitcher had zero same-season RS starts, and it never
silently replaces Champion Expected IP or higher-priority current-context usage
evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATUS = ROOT / "research" / "postseason_expected_ip" / "canonical_recovery_status.json"
DEFAULT_CONTRACT = ROOT / "research" / "postseason_expected_ip" / "live_operating_contract_2026.json"

SAFE_ROLES = {"ESTABLISHED_STARTER", "NEWLY_ESTABLISHED_STARTER"}
REVIEW_ROLES = {"OPENER", "BULLPEN_GAME", "EMERGENCY_START", "UNKNOWN"}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_canonical(status_path: Path = DEFAULT_STATUS, contract_path: Path = DEFAULT_CONTRACT) -> tuple[dict, dict]:
    status = load_json(status_path)
    contract = load_json(contract_path)
    if status.get("status") != "CLOSED_CANONICAL_STRICT_REPRODUCTION":
        raise ValueError(f"unexpected canonical recovery status: {status.get('status')}")
    if contract.get("status") != "ACTIVE_RESEARCH_CHALLENGER_CANONICAL_STRICT":
        raise ValueError(f"unexpected operating contract status: {contract.get('status')}")
    shift_status = float(status["canonical_2026_shift_ip"])
    shift_contract = float(contract["canonical_shift_ip"])
    if abs(shift_status - shift_contract) > 1e-12:
        raise ValueError("canonical shift mismatch between recovery status and live contract")
    return status, contract


def build_forecast(
    rs_ip: float,
    rs_starts: int,
    status: dict,
    contract: dict,
    role: str | None = None,
    round_name: str | None = None,
    champion_ip: float | None = None,
    pitch_cap: int | None = None,
    injury_news: bool = False,
    short_rest: bool = False,
    manager_leash: str | None = None,
) -> dict:
    if not (0.0 <= rs_ip <= 9.0):
        raise ValueError("--rs-ip must be decimal innings in [0, 9]")
    if rs_starts < 0:
        raise ValueError("--rs-starts must be >= 0")
    if champion_ip is not None and not (0.0 <= champion_ip <= 9.0):
        raise ValueError("--champion-ip must be in [0, 9]")
    if pitch_cap is not None and pitch_cap <= 0:
        raise ValueError("--pitch-cap must be > 0 when provided")

    shift = float(status["canonical_2026_shift_ip"])
    role_norm = (role or "UNKNOWN").upper()
    leash_norm = (manager_leash or "UNKNOWN").upper()

    warnings: list[str] = []
    context_flags: list[str] = []

    if rs_starts == 0:
        return {
            "model": contract["model_id"],
            "status": "UNAVAILABLE",
            "reason": "NO_SAME_SEASON_RS_START_HISTORY",
            "rs_starter_avg_ip": round(rs_ip, 7),
            "rs_starts": 0,
            "historical_ps_mean_shift_ip": shift,
            "challenger_mean_ip": None,
            "champion_ip": champion_ip,
            "operational_selected_ip": None,
            "role": role_norm,
            "round": round_name,
            "warnings": ["STRICT_RULE: zero same-season RS starts; no fallback is allowed"],
            "priority_order": contract["priority_order"],
            "reproducibility": contract["reproducibility"],
        }

    challenger = max(0.0, min(9.0, rs_ip + shift))

    if role_norm not in SAFE_ROLES:
        context_flags.append(f"ROLE_{role_norm}")
        warnings.append("SPECIAL_ROLE_CONTEXT_REQUIRED: do not auto-apply historical mean shift")
    if pitch_cap is not None:
        context_flags.append("PITCH_CAP_PRESENT")
        warnings.append("PITCH_CAP_PRESENT: pitch count cannot be converted mechanically to IP; current cap outranks history")
    if injury_news:
        context_flags.append("INJURY_OR_CURRENT_NEWS")
        warnings.append("INJURY_OR_CURRENT_NEWS: current evidence outranks historical shift")
    if short_rest:
        context_flags.append("SHORT_REST_OR_WORKLOAD")
        warnings.append("SHORT_REST_OR_WORKLOAD: current workload outranks historical shift")
    if leash_norm == "TIGHT":
        context_flags.append("TIGHT_MANAGER_LEASH")
        warnings.append("TIGHT_MANAGER_LEASH: manager usage evidence outranks historical shift")

    if rs_ip < 5.0:
        warnings.append("LOW_RS_STARTER_AVG_IP: treat historical shift as secondary usage evidence")
    if round_name and round_name.lower() in {"wd", "wc", "wildcard", "wild_card"}:
        warnings.append("WILDCARD_CONTEXT: do not assume identical usage across postseason rounds")

    operational_status = "CONTEXT_REVIEW_REQUIRED" if context_flags else "CHALLENGER_ONLY"

    result = {
        "model": contract["model_id"],
        "status": operational_status,
        "rs_starter_avg_ip": round(rs_ip, 7),
        "rs_starts": int(rs_starts),
        "historical_ps_mean_shift_ip": shift,
        "challenger_mean_ip": round(challenger, 7),
        "champion_ip": champion_ip,
        "challenger_minus_champion_ip": round(challenger - champion_ip, 7) if champion_ip is not None else None,
        "operational_selected_ip": None,
        "role": role_norm,
        "round": round_name,
        "pitch_cap": pitch_cap,
        "injury_news": bool(injury_news),
        "short_rest": bool(short_rest),
        "manager_leash": leash_norm,
        "context_flags": context_flags,
        "warnings": warnings,
        "priority_order": contract["priority_order"],
        "historical_support": contract["historical_support"],
        "reproducibility": contract["reproducibility"],
        "usage_rule": "Research challenger only. Display separately from Champion; current role/cap/news/rest/leash evidence has priority.",
    }
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rs-ip", type=float, required=True, help="Same-season RS starter average IP in decimal innings")
    p.add_argument("--rs-starts", type=int, required=True, help="Same-season RS starts; 0 makes the challenger UNAVAILABLE")
    p.add_argument("--role", default="UNKNOWN")
    p.add_argument("--round", dest="round_name", default=None)
    p.add_argument("--champion-ip", type=float, default=None, help="Optional Champion Expected IP for side-by-side audit only")
    p.add_argument("--pitch-cap", type=int, default=None, help="Optional known current-game pitch cap; triggers context review")
    p.add_argument("--injury-news", action="store_true")
    p.add_argument("--short-rest", action="store_true")
    p.add_argument("--manager-leash", choices=["normal", "tight", "long", "unknown"], default="unknown")
    p.add_argument("--status-file", type=Path, default=DEFAULT_STATUS)
    p.add_argument("--contract-file", type=Path, default=DEFAULT_CONTRACT)
    args = p.parse_args()

    status, contract = load_canonical(args.status_file, args.contract_file)
    result = build_forecast(
        rs_ip=args.rs_ip,
        rs_starts=args.rs_starts,
        status=status,
        contract=contract,
        role=args.role,
        round_name=args.round_name,
        champion_ip=args.champion_ip,
        pitch_cap=args.pitch_cap,
        injury_news=args.injury_news,
        short_rest=args.short_rest,
        manager_leash=args.manager_leash,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
