from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROSPECTIVE = ROOT / "research" / "mu-v1.2" / "prospective_gate_status.json"
RECOVERY = ROOT / "research" / "mu-v1.2" / "recovery_status.json"
ACTIVE = "mu-v1.2-recovered-challenger-v0.9-bp30-cal36-p25.5"
PREPARER_RUN = 38875461292
DEDUPE_RUN = 38875572955


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    p = load(PROSPECTIVE)
    r = load(RECOVERY)

    if p.get("official_shadow_allowed") is not False or r.get("official_shadow_allowed") is not False:
        raise SystemExit("Refusing registrar update: Official Shadow must remain false.")
    if p.get("active_challenger") != ACTIVE:
        raise SystemExit(f"Unexpected prospective active challenger: {p.get('active_challenger')}")
    if r.get("active_challenger", {}).get("version") != ACTIVE:
        raise SystemExit(f"Unexpected recovery active challenger: {r.get('active_challenger', {}).get('version')}")

    p["version"] = "mu-v1.2-prospective-gate-status-v0.4"
    p["development_state"] = "DEVELOPMENT_COMPLETE_LIVE_SAMPLE_COLLECTION_ONLY"
    p["daily_clean_pre_input_preparer"] = {
        "status": "SMOKE_PASS",
        "script": "scripts/prepare_mu_v1_2_daily_clean_pre_input.py",
        "workflow_run_id": PREPARER_RUN,
        "workflow_conclusion": "success",
        "behavior": [
            "requires explicit --market-unseen assertion",
            "validates target_date and non-empty game/starter input contract",
            "rejects obvious market-bearing fields such as odds/Proto/Betman/sportsbook keys",
            "automatically inserts the exact prospective_capture CLEAN PRE declaration used by daily auto-capture"
        ],
        "integrity_note": "The preparer automates metadata insertion; it cannot infer whether a human actually viewed market prices. The explicit market-unseen assertion remains a process-integrity requirement."
    }
    p["bp30_gate_evaluator"]["status"] = "SMOKE_PASS_WITH_DEDUPE"
    p["bp30_gate_evaluator"]["evaluation_version"] = "mu-v1.2-bp30-prospective-promotion-gate-evaluation-v0.2-deduped"
    p["bp30_gate_evaluator"]["dedupe_smoke_run_id"] = DEDUPE_RUN
    p["bp30_gate_evaluator"]["dedupe_smoke_conclusion"] = "success"
    verified = list(p["bp30_gate_evaluator"].get("verified", []))
    for item in [
        "exact duplicate game observations do not inflate prospective game count",
        "dedupe key is input_sha256 + target_date + game_pk-or-matchup",
        "duplicate files remain auditable while duplicate game observations are excluded from metrics"
    ]:
        if item not in verified:
            verified.append(item)
    p["bp30_gate_evaluator"]["verified"] = verified
    p["dedupe_rule"] = {
        "status": "ACTIVE_SMOKE_PASS",
        "key": "input_sha256 + target_date + game_pk_or_away>home",
        "scope": "Exact same-input same-date same-matchup observations count once.",
        "important_limit": "A genuinely revised input with a different input_sha256 is not treated as an exact duplicate by this rule.",
        "workflow_run_id": DEDUPE_RUN
    }
    p["next_gate"] = (
        "LIVE_SAMPLE_COLLECTION_ONLY: prepare the next genuine market-unseen daily baseball input with "
        "scripts/prepare_mu_v1_2_daily_clean_pre_input.py, commit the generated state/mechanical_mu_input.json, "
        "and let the existing auto-capture workflow accumulate immutable deduped evidence. No further model or "
        "threshold development is required before the preregistered sample gate is reached. Official Mechanical "
        "Shadow remains OFF."
    )

    r["version"] = "mu-v1.2-preprocessing-recovery-status-v0.13"
    r["status"] = "RECOVERY_DEVELOPMENT_COMPLETE_LIVE_COLLECTION_PENDING"
    r["development_state"] = {
        "status": "DEVELOPMENT_COMPLETE_LIVE_SAMPLE_COLLECTION_ONLY",
        "completed_components": [
            "active v0.9 challenger regression gate",
            "immutable prospective CLEAN PRE capture",
            "embedded declaration validation",
            "daily PRE auto-capture integration",
            "safe daily CLEAN PRE input preparer",
            "BP30 calendar36-vs-calendar30 discrepancy comparator",
            "preregistered BP30 prospective promotion evaluator",
            "exact-duplicate prospective game dedupe guard"
        ],
        "remaining_work": "Collect genuine future market-unseen observations and evaluate the frozen preregistered gate; do not retune from those observations.",
        "official_shadow_allowed": False
    }
    ac = r["active_challenger"]
    ac["daily_clean_pre_input_preparer"] = {
        "script": "scripts/prepare_mu_v1_2_daily_clean_pre_input.py",
        "smoke_run_id": PREPARER_RUN,
        "smoke_conclusion": "success",
        "market_data_guard": "Rejects obvious market-bearing fields and requires explicit --market-unseen assertion before inserting prospective_capture metadata."
    }
    gate = ac["bp30_prospective_gate"]
    gate["evaluator_version"] = "mu-v1.2-bp30-prospective-promotion-gate-evaluation-v0.2-deduped"
    gate["dedupe_smoke_run_id"] = DEDUPE_RUN
    gate["dedupe_smoke_conclusion"] = "success"
    gate["dedupe_rule"] = "input_sha256 + target_date + game_pk-or-matchup; exact duplicate observations count once"
    gate["dedupe_limit"] = "Different input_sha256 revisions are not exact duplicates under this rule."
    r["promotion_gate"] = (
        "Development/infrastructure work is complete for the current v0.9 recovery path. Regression, capture/declaration, "
        "safe daily input preparation, auto-capture integration, BP30 comparator/evaluator, and exact-duplicate dedupe smoke "
        "tests have passed. From here the required work is genuine market-unseen prospective sample collection under the frozen "
        "v0.1 thresholds (minimum 10 games / 4 dates / 6 teams) and gate evaluation. Do not relax or retune thresholds after "
        "observing eligible snapshots. Official Mechanical Mu Shadow remains unavailable until the gate is satisfied and promotion "
        "is explicitly approved. Exact historical row parity remains untestable because the original feature-row artifact does not survive."
    )

    save(PROSPECTIVE, p)
    save(RECOVERY, r)
    print("MU_V1_2_LIVE_COLLECTION_READY_REGISTERED")


if __name__ == "__main__":
    main()
