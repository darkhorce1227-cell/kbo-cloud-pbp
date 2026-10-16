from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import mu_v1_2_recovered_challenger as challenger

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = "state/mechanical_mu_input.json"
DEFAULT_DIR = "state/prospective_mu_v1_2"
EMBEDDED_DECLARATION = "MARKET_UNSEEN_AT_INPUT_COMMIT"


def _canonical_bytes(obj: object) -> bytes:
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(obj: object) -> str:
    return hashlib.sha256(_canonical_bytes(obj)).hexdigest()


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def embedded_clean_pre_declaration(payload: dict) -> tuple[bool, dict]:
    """Validate the declaration carried by the committed daily PRE input.

    This lets the daily workflow capture prospectively without pretending that every
    mechanical_mu_input.json commit was market-unseen. A missing/malformed declaration
    is ineligible for prospective evidence.
    """
    meta = payload.get("prospective_capture")
    if not isinstance(meta, dict):
        return False, {}
    ok = (
        meta.get("clean_pre_lock") is True
        and meta.get("market_prices_inspected") is False
        and meta.get("declaration") == EMBEDDED_DECLARATION
    )
    return bool(ok), dict(meta)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Capture an immutable prospective pregame snapshot for the active "
            "mu-v1.2 recovered challenger. This is a research audit artifact, "
            "not Official Mechanical Shadow output."
        )
    )
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--out-dir", default=DEFAULT_DIR)
    ap.add_argument(
        "--market-unseen",
        action="store_true",
        help="Manual explicit declaration that betting-market prices have not been inspected.",
    )
    ap.add_argument(
        "--require-input-declaration",
        action="store_true",
        help=(
            "Require prospective_capture metadata embedded in the committed input: "
            "clean_pre_lock=true, market_prices_inspected=false, "
            "declaration=MARKET_UNSEEN_AT_INPUT_COMMIT."
        ),
    )
    ap.add_argument(
        "--note",
        default="",
        help="Optional contemporaneous note; do not add hindsight information later.",
    )
    args = ap.parse_args()

    if not args.market_unseen and not args.require_input_declaration:
        raise SystemExit(
            "Refusing prospective CLEAN snapshot: use --market-unseen for a manual "
            "market-unseen capture or --require-input-declaration for an eligible "
            "committed CLEAN PRE input."
        )

    input_path = ROOT / args.input
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    embedded_ok, embedded_meta = embedded_clean_pre_declaration(payload)

    if args.require_input_declaration:
        if not embedded_ok:
            raise SystemExit(
                "Refusing prospective CLEAN snapshot: committed input is missing a valid "
                "prospective_capture CLEAN PRE market-unseen declaration."
            )
        declaration = EMBEDDED_DECLARATION
        declaration_source = "INPUT_COMMIT_METADATA"
    else:
        declaration = "MARKET_UNSEEN_AT_CAPTURE"
        declaration_source = "CLI_EXPLICIT_DECLARATION"

    captured_at = datetime.now(timezone.utc)
    scored = challenger.score_input(payload)

    snapshot = {
        "snapshot_schema": "mu-v1.2-prospective-snapshot-v0.2",
        "capture_type": "PROSPECTIVE_CLEAN_PRE_RESEARCH",
        "captured_at_utc": captured_at.isoformat().replace("+00:00", "Z"),
        "market_state": {
            "prices_inspected": False,
            "declaration": declaration,
            "declaration_source": declaration_source,
            "input_commit_metadata": embedded_meta if declaration_source == "INPUT_COMMIT_METADATA" else None,
        },
        "official_shadow_allowed": False,
        "git_head": _git_head(),
        "input_path": str(args.input),
        "input_sha256": _sha256(payload),
        "input_payload": payload,
        "challenger_version": scored.get("model"),
        "challenger_output_sha256": _sha256(scored),
        "challenger_output": scored,
        "note": args.note,
        "immutability_rule": (
            "Do not overwrite this file. Any correction or later market-aware run "
            "must be written as a separate artifact."
        ),
    }

    target_date = str(scored.get("target_date", "unknown"))
    stamp = captured_at.strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{target_date}_{stamp}.json"

    if out_path.exists():
        raise SystemExit(f"Refusing to overwrite existing prospective snapshot: {out_path}")

    out_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(out_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
