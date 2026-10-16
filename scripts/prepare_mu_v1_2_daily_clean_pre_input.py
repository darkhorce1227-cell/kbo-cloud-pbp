from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = "state/mechanical_mu_input.json"
DECLARATION = "MARKET_UNSEEN_AT_INPUT_COMMIT"

MARKET_KEY_EXACT = {
    "odds",
    "bookmaker",
    "bookmakers",
    "betman",
    "proto",
    "sportsbook",
    "sportsbooks",
    "market_price",
    "market_prices",
    "market_odds",
    "betman_odds",
    "proto_odds",
    "sportsbook_odds",
}


def _find_market_keys(obj: object, path: str = "$") -> list[str]:
    hits: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            k = str(key).strip().lower()
            here = f"{path}.{key}"
            if k in MARKET_KEY_EXACT or k.endswith("_odds") or k.endswith("_bookmaker"):
                hits.append(here)
            hits.extend(_find_market_keys(value, here))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            hits.extend(_find_market_keys(value, f"{path}[{i}]"))
    return hits


def _validate_core_payload(payload: dict) -> None:
    if not isinstance(payload, dict):
        raise ValueError("daily input must be a JSON object")
    if not payload.get("target_date"):
        raise ValueError("daily input missing target_date")
    games = payload.get("games")
    if not isinstance(games, list) or not games:
        raise ValueError("daily input must contain a non-empty games list")
    required = ("away_team", "home_team", "away_starter", "home_starter")
    for i, game in enumerate(games):
        if not isinstance(game, dict):
            raise ValueError(f"games[{i}] must be an object")
        missing = [k for k in required if not game.get(k)]
        if missing:
            raise ValueError(f"games[{i}] missing required fields: {missing}")


def prepare_payload(raw: dict, *, market_unseen: bool) -> dict:
    if not market_unseen:
        raise ValueError(
            "Refusing CLEAN PRE preparation: explicit market-unseen declaration is required."
        )
    _validate_core_payload(raw)

    hits = _find_market_keys(raw)
    if hits:
        raise ValueError(
            "Refusing CLEAN PRE preparation: market-related keys found in raw payload: "
            + ", ".join(hits[:20])
        )

    payload = dict(raw)
    payload.pop("prospective_capture", None)
    payload["prospective_capture"] = {
        "clean_pre_lock": True,
        "market_prices_inspected": False,
        "declaration": DECLARATION,
        "prepared_by": "scripts/prepare_mu_v1_2_daily_clean_pre_input.py",
    }
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Prepare state/mechanical_mu_input.json for a genuine market-unseen "
            "daily CLEAN PRE run. The script inserts the prospective_capture "
            "declaration automatically and rejects obvious market-price fields."
        )
    )
    ap.add_argument("--source", required=True, help="Raw daily baseball-only JSON input")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument(
        "--market-unseen",
        action="store_true",
        help="Required explicit assertion that market prices have not been inspected.",
    )
    args = ap.parse_args()

    source = Path(args.source)
    if not source.is_absolute():
        source = ROOT / source
    raw = json.loads(source.read_text(encoding="utf-8"))

    try:
        payload = prepare_payload(raw, market_unseen=args.market_unseen)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        label = str(out.relative_to(ROOT))
    except ValueError:
        label = str(out)
    print(label)


if __name__ == "__main__":
    main()
