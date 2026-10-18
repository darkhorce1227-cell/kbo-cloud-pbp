#!/usr/bin/env python3
"""Audit market-implied fair prices separately from genuine CLEAN PRE forecasts.

Input: one market-outcome price per row. This deliberately will NOT invent
historical odds, model probabilities or a timestamped prediction history.

    game_pk,market_id,market_type,line,price_asof_kst,game_start_kst,
    settlement_status,settled_outcome,outcome,decimal_odds,
    model_probability,model_lock_kst,model_lock_ref

market_type in FULL_ML2,F5_1X2,FULL_OU,F5_OU,FULL_HC2,FULL_HC3.
For no-draw ML2 and HC2, a tied final/adjusted score is PUSH (refund),
not a home or away loss. VOID also excluded from probability calibration.
OU on an integer threshold may PUSH. The "line" is the home-team handicap
or total runs threshold, NOT a string like 'over 9.5'.

odds_asof must precede first pitch; a model row requires an earlier LOCK
with a verifiable external ref (ref verification is separate from this script).
No-vig probabilities are a conditional forecast of the offered non-push
outcomes, not playable payout odds. No one can bet at synthetic fair odds.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime
import json
import math
from pathlib import Path

MARKETS = {
    "FULL_ML2": ("HOME", "AWAY"),
    "F5_1X2": ("HOME", "DRAW", "AWAY"),
    "FULL_OU": ("OVER", "UNDER"),
    "F5_OU": ("OVER", "UNDER"),
    "FULL_HC2": ("HOME", "AWAY"),
    "FULL_HC3": ("HOME", "DRAW", "AWAY"),
}
REQUIRED = {
    "game_pk", "market_id", "market_type", "line", "price_asof_kst",
    "game_start_kst", "settlement_status", "settled_outcome",
    "outcome", "decimal_odds", "model_probability",
    "model_lock_kst", "model_lock_ref",
}
SETTLEMENTS = {"SETTLED", "PUSH", "VOID"}


def date_(s):
    t = datetime.fromisoformat(s)
    if t.tzinfo is None or t.utcoffset() is None:
        raise ValueError(f"Timezone required for {s}")
    return t


def load_events(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("No historical odds rows. Backtest unavailable; never fabricate them.")
    groups = defaultdict(list)
    for i, r in enumerate(rows, 2):
        typ = r["market_type"].strip().upper()
        if typ not in MARKETS:
            raise ValueError(f"Line {i} unsupported market {typ!r}")
        if not r["game_pk"].strip() or not r["market_id"].strip():
            raise ValueError(f"Line {i} missing game or market identifier")
        groups[(r["game_pk"], r["market_id"], r["price_asof_kst"])].append(r)

    events = []
    for key, group in groups.items():
        a = group[0]
        typ = a["market_type"].strip().upper()
        offered = MARKETS[typ]
        info_fields = ["market_type","line","game_start_kst","settlement_status","settled_outcome"]
        for f in info_fields:
            if any(r[f] != a[f] for r in group):
                raise ValueError(f"{key}: inconsistent {f} across price outcomes")
        price_time = date_(a["price_asof_kst"])
        start_time = date_(a["game_start_kst"])
        if price_time >= start_time:
            raise ValueError(f"{key}: POST_START_PRICE_LEAKAGE")
        if typ.endswith("_OU") or "_HC" in typ:
            if not a["line"].strip():
                raise ValueError(f"{key}: missing total/handicap line")
            line = float(a["line"])
            if not math.isfinite(line):
                raise ValueError(f"{key}: nonfinite line")
        elif a["line"].strip():
            raise ValueError(f"{key}: unexpected line for non-line market")
        options = [r["outcome"].strip().upper() for r in group]
        if len(options) != len(offered) or set(options) != set(offered):
            raise ValueError(f"{key}: expected outcomes {offered}, received {options}")
        outcome_odds = {}
        model_probs = {}
        for r in group:
            label = r["outcome"].strip().upper()
            o = float(r["decimal_odds"])
            if not math.isfinite(o) or o <= 1.0:
                raise ValueError(f"{key}: invalid decimal odds")
            outcome_odds[label] = o
            if r["model_probability"].strip():
                p = float(r["model_probability"])
                if not math.isfinite(p) or p < 0 or p > 1:
                    raise ValueError(f"{key}: invalid model probability")
                model_probs[label] = p
        if model_probs and set(model_probs) != set(offered):
            raise ValueError(f"{key}: partial model probabilities are forbidden")
        if model_probs:
            if abs(sum(model_probs.values()) - 1) > 1e-6:
                raise ValueError(f"{key}: model probabilities do not sum to one")
            lock_time = a["model_lock_kst"].strip()
            lock_ref = a["model_lock_ref"].strip()
            if not lock_time or not lock_ref:
                raise ValueError(f"{key}: no external PRE LOCK provenance")
            if any(r["model_lock_kst"] != lock_time or r["model_lock_ref"] != lock_ref for r in group):
                raise ValueError(f"{key}: mismatched model LOCK provenance")
            if date_(lock_time) >= start_time:
                raise ValueError(f"{key}: POST_START_MODEL_LEAKAGE")
        status = a["settlement_status"].strip().upper()
        settled = a["settled_outcome"].strip().upper()
        if status not in SETTLEMENTS:
            raise ValueError(f"{key}: missing or invalid settlement status")
        if status == "SETTLED":
            if settled not in offered:
                raise ValueError(f"{key}: invalid winning outcome")
        elif settled:
            raise ValueError(f"{key}: push/void must not specify winning outcome")

        inverse = {name:1/odd for name,odd in outcome_odds.items()}
        overround = sum(inverse.values())
        if overround <= 0 or not math.isfinite(overround):
            raise ValueError(f"{key}: invalid overround")
        fair = {name: p / overround for name,p in inverse.items()}
        events.append({
            "game_pk": key[0], "market_id":key[1], "market_type":typ,
            "price_asof_kst":a["price_asof_kst"], "game_start_kst":a["game_start_kst"],
            "line":a["line"], "settlement_status":status, "settled_outcome":settled,
            "odds":outcome_odds, "fair_probabilities":fair,
            "overround":overround-1, "model_probabilities":model_probs or None,
            "model_lock_ref":a["model_lock_ref"] if model_probs else None,
        })
    # Same market may have OPEN / LATEST snapshots; do NOT count all as independent games.
    return sorted(events, key=lambda z:(z["game_start_kst"], z["game_pk"],z["market_id"],z["price_asof_kst"]))


def brier(p, observed):
    return sum((prob - (1.0 if label == observed else 0.0))**2 for label,prob in p.items())


def calibration_bins(events, model, width=0.1):
    bins = defaultdict(list)
    for ev in events:
        p = ev["fair_probabilities"] if model=="market" else ev["model_probabilities"]
        if not p or ev["settlement_status"]!="SETTLED":
            continue
        favorite = sorted(p, key=lambda name:(-p[name], name))[0]
        conf=p[favorite]
        ix=min(9,int(conf/width))
        bins[ix].append((conf,1 if favorite==ev["settled_outcome"] else 0))
    return [
        {"bin":f"{i/10:.1f}-{(i+1)/10:.1f}",
         "n":len(values),
         "predicted":sum(p for p,_ in values)/len(values),
         "observed":sum(y for _,y in values)/len(values),
         "gap":sum(y for _,y in values)/len(values)-sum(p for p,_ in values)/len(values)}
        for i,values in sorted(bins.items())
    ]


def analyze(events:list[dict])->dict:
    # Market snapshots are different availability horizons, not independent trials.
    by_type=defaultdict(list)
    for ev in events:
        by_type[ev["market_type"]].append(ev)
    results={}
    for typ, xs in sorted(by_type.items()):
        # Input can contain the OPEN & CLOSE of one market. Score each once per
        # (game_pk,market_id); earliest pregame snapshot is the primary universe.
        earliest={}
        for x in xs:
            k=(x["game_pk"],x["market_id"])
            if k not in earliest or date_(x["price_asof_kst"])<date_(earliest[k]["price_asof_kst"]):
                earliest[k]=x
        base=list(earliest.values())
        settled=[e for e in base if e["settlement_status"]=="SETTLED"]
        model=[e for e in settled if e["model_probabilities"]]
        block={
            "total_market_offerings":len(base),
            "price_snapshots":len(xs),
            "settled":len(settled),
            "push":sum(e["settlement_status"]=="PUSH" for e in base),
            "void":sum(e["settlement_status"]=="VOID" for e in base),
            "mean_overround":sum(x["overround"] for x in base)/len(base),
            "market_multiclass_brier":(
                sum(brier(e["fair_probabilities"],e["settled_outcome"]) for e in settled)/len(settled)
                if settled else None),
            "market_favorite_reliability":calibration_bins(settled,"market"),
            "model_comparison":None,
        }
        if model:
            baseline=sum(brier(e["fair_probabilities"],e["settled_outcome"]) for e in model)/len(model)
            candidate=sum(brier(e["model_probabilities"],e["settled_outcome"]) for e in model)/len(model)
            block["model_comparison"]={
                "n":len(model),"market_brier_same_rows":baseline,
                "locked_model_brier":candidate,
                "model_minus_market_brier":candidate-baseline,
                "locked_model_favorite_reliability":calibration_bins(model,"model"),
                "ROI_status":"NOT_COMPUTED_UNLESS_EXECUTABLE_PRE_GAME_PRICES_AND_SELECTION_POLICY_PROVEN",
            }
        results[typ]=block
    return {
        "status":"AUDIT_CALCULATED_NOT_PROOF_OF_CLEAN_PRE",
        "definition":"Opening-price no-vig benchmark per market; only settled non-refund outcomes scored",
        "model_lock_rule":"external ref/timestamp required, must independently authenticate ref's first commit or file hash",
        "dedup":"earliest quote per game_pk+market_id; distinct prices not separate games",
        "markets":results,
        "limitations":[
            "Odds margin removal is only the proportional no-vig method, not a unique unbiased probability estimate",
            "Two-way baseball tied games marked PUSH; calibration conditional on non-refunds",
            "Line market is scored against official sportsbook settlement, not guessed from PBP",
            "Missing historical odds and missing PRE LOCK cannot be inferred",
            "This tool does not imply real-money edge or calculate bet return at nonexistent fair odds"
        ],
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    report=analyze(load_events(args.input))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({t:{"settled":m["settled"],"brier":m["market_multiclass_brier"],
                         "model":m["model_comparison"]} for t,m in report["markets"].items()},ensure_ascii=False))


if __name__=="__main__":
    main()
