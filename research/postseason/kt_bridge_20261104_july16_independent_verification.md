# KT bridge research — independent 2026-07-16 split replay (2026-11-04)

Status: `RESEARCH_VERIFIED_DESCRIPTIVE_ONLY`, no Champion coefficient / PRE retroactive modification.

## Origin
User-provided Library bundle `KT_bridge_regular_season_20261104.zip`, original research commit `19e380317b27f7725175b1b59a291f7fa37ef520`.
Original reproducibility note: ZIP's `analyze.py` uses **July 1** for the names `first_half/second_half` whereas the accompanying **second** report explicitly uses **July 16**. The second report was independently replayed with the July-16 split, not treated as identical to the original ZIP code.
Immutable PBP sha256 `32c70d31c89c83a9e8606254ff6ec084f599371b6debc6dde36066bf122c74e1`, regular season 720 games, KT 144 games; exclude 2026-06-04 LG-KT ambiguous role-boundary attribution from role measures (KT **143** games).
Replayer `scripts/research_kt_bridge_20261104_july16_replay.py`, [Actions run 41786375442](https://github.com/darkhorce1227-cell/kbo-cloud-pbp/actions/runs/41786375442) **PASS**.

## KT runs charged to score occurring while each role is active (9-inning rates, not official earned-run ERA)
Actual active role: `SP`; `BRIDGE` from SP exit to **first Park Young-hyun PA**; `PARK` Park Young-hyun; `AFTER_PARK` beyond. All games including those Park did not play are preserved.

| Role (PA counted only when pitcher initially entered with 0 outs & bases empty) | Pre-Jul16 82 games | Jul16+ 61 games | Combined 143 |
|---|---:|---:|---:|
| SP | **4.45191** (216 R/1310 outs) | **3.08496** (117 R/1024 outs) | 3.85219 |
| BRIDGE | **4.27624** (86 R/543 outs) | **4.35246** (59 R/366 outs) | 4.30693 |
| PARK | **1.09459** (3 R/74 outs) | **1.28571** (2 R/42 outs) | 1.16379 |

| Unfiltered role | Pre-Jul16 | Jul16+ |
|---|---:|---:|
| SP | 4.45191 | 3.08496 |
| BRIDGE | **6.40431** (176 R/742 outs) | **5.07182** (102 R/543 outs) |
| PARK | 3.55263 | 4.18310 |

Leading bridge entries and blown *intermediate* leads: 11/41 (26.8%) → 6/38 (15.8%). Not necessarily final losses.

The clean-start split uses the starting 0 out/bases-empty state of each reliever stint: it reduces inherited-runner contamination but does not control opponent lineup quality, parks, score state, usage, or in-game manager selection. The small Park clean sample is 3 runs in 24⅔ innings then 2 runs in 14 innings; interpret the near-1 RA9 as descriptive rather than stable individual talent estimate.

## Full 2026 report in user-provided ZIP
- KT mid-bridge whole-regulation role RA9 **5.84** overall vs SP **3.85**, Park **3.79**.
- On 0out/empty-base stint entry, KT bridge **4.31**, Park **1.16**.
- League-comparable inning6–8 non-start/non-closer rates: KT **5.94** vs other nine teams innings-weighted **5.91**. No significant league-exception evidence.
- Pre-date opponent starter strong: 38 KT games bridge **4.74** vs 61 otherwise qualified games **6.89**, non-significant given game-cluster CI (strong minus other ~−2.16, CI [−4.50,+0.23]). No independent KT bridge penalty justified by opponent's *pregame* strong SP label.
- Actual good SP performance 6+IP and <=2 runs is postgame outcome-selected. Do not use in PRE or as causal grouping.

## Model implication — not a fitted coefficient

Previous descriptive hypothesis `KT_BP_BRIDGE_IS_LEAGUE_BAD` -> **NOT SUPPORTED**. More precise statement: `KT_SP_STRONGER_LATE_SEASON; BRIDGE RELATIVELY WEAKER TO SP AND PARK`.

Do **not**:
- install unconditional KT bullpen penalty vs league,
- add an extra penalty just for strong opposing starter,
- turn within-team group RA9 differences into plug-in run contributions,
- treat pitcher-on-watch team scored runs as official individual earned runs,
- claim 1 game's KS1 8-7 demonstrates a fixed late-game collapsibility factor.

Next prospective Challenger only:
1. `PROJECTED_SP_OUTS` with uncertainty and early hook branch.
2. `PROJECTED_BRIDGE_OUTS` and per-player availability/role weights from **pregame** rosters, recent pitches and usage.
3. `P_PARK_USED` conditional on simulated inning/score and Park availability; keep no-Park appearances.
4. Calibrate hierarchical player-level K-BB, HR, BB and run risk, separating inherited-runner states and exposure.
5. Chronological OOS compare F5→Full side, FG run total, late reversal, full score NLL and Brier against frozen pregame operational model; game-level bootstrap CIs and market price matching.
6. Champion remains UNCHANGED unless prospective performance passes promotion gate.

## Revision tracking
July16 independent replay exact positive controls:
- games 82 + 61;
- starter rates 4.45→3.08;
- clean bridge rates 4.28→4.35;
- clean Park rates 1.09→1.29;
- unfiltered bridge 6.40→5.07.
