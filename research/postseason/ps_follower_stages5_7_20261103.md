# PS follower-path research stages 5–7 — verified historical replay (2026-11-03)

**Status:** RESEARCH ONLY · NO CHAMPION PROMOTION · official PRE LOCK and existing betting ledgers unchanged.

## Sources and reproducibility

- Repository `darkhorce1227-cell/kbo-cloud-pbp`, isolated branch `research/ps-follower-runs-20261102`.
- Immutable pitch input: GitHub release `historical-ps-master/postseason_screen_pass.parquet`.
- Historical production coverage: 2016–2025 **143 postseason games, 286 team/game pitcher paths**, with original source and score-state hard audit from Stage4.
- Scripts:
  - `scripts/research_ps_stage5_first_follower.py`
  - `scripts/research_ps_stage6_bullpen_history.py`
  - `scripts/research_ps_stage7_live_at_hook.py`
- Successful independent GitHub Actions runs:
  - [Stage5 first-follower recurrence](https://github.com/darkhorce1227-cell/kbo-cloud-pbp/actions/runs/41553327093)
  - [Stage6 reliever pregame quality / run model](https://github.com/darkhorce1227-cell/kbo-cloud-pbp/actions/runs/41553723363)
  - [Stage7 live hook-inning conditional](https://github.com/darkhorce1227-cell/kbo-cloud-pbp/actions/runs/41554030120)

## Stage 5: pregame first-follower ID

Inputs: only **earlier calendar-date games of the same team in the same postseason season**. The true future first reliever, true hook timing and same-date results are not prediction inputs.

Of 286 total team-games, **235** have ≥1 earlier same-team postseason game; 234 actually had a first follower. Holdout **2021–2025 = 120** eligible team-games; 96/120 (80%) first followers had already appeared as a reliever in an earlier postseason game of the same team, but **24/120 (20%) were outside that roster proxy**.

| Ranker | Holdout Top1 | Holdout Top3 |
|---|---:|---:|
| Previous game first follower | 16/120, **13.3%** | 60/120, 50.0% |
| Most frequent first follower | 20/120, 16.7% | 58/120, 48.3% |
| Most recent relief appearance | 17/120, 14.2% | 41/120, 34.2% |
| **Historical role weighted**: 2×FIRST + 0.5×LATER | **24/120, 20.0%** | 54/120, 45.0% |
| Role weighted, 2-day pitches discounted | 19/120, 15.8% | 59/120, 49.2% |

Role-weighted Top1 compared to previous-game first follower is +6.667 percentage points, paired game-cluster bootstrap 95% CI **[+0.820,+13.333] pp** (12k). **Exploratory multi-rule selection:** this isolated CI excludes 0 but does not establish stable improvement after trying several rules; no prospectively frozen benchmark.

Top1 20% is too low to use one projected follower as certain, and pregame candidate pool misses 20% of actual first followers. Explicit roster and regular-season reliever-history joins are needed.

## Stage 6: do actual prior bullpen scoring histories improve pregame probabilities?

2016–20 model fit: **139** starters matched to current RS starter average IP. 2021–25 testing: **145** matched starter appearances, **72 complete paired games**, of which only **51/72** had earlier-postseason relief data on BOTH teams. All historical reliever scored-runs-per-PA, predicted top1 first-follower quality, and 2-day workload come only from same-team, same-season, earlier calendar dates, shrunk to training-only relief mean with 30-PA pseudo-count. Simple regularized Poisson mean models; **NOT the real Champion**.

| Tested model | Full home-win Brier | F5 W/D/L LogLoss | Full O8.5 Brier |
|---|---:|---:|---:|
| RS starter IP only | .251886 | 1.003387 | **.249687** |
| + previous relief team run/PA | .252616 | 1.001336 | .250598 |
| + prospective first follower run/PA | .246169 | 1.005071 | .253406 |
| + both reliever histories | .246127 | **.999538** | .254934 |
| + first follower previous 2-day pitch count | **.243892** | .998413 | .259955 |

For `+ both` versus starter-only, game-cluster bootstrap (10k):
- Full home-win Brier Δ **−.005759**, CI **[−.018069,+.007091]**.
- F5 3-way LogLoss Δ **−.003849**, CI **[−.038525,+.031345]**.
- Full O8.5 Brier Δ **+.005247**, CI **[−.006720,+.017550]**.

Every tested gain/loss has wide intervals crossing zero. Neither group-quality nor candidate-specific quality is proven to improve price-side probabilities; the Under/Over trends may worsen. A reliever's scoring event while active is not a pitcher ER, causal talent estimate, nor a shift to Champion.

## Stage 7: conditional on live-known hook inning

The LIVE-at-hook experiments additionally use the **actual first-follower entry inning** when considering which pitcher to enter. This is information already observable **at the replacement moment** but impossible to know at initial pregame PRE. Accordingly, it can only be used for a separately defined in-game module, **never backfilled into PRE**.

Holdout 2021–25 = 120 eligible follower appearances:

| Conditional ranking | Top1 | Top3 |
|---|---:|---:|
| Pregame role-frequency baseline | 24/120 = 20.0% | 54/120 |
| Similar past entry-inning kernel | 22/120 = 18.3% | 63/120 |
| Similar early/late first-follower bucket | 28/120 = 23.3% | 53/120 |
| First-follower only + entry-inning recency | 25/120 = 20.8% | 55/120 |

Entry bucket vs pregame baseline Top1 Δ **+3.33 pp**, 10k game-cluster bootstrap CI **[−2.36,+9.17] pp**. Zero is included.

In the **53/120** early-follower entries (≤5th inning), role baseline Top1 **6/53=11.3%**, inning-bucket Top1 **10/53=18.9%**. This is hypothesis-generating: the sample is small and context conditioning does not yet improve authenticated betting probabilities.

## Concluding judgment

- **Supported:** historical post-starter path, exact entering reliever, score-events and F5/full segmentation can be reconstructed; selecting an individual first follower pregame from previous PS role alone is very unreliable.
- **Weak/unstable:** role-weighted first-follower ranker is somewhat better in one retrospective comparison but candidates/feature choices remain exploratorily designed.
- **Not supported:** generic PS early-hook run premium, generic first follower talent penalty, unconditional 2-day fatigue penalty, or Champion probability adjustment.
- **Next high-EV data engineering:** join official **pregame active roster**, regular-season **reliever appearance and prior rolling workload** (same pitcher previous 1/2/3 days, at each target timestamp), roles/leverage context, and current defense/runner-state. Then test actual projected multi-reliever availability/quality mixture against **frozen Champion** on matched game dates. Do not accept improvements from actual future follower ID/hook time or postgame-selected pitchers.
- Full betting ROI/EV is **UNAVAILABLE** without the genuine contemporaneous price and properly frozen model probability.

### Operational note

The GitHub branch is intentionally non-production. Research scripts and successful workflow runs are preserved on that branch; **main was not modified**.
