# KBO PS starter → first follower → later bullpen: actual PBP path and temporal test
Research date: 2026-11-02 KST

## Status
`PBP_SCORE_AUDIT = PASS`; `RESEARCH_PROVISIONAL`; **NO Champion promotion**.

Immutable source: GitHub release `historical-ps-master/postseason_screen_pass.parquet`.
Starter identity: `historical_ps/ps_starter_appearances.csv`.
RS-to-PS player link: `research/postseason/ps_rs_starter_avg_ip.csv`.
Runner: `scripts/research_ps_stage4_hook_to_runs.py`.
Workflow verification: [Actions run 41465164201](https://github.com/darkhorce1227-cell/kbo-cloud-pbp/actions/runs/41465164201).

## Complete actual score-path reconstruction
- **2016–2025: 143 PS games, 286 defense-team games**, 43,100 pitch rows, 11,126 deduplicated PA.
- In all 143 games, reconstructed scored runs sum to observed final score; starter+first follower+later bullpen = opponent runs; F5 + late = full runs.
- Mid-PA pitcher changes attribute PA scoring to the final-effective pitcher. Scoring while active is **not** official R/ER or runner-responsibility attribution.
- Excludes historically unavailable/unscreened games; manual partial PA reconstructions are **not** merged.

| Actual starter duration | n | Runs in F5 (opponent) | Full opponent runs | Runs against starter on watch | Runs against first follower | Runs against later bullpen |
|---|---:|---:|---:|---:|---:|---:|
| <3 IP | 38 | 4.974 | 6.368 | 2.684 | 1.342 | 2.342 |
| 3 to <5 IP | 78 | 4.038 | 6.462 | 2.756 | 0.897 | 2.808 |
| >=5 IP | 170 | 1.200 | 3.235 | 1.571 | 0.400 | 1.265 |
| All | 286 | 2.476 | 4.531 | 2.042 | 0.661 | 1.829 |

**Critical reverse causality**: realized poor scoring makes early hooks more likely. These conditional differences are descriptive, not causal run multipliers. First-follower exposure and inning, inherited runners, manager decisions, and role/skill all confound the differences.

## Time-split, pregame-only prototype

- Training 2016–2020: 139 starter appearances with matched same-season RS avg starter IP.
- Evaluation 2021–2025: 145 appearances, **72 complete paired games** (one unmatched starter's game excluded).
- Pooled baseline: training mean PS team scoring, Poisson+Skellam.
- Direct RS-IP model: ridge Poisson rate regression using same-season RS starter mean IP, no branching.
- Branch mixture: estimated per-starter `P(<3 IP), P(3 to <5 IP), P(>=5 IP)` from training PS minus RS-IP empirical residuals; branch means estimated only from training score outcomes with 15-team-game pseudo-count shrinkage. Then marginalize to pregame expected team runs. **No realized hook state is used at test prediction time**.
- These are weak explanatory models, **not the user's full Champion**. No hitters, lineups, ballpark, reliever names, opponent skill, weather or season run environment control. Poisson scoring is a simple benchmark and is not claimed to be the best distribution.

| 2021–2025 metric (lower better) | Pooled | Direct RS-IP | Hook mixture |
|---|---:|---:|---:|
| Full two-team score Poisson NLL | 5.30237 | 5.27710 | 5.26872 |
| Full home-win binary Brier | 0.260473 | 0.251886 | 0.250806 |
| Full O8.5 Brier | 0.250135 | 0.249687 | 0.248732 |
| F5 two-team score Poisson NLL | 4.78828 | 4.78979 | 4.81861 |
| F5 home-win binary Brier | 0.248555 | 0.249762 | 0.255417 |
| F5 O4.5 Brier | 0.251069 | 0.250496 | 0.254426 |

Test scoring totals: Full 9.444 actual vs 8.662 pooled, 8.962 direct, 8.869 hook; F5 5.264 actual vs 4.590, 4.652, 4.794.

### Paired game-cluster bootstrap (12,000 repetitions)

Hook minus direct RS-IP (negative is favorable):
- Full score NLL: **−0.008379**, 95% CI **[−0.041110,+0.021640]**.
- Full home-win Brier: **−0.001079**, CI **[−0.007545,+0.004953]**.
- Full O8.5 Brier: **−0.000955**, CI **[−0.007578,+0.005192]**.
- F5 score NLL: **+0.028822**, CI **[−0.059453,+0.119227]**.
- F5 home-win Brier: **+0.005655**, CI **[−0.009865,+0.021356]**.
- F5 O4.5 Brier: **+0.003930**, CI **[−0.010235,+0.018565]**.

All six CIs cross zero: incremental value of branching beyond direct RS IP is **not established**. The comparisons are exploratorily specified after the broader data were known; this is a held-out year split, **not** prospectively frozen independent validation.

## Inference and next gate
1. Validated: reproducible pitcher-path **score-event accounting** (not pitcher's official ER).
2. Weak evidence: Full score/side/total slightly better than historical pooled/direct baseline, F5 worse.
3. **Do not treat the historical observed 6.37 vs 3.24 run contrast as a predictive early-hook run penalty**. It is outcome-selected.
4. Next stage: first follower **identity, quality (K-BB/HR), relief availability, inning/out/runners at entry and leverage**, plus relative opponent offense and projected hook probability **using only pregame-available features**.
5. Compare **paired Champion vs Challenger** F5 three-way probabilities and FG two-way probabilities, scores and totals, with one immutable pregame snapshot and game-cluster validation; no betting ROI/EV without timestamped actual odds.
