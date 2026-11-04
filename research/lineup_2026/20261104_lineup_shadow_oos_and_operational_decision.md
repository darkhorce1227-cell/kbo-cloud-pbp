# 2026 KBO confirmed-lineup scoring research — 2026-11-04

**Decision: MARKET-BLIND RESEARCH ONLY — NO CHAMPION PROMOTION; ACTIVE READ-ONLY LINEUP AUDIT.**

## 1. What actually exists in the repository
- Original daily Operational Champion μ, including historical lineup/park/CQ/manual overlays, is **not fully source-reconstructible** from the current main tree. There is no `Champion` source model artifact and no validated numeric `confirmed-lineup → F5/FG μ` update function.
- Reproducible `research/mu-v1.2/model_spec.json` and `research/mu-v1.3/model_spec.json` are separate `RESEARCH_CANDIDATE_NOT_CHAMPION` models; they are not exact original Champion.
- More advanced `mu-v1.2-recovered-challenger-v0.9-bp30-cal36-p25.5` is `RESEARCH_CHALLENGER_ACTIVE` and `official_shadow_allowed=false`. The calendar36/P25.5 reconstruction is strong but original preprocessing code is lost; its preregistered prospective gate is collecting evidence and MUST NOT be relaxed/retuned to this study.
- Current `state/mechanical_mu_input.json` remains target date **2026-10-20**, not a valid captured official KS2 Champion. An archived input is not an automatically regenerable Operational Champion.
- Accordingly no reproducible **KS2 lineup-adjusted official Champion** or Official Raw/Max TOP5 can truthfully be claimed. The market-exposed 11/04 research does not count as a fresh official price-blind PRE.

## 2. Retrospective 2026 single-season lineup proxy evaluation

**Data:** SHA256 `32c70d31c89c83a9e8606254ff6ec084f599371b6debc6dde36066bf122c74e1`, validated 2026 PBP available before KS2, 224,611 pitch-level rows. For backtest, exclude postseason and retain 720 RS games through Oct12. All 720 games have recoverable first nine unique PA batters per team by inning 4 and checked F5/Full scoreboard; however **these lineups are reconstructed from completed game PBP, not pre-timestamped official lineup announcements**. This is a retrospective proxy and cannot qualify as prospective PRE/LOCK evidence.

- For each target date, team/player history is constructed **only from earlier completed dates**; same-date games do not enter features of other games that date.
- Team-only reference features: team prior long/30-day scored-run ratios, opponent prior long/30-day allowed-run ratios, home flag.
- Challenger adds lineup-order-weighted hitter historical hit/PA, total-bases/PA, HR/PA, BB+HBP/PA, strikeout/PA, each derived strictly prior to game; 120-PA pseudo-sample toward observed league rates; frozen weights favor earlier batting slots.
- Both are standalone Poisson GLMs (scikit-learn PoissonRegressor alpha 0.10, train-only feature scaling), **not original μ-v1.2 coefficient modifications**.
- Fixed temporal split: training **May1–July15 287 games/574 team-games**; test **July16–Oct12 296 games/592 team-games**, same rows for both. Game-level paired bootstrap 5000 resamples.
- Both F5 and Full counts from mechanically deduplicated PBP PA post-score; approximate Poisson score NLL and independent-Poisson 3-way side model diagnostic (not official NB2 + overtime rules).

| Metric (lower=better) | Team baseline | + lineup proxy | Interpretation |
|---|---:|---:|---|
| F5 score Poisson NLL | **2.4407070** | 2.4652835 | worsens +0.0245764 |
| F5 WDL logloss | **1.0147996** | 1.0405674 | worsens |
| F5 score 3+ Brier | **0.2547705** | 0.2628496 | worsens |
| Full score Poisson NLL | **2.8449023** | 2.8931848 | worsens +0.0482825 |
| Full WDL logloss | **0.8616096** | 0.9005270 | worsens |
| Full score 3+ Brier | **0.1993234** | 0.2023372 | worsens |

**Game-cluster 95% bootstrap CI for lineup-minus-team NLL (higher=worse):** F5 **[+0.0011612, +0.0484946]**, Full **[+0.0156048, +0.0809123]**. These specific five raw-hitter features hurt heldout prediction in this isolated study. Multiple model architectures/datasets may differ; one 2026 season does not prove lineups don't matter.

**Audit:** [GitHub Actions 41807329219](https://github.com/darkhorce1227-cell/kbo-cloud-pbp/actions/runs/41807329219) **SUCCESS**, code `scripts/research_lineup_pre_date_oos_2026.py` and machine JSON output `lineup_shadow_oos.json` retained in that action's artifact.

## 3. Read-only CONFIRMED LINEUP / ASOF module — SUCCESS
- Script: `scripts/kbo_lineup_pregame_audit.py`
- Input: `research/lineup_2026/ks2_confirmed_20261104.json` from user-supplied lineup graphic.
- Output: data cutoff strictly before game date and PBP SHA verification, identity check, 9 slots per team, bat side from previous observed PA, seasonal / last 30-day per-player H, TB, HR, BB+HBP, SO and samples; **does not alter μ**.
- Replayed on KS2 as-of Nov4 2026, [GitHub Actions 41807780458](https://github.com/darkhorce1227-cell/kbo-cloud-pbp/actions/runs/41807780458) **SUCCESS**.
- Samsung 9/9 matched, 7 L / 2 R, L at all slots 1–6; KT 9/9 matched, 5 L / 4 R; lineup 2026 PAs total 4289 Samsung, 3984 KT. This describes player histories, not inferred matchup advantage.
- Today's opposing starter throwing hand must come from an auditable roster/player source, not the batting `stand` column alone; hitter vs pitcher handedness is **NOT QUANTITATIVELY IMPLEMENTED** in this prototype.
- `LINEUP_PROBABILITY_CORRECTION = UNAVAILABLE`; no retroactive lineup adjustment after odds were inspected; no official betting ledger entries.

## 4. Why the test may fail (hypotheses, NOT verified causes)
- Team run rates already capture much of the regular lineup talent, causing double-counted or noisy redundant information.
- Raw hit/TB/HR and K are highly correlated, and 574 train team-games may not safely support five added inputs.
- Regular season 2026 only, early player sample and substitution/park/health confounding.
- PBP-reconstructed starting nine is a useful proxy but lacks independent officially timestamped publication evidence.
- Pitcher throws, genuine batter-vs-pitcher handedness, defense, price, game roster, expected PA, CQ/EV/LA, and substitution risks not jointly modeled.

## 5. Approved next research
1. **Longitudinal multi-year data**: historical 2024–25 pregame official lineup logs and player PA/PBP, verified snapshots; GitHub Releases currently carry 2026 validated PBP, historic PS, and PvB aggregates, but not a listed full historical RS lineup/PBP binary. Do not fabricate availability.
2. **Low-dimensional player delta**: compare expected lineup quality relative to a team's *normal expected line-up*, not absolute batter rates; avoid using July–Oct 2026 heldout repeatedly for hyperparameter shopping. Freeze method before new independent forward holdout.
3. **Pitcher handedness & split**: source reliable pitcher throws and platoon strikeout/walk/HR prior data. First test no-lookahead end-to-end with historical announced lineup availability.
4. **Official model restoration**: locate original manually produced Operational Champion μ rows/coefficients and timestamped input decisions; if unobtainable, write a **new separately named reproducible model**, run time-ordered multi-year tests, then apply explicit promotion gate. Do not relabel the recovered v0.9 as exact Champion.
5. **Betting objective**: evaluate F5 / FG score NLL, WDL Brier/logloss, full totals, market de-vigged edge, uncertainty, and game-cluster bootstrap, with Clean PRE/LOCK timestamps and no opportunistic postprice adjustments.

`DECISION: AUDIT_ONLY_TO_MAIN; FAILED_SCORING_FEATURES_CHALLENGER_REJECTED; CHAMPION_UNCHANGED; PS PRACTICAL_RAW_MAX_ACTUAL_UNCHANGED.`
