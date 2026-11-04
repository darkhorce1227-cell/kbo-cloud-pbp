# 2026 μ-v1.3 F5→Full scoring path consistency and coherent late challenger
**Date:** 2026-11-04. **Decision:** `STRUCTURAL_GUARDRAIL_VALIDATED; COHERENT_LATE_RESEARCH_CHALLENGER; NO_CHAMPION_PROMOTION`.

## Inputs and reproducibility
- Frozen 2026 original PBP through 2026-11-03: 224,611 pitch/event rows; SHA256 `32c70d31c89c83a9e8606254ff6ec084f599371b6debc6dde36066bf122c74e1`.
- Frozen model coefficient files `research/mu-v1.2/model_spec.json` and `research/mu-v1.3/model_spec.json`. For 2026 feature extraction, this study uses **2026 history only**, NOT full 2024–25 historical priors. Explicitly NOT the recovered v1.2-v0.9 and NOT original operational Champion.
- Target-game outcomes never used for feature construction; each game uses earlier dates. Market prices never used to fit or select coefficients.
- Stage1 structural audit, July 16–Oct 12 (296 games): 276 games have enough prior starting-appearance data to compare BOTH v1.2 and v1.3; 20 games excluded from two-model comparison because v1.2 research feature constructor rejects a brand-new opposing starter. [Reproducibility Actions 41808459261](https://github.com/darkhorce1227-cell/kbo-cloud-pbp/actions/runs/41808459261).
- Stage2 coherent variant uses v1.3's shrunk cold-start starter features, so all **296** late games are eligible, even where v1.2 cannot be computed. [Actions 41808791834](https://github.com/darkhorce1227-cell/kbo-cloud-pbp/actions/runs/41808791834). Early-period May1–Jul15 **287** games [Actions 41808973250](https://github.com/darkhorce1227-cell/kbo-cloud-pbp/actions/runs/41808973250). Original v1.3 F5 μ kept fixed throughout.

## Hard physical constraint
For any single game and team, `E[Full Runs] >= E[F5 Runs]`, as scored runs cannot decrease. The old v1.3 creates **F5 and Full team shares independently** using different `gamma`-powered submodel ratios. This does not enforce the constraint, even if both game-total means are positive.

| Diagnostic, July16–Oct12 276 eligible games | v1.2 | v1.3 |
|---|---:|---:|
| Games with at least one team Full μ < F5 μ | **0** | **1** |
| Team-games with 6+ late μ < 0.25 | 0 | **8** |
| Games with opposite F5/Full projected margin leaders | 2 | **47** |

- Hard-invalid specific game `20260822HTWO02026` (Hanwha away vs Kiwoom home): original v1.3 predicted Kiwoom F5 μ **3.87268** and Full μ **3.10078**, implying late μ **−0.77191**, impossible for any nonnegative joint scoring path. Another issue: that game predicted Hanwha F5 1.747 / Full 7.609 and hence extreme late swing, then actually Hanwha scored 1, Kiwoom 3.
- A projected-margin lead flip alone is **not mathematically invalid**; it can reflect legitimate comeback chances. But the 47 vs 2 contrast makes v1.3's phase allocation clearly worthy of scrutiny. Do NOT identify its frequency as definitive evidence of an actual empirical error without backtest.

## One deterministic coherent research alternative (NOT tuned to 2026)
- Preserve old v1.3 F5 team scores `f5a,f5h`.
- Preserve full GAME scoring level `f5_total + simple_late_raw_away + simple_late_raw_home`.
- Compute `L = simple_late_raw_away+simple_late_raw_home` and natural positive `share_away = core_late_raw_away/(core_late_raw_away+core_late_raw_home)`.
- Set `full_away = f5a + L*share_away`; `full_home = f5h + L*(1-share_away)`.
- No new learned parameters or implicit `gamma_late`. Guarantees nonnegative team late-scoring means and preserves F5 marginals and Full game-total mean. It **does change Full side probabilities**, so cannot be silently merged into a frozen model or a pre-existing pre-price snapshot.

## 2026 period-separated diagnostic

| Period | Model | Full score Poisson NLL ↓ | Full WDL NB2 logloss ↓ |
|---|---|---:|---:|
| 2026 May1–Jul15, 287 games | Original v1.3 | 2.9386560 | 0.8710245 |
| | Coherent challenger | **2.8283669** | **0.8167583** |
| 2026 Jul16–Oct12, 296 games | Original v1.3 | 3.1345409 | 0.9089915 |
| | Coherent challenger | **2.9354956** | **0.8429954** |

- Early period delta new−old: score **−0.1102891**, 95% game bootstrap **[−0.1675772,−0.0535223]**; Full WDL logloss **−0.0542662**, 95% CI **[−0.0909209,−0.0175925]**.
- Late period delta new−old: score **−0.1990453**, 95% game bootstrap **[−0.2787841,−0.1250008]**; Full WDL logloss **−0.0659961**, 95% CI **[−0.1096564,−0.0246245]**.
- The new variant is a **post-diagnostic** design derived after observing 2026 path inconsistency. These are separate periods within the SAME season but not truly preregistered future-game independent evidence. Do not call them Championship-grade OOS validation. Original v1.3 was fit to 2024–25, so inspecting 2025 is not an independent prospective test of its performance.
- Scores evaluated with Poisson count NLL; Full side evaluated using NB2 alpha=0.322 (same full overdispersion parameter as original research model), numeric support 0..44. Postseason extras, official tie/void rules and Betman payouts are not part of this study.

## Correct operational implementation
- **IMMEDIATE validated guardrail:** Any paired F5/Full joint simulation must check `HF>=H5, AF>=A5` on all paths, each team `μFull>=μF5`; invalid outcomes must be marked `DISTRIBUTION_INCONSISTENT` and cannot produce official odds/EV.
- **NO coefficient activation:** Do not replace original Champion, old v1.2 recovered challenger-v0.9 or v1.3 frozen μ in prior snapshots.
- **NEXT research:** independent published lineup/roster and bullpen time-series, cross-year cold-start first-start checks; freeze coherent alternative BEFORE prospectively comparing future 2027 F5/Full joint proper scoring, R/G, WDL, totals and matched actual market prices.
- **KS2 2026-11-04:** market prices were already inspected before this variant was developed. **Any new coherent prediction is postprice EXPLORATORY only**, NOT a pregame LOCK, cannot be used to retroactively generate official bets.

`CHAMPION_UNCHANGED; PRICE_SNAPSHOTS_UNCHANGED; PRACTICAL_RAW_MAX_ACTUAL_UNCHANGED; COHERENT_LATE=CHALLENGER_EXPLORATORY`.
