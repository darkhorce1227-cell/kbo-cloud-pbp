# KBO bullpen bridge / closer — operational audit contract (2026-11-04)

**Status: ACTIVE BASEBALL-INPUT AUDIT; PROBABILITY COEFFICIENT NOT PROMOTED.** This document is a durable input/checklist rule, NOT a modification of the frozen Champion, mu-v1.2 recovered challenger, market-blind PRE, or PS official betting ledgers.

## Confirmed research findings
Source: user-supplied `KT_bridge_regular_season_20261104.zip`, validated 2026 regular-season PBP (720 games), one KT ambiguous pitcher-boundary game excluded, **143 KT games**. Dataset SHA256 `32c70d31c89c83a9e8606254ff6ec084f599371b6debc6dde36066bf122c74e1`.

- KT bridge is from starter exit until first Park Young-hyun appearance; in no-Park games the role continues to game end. In raw scoring-on-mound rates, SP 3.85, bridge 5.84, Park 3.79 R/9; these are not official individual ERAs.
- For reliever **stints entered with zero outs and empty bases**, bridge 4.31 vs Park 1.16 R/9. Park subgroup is limited (5 runs over 38⅔ innings), so do not adopt 1.16 as future true talent.
- KT 6–8 inning non-starter, non-primary-closer R/9 = 5.94, other 9 teams combined = 5.91: **no evidence KT bridge is unusually bad by league-standard exposure.** Comparisons based on full-season closer identities are descriptive only, not pregame predictors.
- Verified July16 split: starter clean 4.45→3.08; bridge clean 4.28→4.35; Park clean 1.09→1.29. Unfiltered bridge 6.40→5.07. This is **relative gap widening**, not worsening middle-reliever performance.
- Strong opposing **pregame** SP is not associated with larger KT bridge run risk in this sample (strong: 4.74 vs other qualified: 6.89 R/9). Never install a special `OPPOSING_STRONG_SP × KT_BRIDGE` penalty. Actual opposing 6IP/2R grouping is postgame-selected, invalid as PRE.
- Strong baseball narratives can coexist with prediction failure: 2026 season 6–8 run prediction study, trained through July15 (287 games), tested Jul16–Oct12 (296 games), with strictly prior-date features:
  - league-only Poisson NLL **2.16691**, Brier for allowing ≥2 runs **0.25976**;
  - prior offense/defense **2.17101 / 0.26807**;
  - + candidate 6–8 reliever K-BB and HR quality **2.18205 / 0.27008**;
  - + 2-day ≥40 pitch-row workload share **2.18376 / 0.27043**.
  - Quality vs offense/defense NLL delta **+0.01105**, game-bootstrap 95% CI **[+0.00398,+0.01855]** (worse in this exact research architecture).
  - Conclusion: **REJECT ADDITIVE NUMERIC BRIDGE-QUALITY OR WORKLOAD COEFFICIENT FOR NOW**. This does not prove bullpen quality/workload is unimportant; candidate roster and workload proxies are crude and the 2026 one-season test was developed retrospectively.
- Live at first bridge entry (not PRE): historical KT Park use: lead 1–3 runs **31/46=67.4%**, ≥4 lead **7/33=21.2%**, not leading **18/63=28.6%**. Jul16+ out-of-time simple state-bucket predictor Brier 0.23472→0.20710 (n=60), paired CI for delta **[−0.06007,+0.00627]** includes zero. Treat as **LIVE SHADOW / WATCH**, not a locked probability.

## Mandatory daily / postseason audit fields
For **each of both teams**, capture before FIRST price inspection, as supported:
`DATA_ASOF` (strictly earlier completed game dates); `STARTER_ID` / `EIP range` / hook distribution including early branch; `PRED_SP_OUTS`; `PRED_FIRST_FOLLOWER_CANDIDATES`; `PRED_BRIDGE_OUTS`; `PRED_8TH_9TH_RELIEVERS`; `CLOSER_ID/AVAILABILITY`; actual prior 1/2/3-day **pitch-row counts** for each candidate; `PREGAME_ACTIVE_ROSTER_STATUS`; `K_BB/HR sample support`; `BRIDGE_QUALITY_UNKNOWN` if unsupported.

- The past 35-day 6–8 inning appearance counts are only a **candidate-generation proxy**, not the actual active roster, definitive first-follower identity, official pitch count, or calibrated skill.
- **Mandatory role-switch positive control:** KS1 2026-11-03 KT first follower Logan (ID 55912) was missing from a relief-only candidate proxy, because he had appeared as a starter previously. Therefore maintain a **separate prior-starter → long-relief/tandem shadow pool**, excluding today's confirmed starter. As-of-date replay PASS: [workflow 41792661379](https://github.com/darkhorce1227-cell/kbo-cloud-pbp/actions/runs/41792661379). Do not treat ex-starter shadows as confirmed available; actual current roster and manager plan decide eligibility.
- If the exact expected usage path is unidentifiable, **widen uncertainty / label UNAVAILABLE**. Never fill with a guessed player or fake probability.
- Do not interpret `FRESH` as `GOOD`, nor automatically classify a pitcher as A/B/C tier from score-state alone.
- Score state can influence who gets selected. At PRE, scenario-average over *predicted* score states. A **realized** F5/bridge-entry lead and actual first follower belong only to a separate in-game update; never revise historical PRE with them.
- Separate inherited-runner events from pitcher-on-mound team runs; use official individual runs/ER only with auditable runner attribution. Distinguish same-pitcher fatigue (H1-A) from reliever-pool composition/availability (H1-B).

## Integration / promotion gates
1. Run the read-only `scripts/kbo_bullpen_pregame_audit.py` when the source is available; output bullpen candidates and strictly pregame workload without changing model μ.
2. For any new quantitative Challenger, freeze inputs and challenger code **before** future games; compare F5/Full probabilities, scoring likelihood, sides, totals, late-score distribution, and current price on matching timestamps. Use game-cluster intervals and repeat across years/teams.
3. A better identified bridge path alone is **not** a demonstrated betting EV. Keep `Champion=UNCHANGED`, `Raw/Max/Practical/Actual=UNCHANGED`; use model conflict audit and realistic sizing.
4. In 2026 postseason, first-follower and live path models remain research-only; existing PS probability promotion gate stays in force.

## Reproducibility
- [July16 independent replay](https://github.com/darkhorce1227-cell/kbo-cloud-pbp/actions/runs/41786375442)
- [2026 6–8 run chronological study](https://github.com/darkhorce1227-cell/kbo-cloud-pbp/actions/runs/41790793197)
- [Park live-usage chronological study](https://github.com/darkhorce1227-cell/kbo-cloud-pbp/actions/runs/41791551275)
- All new modeling code on dedicated research branch `research/kt-bridge-pregame-20261104`, no probabilistic production promotion.
