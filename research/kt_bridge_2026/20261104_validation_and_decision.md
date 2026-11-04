# KT bridge / Park research: 2026 validation and integration decision

As-of: 2026-11-04 KST. **Research code + audited input-checklist only; Champion probabilities / official ledgers untouched.**

## Data
- Source: original user-provided `KT_bridge_regular_season_20261104.zip` (PBP + game score audit), validated 2026 PBP binary SHA `32c70d31c89c83a9e8606254ff6ec084f599371b6debc6dde36066bf122c74e1`.
- All 2026 regular-season: **720 games**; KT 144 games, one role-boundary ambiguity game `20260604LGKT02026` excluded from clean role comparisons (**143 KT games**).
- Reproduction of official 7/16 split separately PASS [workflow 41786375442](https://github.com/darkhorce1227-cell/kbo-cloud-pbp/actions/runs/41786375442). Original user ZIP `analyze.py` does **not** itself use that split; uses July 1.

## Role decomposition (team runs scored during appearance, not individual ERA)

| 0-out empty-base stint start (run / 9 innings) | Before 7/16 | From 7/16 |
|---|---:|---:|
| KT starter | 4.45 | 3.08 |
| KT bridge before first Park | 4.28 | 4.35 |
| Park Young-hyun | 1.09 | 1.29 |

- Full-season bridge 5.84 vs SP 3.85 vs Park 3.79 R/9; selection/inherited-runner contamination is substantial. Clean bridge 4.31 vs clean Park 1.16, but Park clean sample 5 runs in 38⅔ IP.
- Comparable league 6–8 nonstarter nonprimary-closer: KT 5.94 vs peers 5.91, **not unusually weak by league measure**.
- Pregame elite opposing starter did **not** amplify bridge R/9 (elite 4.74 vs other qualified 6.89); no justified strength-interaction penalty.
- Season split widened within-team starter-vs-bridge gap because starter improved, not because bridge skill demonstrably degraded.

## Research test A: Strictly prior-date team+bridge quality to forecast 6–8 inning team runs

- Base dataset: 720 regular-season games, 1440 team-game observations. Feature construction requires at least 160 prior team-game samples.
- Train: May 1–July 15 **287 games, 574 team-game scores**.
- Test: July 16–October 12 **296 games, 592 team-game scores**.
- Feature cutoff: only games on dates strictly **before** target. Prior 35-day inning6–8 relief appearances rank candidates, up to four arms. Player K-BB and HR rates shrunk toward predate league mean. Workload proxy: pitch/event-row count, not verified official pitch counts.
- Poisson ridge, one fixed penalty 30; no target-season test result in fitting. Evaluated NLL for count score and Brier(≥2 runs); 6000 resamples at game—not individual score—level.

| Model | OOS Poisson NLL ↓ | OOS Brier(≥2) ↓ |
|---|---:|---:|
| Pregame league mean | **2.16691** | **0.25976** |
| Prior team offense/defense | 2.17101 | 0.26807 |
| + role-proxy candidate quality | 2.18205 | 0.27008 |
| + prior 2d workload | 2.18376 | 0.27043 |

- Candidate-quality vs team baseline NLL delta **+0.01105**, 95% game-bootstrap CI **[+0.00398,+0.01855]**, worse.
- With workload delta **+0.01275**, CI **[+0.00533,+0.02043]**, worse.
- KT only test n=61: actual average 6–8 R **1.3934**, baseline predicted **1.9279**, +quality predicted **1.9434**, baseline NLL **2.04963** vs candidate **2.06213**.
- **Decision:** Reject adding numeric score/side probability adjustment from these proxies. This cannot disprove the general effect of actual reliever quality; it rejects only this particular simplified proxy/model. Past-performance model-rule selection is exploratory; test is chronological but not a preregistered independent forward trial.

Reproducible workflow: [41790793197](https://github.com/darkhorce1227-cell/kbo-cloud-pbp/actions/runs/41790793197).

## Research test B: Live-at-first-bridge observed score state → Park usage

- **This is an in-game signal, unavailable in CLEAN PRE.** Eligible KT game-bridge episodes n=142.
- Observed Park use conditional on score **when bridge began**:
  - leading by 1–3: **31/46 = 67.39%**
  - not leading: **18/63 = 28.57%**
  - leading by ≥4: **7/33 = 21.21%**
- Pre-Jul16 train n=82, Jul16+ holdout n=60; 8-pseudocount shrink to overall prior.
- Base vs state-conditioned Brier: **0.23472 → 0.20710**; delta −0.02762, 95% CI **[−0.06007,+0.00627]**.
- Base vs conditioned log-loss: **0.66237 → 0.60623**; delta −0.05614, CI **[−0.12275,+0.01452]**.
- **Decision:** keep conditional Park-use as a *live shadow/watch*, not a production probability, because CI crosses zero and other drivers (inning, Park availability, roster) are unmodeled.

Reproducible workflow: [41791551275](https://github.com/darkhorce1227-cell/kbo-cloud-pbp/actions/runs/41791551275).

## Operational integration: audit-only, no price/data leakage
- `scripts/kbo_bullpen_pregame_audit.py`: from immutable source, target date and team, lists candidate first-followers/bridge arms using strictly earlier date history, prior 1/2/3d raw-pitch-row workload, raw K-BB and HR denominators; explicitly **not actual roster / confirmed available / calibrated pitcher true-talent**.
- Verified sample KS1 11/03: 7 KT role-proxy candidates, 15 Samsung; actual KS roster exclusion (e.g., Kim Jae-yun) can make role-proxy candidates invalid, so official active roster is a mandatory manual crosscheck.
- Live 1–3 run lead is observable at actual bridge start and can be separately investigated; it cannot enter pregame predictions as a realized value.
- Never add a "strong opposing starter makes KT bullpen collapse" factor, a generic KT weak bullpen penalty, or a 40+ pitch-row fatigue coefficient.
- Keep Champion unchanged. Revisit only when active roster + official workloads + independently frozen future-game tests improve F5/Full probability loss on paired games.
- Any first-follower better identity prediction must also pass H4-B probability-layer tests; H4-A (usage) alone insufficient.

Integrity smoke workflow: [KBO bullpen PRE inputs integrity smoke](https://github.com/darkhorce1227-cell/kbo-cloud-pbp/actions/runs/41791937550), **success**.
