# KS3 2026-11-06 | MARKET-UNSEEN RESEARCH PRE FREEZE
**Reference timestamp:** 2026-11-06 15:34 KST (GitHub Actions run 42175215127, successful). Do NOT represent as official Operational Champion CLEAN PRE; official original manual Champion and confirmed-lineup numeric correction are NOT source-reproducible.

## Verified game context
- `77771106KTSS02026`, 2026-11-06 18:30 KST, Daegu, KT away vs Samsung home. 7-game KS tied 1–1.
- Nov3 KS1 KT beat Samsung 8–7. Nov4 KS2 Samsung beat KT 3–2 after 12 innings; official PBP score audit Nov4 PASS (F5 1–1, Full 3–2).
- Starting pitchers as officially reported: KT So Hyeong-jun ID **50030**; Samsung Ariel Hurado ID **53375**. Confirmed batting 9-man card not yet captured at freeze, so `LINEUP_NOT_CONFIRMED`.
- PBP through 2026-11-04: 224,975 rows, SHA256 `0ca95818ac8ea00030f79d8f5aa9c98987acb4edd334a8228ef9fb3503e12d4e`. Nov6 PBP is not required before game. Source status checked Nov6 15:23 target Nov6 `TOO_EARLY` (postgame updater starts 17:00).
- Research source 2026 only due historical preprocessing setup. Coefficients from frozen v1.2 and v1.3 files. v1.3 late-team allocation is the exploratory, physically coherent F5+nonnegative-late method; it is **not** original v1.3 and not promoted. Never claim original Champion comparison.
- Data, model and simulation were frozen BEFORE intentionally inspecting any 2026-11-06 betting prices. `market_prices_inspected=false`; no Proto / Betman / foreign odds in model calculation.

## Starter scouting and Expected IP
- So 2026 RS 23 starts, 131⅓ IP = **5.7101 IP/start**, ERA ~3.15, vs Samsung 3G 2W ERA 3.71. Last real game Oct12 vs Samsung 6IP 1R; U23 practice Oct30 5IP 2R 81 pitches. Whole available 2026 PBP starter PA K% **19.41**, BB% **4.40**, HR/PA **1.83**.
- Hurado 2026 RS 24 starts, 150 IP = **6.25 IP/start**, ERA ~3.54, vs KT 3G ERA 5.17. Last real game Nov1 PO5 KIA 7IP 1 unearned, 4 days' rest. Whole available PBP starter PA K% **15.01**, BB% **5.05**, HR/PA **2.31**.
- Frozen historical PS H4-A simple -0.7598198773 IP/RS start shadow: So **4.9503 IP**, Hurado **5.4902 IP**. This is Challenger usage support, **not** Champion EIP and not a direct run μ factor. Note that recent Hurado seven-inning role and So month-long official absence accompanied by practice start may change actual leash; no validated dynamic numerical override. `P(<2 IP)`, `P(4ER first2)`, `P(<5 IP)` numeric all `UNAVAILABLE`.
- Small direct PvB: Hurado vs Hilliard 9 AB / 4H / 3 HR; So vs Diaz 15 AB / 6H / 2 HR. Extreme samples: watch matchup, do NOT create a bonus coefficient.
- Daegu park/contact/HR tail = WATCH, no confirmed run-adjustment coefficient.

## Price-blind research distribution — immutable model outputs
[Successful premarket Github Actions run 42175215127](https://github.com/darkhorce1227-cell/kbo-cloud-pbp/actions/runs/42175215127), artifact `KS3-20261106-market-unseen-research-pre` includes full 10-family distributions for two research models. Code `scripts/research_ks3_20261106_premarket.py`.

| Measure | v1.2 research | v1.3 phase-coherent experimental |
|---|---:|---:|
| F5 KT μ | 2.800181 | 3.282541 |
| F5 SS μ | 2.918652 | 2.351089 |
| Full KT μ | 4.752428 | 5.415049 |
| Full SS μ | 4.878212 | 4.223293 |
| KT F5 win | 41.9963% | 52.0606% |
| F5 draw | 13.7625% | 13.8122% |
| SS F5 win | 44.2412% | 34.1272% |
| KT Full win to 15 innings | 48.9097% | 60.0163% |
| SS Full win to 15 innings | 50.9753% | 39.8763% |
| 15 inning unresolved draw | 0.1150% | 0.1075% |
| F5 O4.5 | 56.1716% | 55.0653% |
| F5 O5.5 | 45.0944% | 43.9263% |
| Full O9.5 | 46.9081% | 47.0394% |
| Full O10.5 | 39.5291% | 39.6266% |

**Interpretation:** Full total near **9.63 runs** is strikingly similar while who scores it differs by model, producing ~11 percentage-point divergence in KT win chance. Do **not average** two unvalidated models into an invented Champion probability.

Simulation assumptions/limitations: F5 and 6th+ scoring NB blocks independently sampled, F5 alpha=.54 and late alpha=.322; 10–15 inning extra independent Poisson by team approx, ignores walkoff and actual postseason rules for some exotic settlement. Not calibrated for PS, first follower, exact lineup or current bullpen. No official EV or staking from these probabilities.

### Market-family audit
- Regular 7: Full W/L, Full handicap, Full 승1패, Full O/U, F5 W/D/L, F5 handicap, F5 O/U = **7 frozen research surfaces**.
- PS additional 4: winning margin, F5/Full double, exact combined score = **3 frozen research surfaces**; HR O/X×6 score bands = **1 unavailable** (no validated joint HR-run model).
- `PS_PRE_SURFACE_AUDIT games=1 | expected=11 | research_complete=10 | unavailable=HR×score | official_champion_complete=0 | odd_even=EXCLUDED`.
- No actual price checked; odds, market no-vig, Edge/EV, Raw/Max official & Practical are **NOT CAPTURED/UNAVAILABLE** pending official pipeline. `Actual=NOT_CAPTURED`. Do not mark research PRE as official LOCK.

## G1/G2 bullpen workload, immutable 11/04 PBP
[Validated research bullpen PBP audit run 42175416419](https://github.com/darkhorce1227-cell/kbo-cloud-pbp/actions/runs/42175416419), code on main `scripts/kbo_bullpen_pregame_audit.py`, raw pitch *rows* not necessarily official pitch count:
- KT G1→G2: Park Young-hyun **22+23** rows; Kim Jeong-un **13+23**; Oh Won-seok **18+8**; Son Dong-hyun **8+12**; Sugimoto G2 **6**; Bae Je-seong none in G1/G2 (possible fresh long follower).
- Samsung G1→G2: Bae Chan-seung **22+19**; Kim Tae-hoon **13+13**; Jang Chan-hui **32+0**; Lee Seung-min **0+21**; Won Tae-in **0+32**; Yang Chang-seop **41+0**.
- 11/5 rest/travel day. Park has demonstrated willingness to pitch consecutive days; do not install automatic tiredness penalty. Won Tae-in's prospective start in KS4/5 makes his KS3 high-leverage relief use questionable: **verify manager plan**, do not use as default bridge.
- On G1/G2 KT SPs pitched 6 & 7 IP; bridge problems were visible but 2026 whole-season KT 6–8 nonstarter noncloser RA9 **5.94** vs other 9 clubs **5.91**. Do not add a KT-only static bridge penalty or penalty conditional on opposing strong starter.
- As-of roster must be checked: Samsung Kim Jae-yun, lefty injury omissions; KT Kim Sang-su heel injury observed as of Nov4, return not confirmed; confirmed KS3 9-man lineup not captured.

## Decision at premarket
- Baseball: So slightly better K/BB/HR and opponent recent results; Hurado longer innings and recent 2 PO quality starts; both with different rest risks. Strict side edge **INDETERMINATE** between v1.2 and coherent v1.3; KT possible advantage but not sufficient to assign Champion p.
- Price-blind research watchlist: F5 team-run total, Full 9.5/10.5, bullpen path, KT full / 삼성 full; no bet recommendation until operator odds and 9-man lineup, and no official +EV without original Champion.
- No pricing or fitted late/injury/weather coefficients. `PRE_MARKET_RESEARCH_FROZEN` not official Champion LOCK.
