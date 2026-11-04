# KS2 2026-11-04 | FINAL v2 18:04 KST — market refresh (no PRE rewrite)

**Game**: Samsung @ KT, 18:30 KST, KS G2; user-supplied 9-player confirmed batting orders retained.
**Source:** primary indexed Wisetoto 131 market page https://www.wisetoto.com/index.htm?font_type=small&site_domain=default (web inspected at approximately 18:00–18:04 KST), and Yonhap 16:25 manager presser https://www.yna.co.kr/view/AKR20261104143900007 . Indexed result variants disagree for some historical prices; the fresh price below is the visible most recently indexed quote, not an exchange time-stamped trading feed. No user actual order asserted.

**Frozen source model**: 17:16 pre-price GitHub [run 41798174203](https://github.com/darkhorce1227-cell/kbo-cloud-pbp/actions/runs/41798174203), no price data used in fit. v1.2/v1.3 strictly research-only; official Operational Champion unavailable, so this is NOT an official probability LOCK or official Raw/Max TOP5.

## New 18:04 price changes versus former 17:42 reported snapshot
- F5 (home/ draw /away) **KT 2.00 vs 1.96; draw 5.80 vs 5.40; Samsung 2.10 vs 2.20**.
- F5 O/U 4.5: **Under 1.75 vs 1.70; Over 1.73 vs 1.78**.
- F5 KT -1.5 **2.72 unchanged**.
- Full KT win **1.64 unchanged**, Samsung **1.85 unchanged**.
- Full O8.5 **1.68** unchanged versus most recent primary snapshot, Under **1.80**.
- Full handicap -2.5 KT **2.97** / Samsung **1.23**.
- Full win1loss KT by 2+ **2.38** / 1-run **3.20** / Samsung by2+ **2.40**, but an older indexed variant has 2.45/3.20/2.33; do not overstate single-page price precision.

### Highest-concordance research selection table
| Ticket | Updated price | Break-even P | v1.2 P | v1.2 EV | v1.3 P | v1.3 EV |
|---|---:|---:|---:|---:|---:|---:|
| KT F5 -1.5 | 2.72 | 36.765% | 41.035% | +11.614% | 57.045% | +55.163% |
| KT F5 Win | 2.00 | 50.000% | 52.909% | +5.818% | 69.187% | +38.374% |
| F5 Over4.5 | 1.73 | 57.803% | 60.326% | +4.363% | 58.117% | +0.543% |
| Full Over8.5 | 1.68 | 59.524% | 60.422% | +1.509% | 59.014% | −0.856% |

**Key changes:**
1. Previous qualitative F5 O4.5 favorite is **WITHDRAWN** as price dropped to 1.73, leaving just **0.31 percentage points** v1.3 above break-even (so fragile relative to unquantified confirmed-lineup effect). Not an official Practical bet ever.
2. The more consistent research-side raw EV is KT F5 -1.5, but under lower model v1.2 the probability is just +4.27pp above break-even and Samsung 7-L-handed lineup vs Ko's empirical L splits is an unresolved threat.
3. Stress only, **NOT calibrated posterior**: artificially shift v1.2 Samsung F5 run mean +0.6 while holding KT fixed, NB alpha=.54 -> approximate KT -1.5 probability ~36.85%, reducing EV to ~+0.2%. This is **not** a predicted +0.6 effect; purely break-even sensitivity.
4. F5 market update toward Samsung is consistent with possible matchup risk but is **not proof** of why traders shifted price.
5. No exceptional KT bullpen league penalty or strong opposing starter interaction; Park Young-hyun considered plausibly available on back-to-back after 5-out KS1 save, not guaranteed or gifted better talent.
6. KT Sang-su excluded heel injury; Ryu Hyeon-in starts. KT Logan now preparing KS4 start, not default KS2 first reliever. Coach identified Bae Je-seong/Oh Won-seok as early/long-relief options; Paddack 4d rest after PO pitch loads.

## Live suitability and integrity
The first-stage model cannot reproduce exact lineup-adjusted 11-family Official Champion probability. The original 10 research-family PRE vector remains frozen; HR×score joint ticket distribution unavailable; price coverage is seven fixed-odds families (16 conventional non-odd/even outcomes), other records prices missing.
- Baseball direction: KT **slight full-game lean**, Full @1.64 not robust +EV.
- `Research Raw pick (not recommendation): KT F5 -1.5 @2.72`. Avoid treating full win and F5 spread exposures as independent.
- `Official Practical: HOLD 0u`; Official Raw/Max: `NOT_CAPTURABLE_FROM_RESEARCH_MODELS`; Actual: `NOT CAPTURED`; unreconciled official Champion: `UNAVAILABLE`.

**FINAL_STAGE_STATUS: additional price-only refresh / prior existing research FINAL replaced by new price version only; no official Champion FINAL 4/4 claim.**
`PRICE_BOARD_AUDIT games=1 | PS families expected=11 | fixed quote families=7 | selection outcomes=16 | HR_SCORE=UNAVAILABLE | 3 record-style families missing prices | odd-even excluded`.
