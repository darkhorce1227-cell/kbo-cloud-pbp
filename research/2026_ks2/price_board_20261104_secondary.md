# KS2 2026-11-04 postmarket price audit (SECONDARY, NOT EXECUTION)
Reference CLEAN PRE frozen before price: [Action 41798174203](https://github.com/darkhorce1227-cell/kbo-cloud-pbp/actions/runs/41798174203), freeze commit `7d45f20a8edaef2fb6c2fc0d4909757ebfda4541`.

**Official operational Champion unavailable, neither v1.2/v1.3 is official.**
Wisetoto was checked first but is unreadable in the accessible web view (no lines). Secondary source: Zentoto 2026 PROTO round 131 indexed results. Quotes may change, not authoritative Wisetoto snapshot. The price audit consumed the immutable PRE artifact rather than recalculating postprice μ. [Actions run 41798751306](https://github.com/darkhorce1227-cell/kbo-cloud-pbp/actions/runs/41798751306).

## Seven priced families, all 16 outcomes

| Family | Outcome | Decimal odds | v1.2 P | v1.2 EV | v1.3 P | v1.3 EV |
|---|---|---:|---:|---:|---:|---:|
| Full W/L | KT Win | 1.64 | 59.9141% | -1.741% | 48.9511% | -19.720% |
| Full W/L | Samsung Win | 1.85 | 39.9757% | -26.045% | 50.9879% | -5.672% |
| Full Win1Lose | KT by 2+ | 2.38 | 46.7544% | +11.275% | 38.8430% | -7.554% |
| Full Win1Lose | 1 run or tie | 3.20 | 25.0880% | -19.718% | 24.3381% | -22.118% |
| Full Win1Lose | Samsung by 2+ | 2.40 | 28.1576% | -32.422% | 36.8189% | -11.635% |
| Full H -2.5 | KT -2.5 | 2.97 | 36.9111% | +9.626% | 30.9400% | -8.108% |
| Full H -2.5 | Samsung +2.5 | 1.23 | 63.0889% | -22.401% | 69.0600% | -15.056% |
| Full U/O 8.5 | Under | 1.77 | 39.5779% | -29.947% | 40.9856% | -27.455% |
| Full U/O 8.5 | Over | 1.71 | 60.4221% | +3.322% | 59.0144% | +0.915% |
| F5 W/D/L | KT Win | 1.96 | 52.9089% | +3.701% | 69.1871% | +35.607% |
| F5 W/D/L | Draw | 5.40 | 12.7020% | -31.409% | 11.7083% | -36.775% |
| F5 W/D/L | Samsung Win | 2.20 | 34.3891% | -24.344% | 19.1046% | -57.970% |
| F5 H-1.5 | KT -1.5 | 2.72 | 41.0347% | +11.614% | 57.0454% | +55.163% |
| F5 H-1.5 | Samsung +1.5 | 1.28 | 58.9653% | -24.524% | 42.9546% | -45.018% |
| F5 U/O4.5 | Under | 1.70 | 39.6744% | -32.554% | 41.8829% | -28.799% |
| F5 U/O4.5 | Over | 1.78 | 60.3256% | +7.380% | 58.1171% | +3.448% |

Full 15-inning draw probability unquoted 0.1% approx; price settlement/void rule not audited.
Overround across the seven families around **14.89–15.03%**.
Four outcomes both research models see positive EV: Full O8.5, KT F5 Win, KT F5 -1.5, F5 O4.5.
Full side and KT by 2+ carry **strong cross-model disagreement**; avoid treating them as robust best bets.
KT F5 -1.5 @2.72 research-only best simultaneous EV but 41.0% lower-model probability is only +4.27 pp above BE 36.76%; moderate parameter error erases edge. The KBO daily official Champion forecast and lineups are not authenticated, so **Practical HOLD 0u**, Raw/Max official benchmarks **NOT CAPTURED**, actual purchase **NOT CAPTURED**. Do not assume user placed any bet.
All seven families preserved. Margins, double, exact-total frozen in earlier PRE artifact but current matching posted odds unknown; HR×score joint distribution unavailable.

## Key baseball-path concerns
- Paddack 10/30 PO4 5.333IP 99 pitches, after four days' rest today; Ko last official RS start 10/11.
- KS1 11/03 KT bullpen: Logan 12 PBP pitch rows, Son Dong-hyun 8, Kim Jeong-un 13, Oh Won-seok 18, Park Young-hyun 22 (5 outs), all candidate rest flags are descriptive, not automatic skill penalties.
- KS1 Samsung bullpen: Yang Chang-seop 41, Jang Chan-hui 32, Im Gi-yeong 4, Kim Tae-hun 13, Bae Chan-seung 22, Satoshi 9 rows. Kim Jae-yun was not on KS active roster as previously reported; official current roster still requires verification.
- 2026 KT clean-entry bridge gap relative to starters widened after July16, but the bridge was not league-exceptionally weak. Do not add KT bullpen penalty because opponent starter elite.
- Weather Suwon 18:00 ≈ 13℃, 20:00 ≈12℃, cloudy with 20% precip in a third-party KMA-derived forecast; coefficient unvalidated, no run-mean adjustment.
- Published lineup claim “same top order” is secondary. No independently verified official 9-batter batting orders means lineup-based SOL / 1-hit P unavailable.
- Any late baseball changes create versioned **separate** model snapshot; do not rewrite this original frozen PRE.

`PS_PRE_SURFACE_AUDIT: 1 game / 11 expected / 10 frozen research / 1 HR-score unavailable / official 0`
`PRICE_BOARD_AUDIT: 1 game / 11 expected / 7 secondary-quoted / 16 outcomes / 4 unavailable/missing / odd-even excluded`
`Practical HOLD 0u | Raw/Max official NOT CAPTURED | Actual NOT CAPTURED`
