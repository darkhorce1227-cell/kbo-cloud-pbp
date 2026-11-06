# KS3 2026-11-06 PRICE OPEN — user-provided BETMAN MHT audit

**Market snapshot**: Uploaded BETMAN `프로토 승부식 132회차`, `프로토 기록식 121회차`, `야구토토 매치 17회차`, as received around 2026-11-06 15:50 KST. Match 17 vote timestamp 15:42 KST. These uploaded bytes are the price source; do not replace them with earlier/different internet price boards.

**Frozen before these quotes**: GitHub Actions run [42175215127](https://github.com/darkhorce1227-cell/kbo-cloud-pbp/actions/runs/42175215127), artifact `KS3-20261106-market-unseen-research-pre`, 2026 PBP SHA `0ca95818ac8ea00030f79d8f5aa9c98987acb4edd334a8228ef9fb3503e12d4e`. Models are v1.2 research and phase-coherent v1.3 exploratory. Neither is original Operational Champion.

## All seven Proto fixed odds families (home Samsung / away KT)
| ID | market | options & odds |
|---|---|---|
|2839|Full W/L|Samsung 1.68 / KT 1.80|
|2840|Full win1lose|Samsung by2+ 2.40 / one-run or tie 3.30 / KT by2+ 2.33|
|2841|Full handicap Samsung −2.5|Samsung −2.5 3.16 / KT +2.5 1.20|
|2842|Full U/O9.5|U 1.66 / O 1.83|
|2844|F5 W/D/L|Samsung 2.06 / tie 4.75 / KT 2.20|
|2845|F5 handicap Samsung −1.5|Samsung −1.5 2.55 / KT +1.5 1.32|
|2846|F5 U/O4.5|U 1.95 / O 1.57|

Each complete 7-family inverse-price overround ~14.9–15.1%. No-vig Full Samsung 51.72% / KT 48.28%; F5 Samsung 42.19% / draw 18.30% / KT 39.51%. **Break-even** for actual betting is 1/quoted odds, NOT no-vig probability.

## Record式 121 (40 additional fixed odds, full list)
- 5432 margin Samsung 1/2/3/4/5/6/7+ at 4.90/7.50/10.00/12.50/16.00/21.00/8.70; KT 1/2/3/4/5/6/7+ at 7.00/7.90/9.50/12.00/15.50/21.00/8.70.
- 5433 F5/Full (Samsung perspective): 승/승 2.40; 무/승 8.50; 패/승 7.00; 승/패 8.10; 무/패 11.00; 패/패 **2.80**.
- 5434 exact full total scores 1–19 at [75.00,55.00,17.50,17.00,9.20,10.00,7.20,8.70,7.20,9.30,8.60,11.50,11.50,16.50,17.50,25.00,29.00,40.00,50.00], 20+ at 20.00.
- Displayed `Σ 1/odds - 1`: margin 42.76%, F5/Full 24.87%, exact total 43.38%. Record futures missing 15-inning tie/zero-total ticket definitions; don't pretend these are full risk-free complete books.

## All 56 fixed selections audited by exact prior-frozen model distribution
**No selection has positive EV under both research models.** Most important paired EV results:
| Option | Price | v1.2 P / EV | coherent v1.3 P / EV |
|---|---:|---|---|
| record 패/패 = KT F5 win AND KT Full win |2.80|35.20% / −1.44%|45.44% / +27.23%|
| KT F5 win |2.20|42.00% / −7.61%|52.06% / +14.53%|
| KT Full win|1.80|48.91% / −11.96%|60.02% / +8.03%|
| KT by2+|2.33|35.84% / −16.49%|46.30% / +7.89%|
| KT F5 +1.5|1.32|67.79% / −10.51%|77.05% / +1.71%|
| Full U9.5|1.66|53.09% / −11.87%|52.96% / −12.09%|
| F5 O4.5|1.57|56.17% / −11.81%|55.07% / −13.55%|

Original full 56-cell model comparison preserved in the associated conversation audit export: `kbo_KS3_20261106_price_audit_56.json`.

## Match 17 is pari-mutuel, NOT a fixed 144-ticket price surface
- As-of 15:42 score-band vote table 6×6 sums to **358,271** votes. Highest share Samsung 5–6 runs / KT 3–4 runs, 37,672 votes (10.51%); not a model probability.
- Total sale KRW 35,827,100; first-place estimated pool KRW 17,913,550; not a promised payout to a selected ticket.
- HR O/X × team-score 144 joint probability model uncalibrated and actual per-ticket pool payout unknown. Therefore `POOL_TOP10_MODEL_P=UNAVAILABLE`, `POOL_FAIR_ODDS=UNAVAILABLE`, `POOL_EV=UNAVAILABLE`, `KELLY=UNAVAILABLE`; must not multiply unrelated marginals or infer 144 payouts from 6×6 score-band aggregate vote counts.

## Decision and ledgers
- Research-only watch: KT F5 win×Full win (record 패/패 @2.80) *near* v1.2 break-even, +27.23% exploratory v1.3; **NOT ROBUST** across models or phase correlation.
- `OFFICIAL_CHAMPION=UNAVAILABLE`; `OFFICIAL_PRACTICAL=HOLD_0U`, `OFFICIAL_RAW_TOP5=NOT_CAPTURED`, `OFFICIAL_MAX_EV_TOP5=NOT_CAPTURED`, `ACTUAL=NOT_CAPTURED`. The user did not confirm a purchase.
- `PS_PRICE_BOARD_AUDIT: games=1 | families_expected=11 | fixed-price_families=10 | fixed_outcomes=56 | pool_family=1_unpriced | odds_odd-even=EXCLUDED`.
- This record is **postprice** and DOES NOT rewrite `research/2026_ks3/pre_20261106_market_blind_frozen.md`.
