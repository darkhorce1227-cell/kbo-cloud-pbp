# KBO market-implied fair odds calibration / betting backtest

## Latest: 2026 KBO FULL WIN1LOSS three-outcome prices, 459 games

**Market product:** Proto single-game *승①패* with outcomes
`HOME_2PLUS` (home wins 2+), `WITHIN_1` (absolute final difference 0
or 1), `AWAY_2PLUS` (away wins 2+). The original archive calls the
middle column `무`; it means **NOT** ordinary 9-inning tie. The
official Sports Toto rule confirms the middle **includes exact ties**:
https://www.sportstoto.co.kr/view.php?idx=2402&type=notice

**Source** https://betlive.kr/proto/260050 and related round pages;
40–122 scanned; rounds 40–45 returned 404, 77 pages parsed.
459/459 distinct game records resolved to the SHA-authenticated 2026
Naver PBP final game scores, 2026-04-21 through 2026-10-12. Sources
had 34 other KBO rows not assignable to a unique validated game PK;
these were **quarantined**, never guessed from dates/names.
Round snapshots are **retrospective**, with no proof of original
pre-first-pitch executable price/timestamp. Do not call this a real
market-beating bet or CLEAN PRE out-of-sample betting result.

### Descriptive 459-game no-vig calibration

| Result | Average no-vig probability | Observed 2026 KBO frequency |
| --- | ---: | ---: |
| Home wins by 2+ | 36.48% | 36.17% |
| Difference ≤1 run, including ties | 25.33% | 27.67% |
| Away wins by 2+ | 38.19% | 36.17% |

- Favorite: forecast **44.36%**, observed **44.23%**; deviation
  **-0.13 pp**; game bootstrap 95% interval **[-4.66,+4.34] pp**.
- Three-class Brier **0.64104**; log loss **1.06205**.
- Average offered three-way book overround **15.12%**.
- Flat 1u favorite-only hypothetical return at retrospective displayed
  odds **-13.76%**. This is **NOT** a certified executable betting ROI.

### Prespecified naive market-information benchmark

Compared on identical 459 games:

| Forecast rule | Three-class multiclass Brier |
| --- | ---: |
| Uniform, each outcome = 1/3 | 0.66667 |
| Date-by-date past-only 2026 WIN1LOSS frequencies (Dirichlet pseudo-count α=10 per outcome, update after all games on a date) | 0.66743 |
| No-vig retrospective market probabilities | **0.64104** |

Market Brier improvement versus *strictly past-date only* climatology
= **0.02639 / 3.95% relative**, with prespecified **calendar-week
cluster bootstrap 95% CI [0.01132,0.04072]** over 26 ISO-week
clusters. This is evidence that bookmaker probabilities contain
useful game-specific information *relative to a naive base-rate
model*, not evidence of exploitable return or an advantage over
our independent KBO CLEAN PRE model.

**Do not compare** these three-way Brier scores directly to prior
2026 ordinary two-way moneyline, handicap or total markets (125
independently PBP-verified games ×3 markets). Outcomes and random
baseline scores differ by product. No genuine model-vs-market skill
numbers are available until original timestamped model CLEAN PRE
probabilities and contemporaneous executable prices are joined.

Reproducibility:

```
research/market_fair_odds/betlive_2026_r40_r122_win1loss3_archived.csv
research/market_fair_odds/betlive_2026_r40_r122_win1loss3_source_audit.json
research/market_fair_odds/betlive_2026_r40_r122_win1loss3_pbp_joined.csv
research/market_fair_odds/betlive_2026_r40_r122_win1loss3_calibration.json
research/market_fair_odds/betlive_2026_r40_r122_win1loss3_information_benchmark.json
scripts/collect_2026_betlive_win1loss3.py
scripts/merge_2026_betlive_win1loss3.py
scripts/audit_2026_betlive_win1loss3_vs_pbp.py
scripts/compare_2026_win1loss_market_vs_naive.py
```

---

## Current empirical results — archived 2026 Proto KBO (2026-10-18)

**Measured, NOT a genuine betting-edge or CLEAN PRE test.** Source: 2026
Proto rounds 105–122, game dates starting 2026-09-04. The earlier 2026
rounds 101–104 returned HTTP 404, while sampled rounds 1,25,50,75,90,100
also returned 404. No claims of full-season market coverage are allowed.

- **125 distinct KBO games**, **375 distinct historical markets**:
  125 ordinary 2-way, 125 handicap 2-way, 125 over/under.
- Source final scores independently verified **125/125** against
  SHA-validated Naver KBO pitch-by-pitch release through 2026-10-17.
  Score/source reconciliation PASS.
- Mean posted bookmaker overround ~**13.6%** for all three markets;
  no-vig favorite predicted vs observed hit:
  - Full 2-way: 58.32% vs 64.00%, Brier 0.4568. Game-bootstrap
    95% interval for observed-minus-predicted: **-2.60 to +13.83 pp**.
  - Handicap 2-way: 64.57% vs 60.80%, Brier 0.4720.
    95% gap interval **-12.53 to +4.54 pp**.
  - O/U: 52.38% vs 48.80%, Brier 0.4991.
    95% gap interval **-12.43 to +5.25 pp**.
- The hypothetical fixed 1u-per-market **favorite-only** return at
  the archived displayed prices (NOT verified pregame executable odds):
  **-3.58%** full win/loss, **-17.21%** handicap, **-18.24%** totals.
  This is descriptive, not a genuine historical portfolio ROI.
- Historical **opening** ML2 Brier 0.4603 vs latest archived displayed
  0.4568, but quote timestamps are unverified; no claim of CLV or
  exploitable information.
- All 95% favorite calibration-gap intervals **include zero**;
  no demonstrated systematic mispricing on this limited dataset.
- In particular, no contemporary model probabilities with validated
  independent CLEAN PRE locks are joined to this archive. **Model
  calibration, model-vs-market skill and +EV claims remain unavailable.**

Artifacts:

```
research/market_fair_odds/sharppick_2026_proto_r105_r122_combined_kbo.csv
research/market_fair_odds/sharppick_2026_proto_r105_r122_combined_audit.json
research/market_fair_odds/sharppick_2026_proto_r105_r122_pbp_result_audit.json
research/market_fair_odds/sharppick_2026_proto_r105_r122_calibration.json
```

Executable audit/workflow:

```
scripts/collect_2026_kbo_proto_archive.py
scripts/merge_2026_kbo_proto_archives.py
scripts/audit_2026_proto_market_vs_pbp.py
scripts/analyze_2026_kbo_proto_archive.py
.github/workflows/run-2026-kbo-proto-expanded-calibration.yml
```

All retrospective prices must retain
`HISTORICAL_PAGE_RETROSPECTIVE_PREMATCH_PRICE_TIME_UNVERIFIED`.
Only externally evidenced pregame quotes may be used for real ROI or
a CLEAN PRE model comparison. Synthetic fair odds are **not** executable
payout prices.

---

**Status: ingestion + calibration code present; no historical full-season odds
has been certified as loaded.** Do not report the synthetic unit tests as
historical betting performance.

## Important product distinction

The auditing script targets **Proto fixed-decimal-odds markets** only.
"야구토토 승1패" is a separate **14-game, three-way, fixed payout-ratio
(pool/parimutuel)** game: HOME by 2+ runs / within one run including ties /
AWAY by 2+ runs. A 14-game ticket payout is determined by the winning
pool and tier, not by a fixed per-game decimal quote. It requires historical
ticket shares, official round outcome and payout (and possibly rollovers)
for an actual profit backtest, plus a separate combinatorial calibration
framework. Do not repurpose Proto decimal odds as if those were fixed
야구토토 승1패 prices. Official source:
https://www.sportstoto.co.kr/base_victory.php

## Two distinct research tracks

1. **Market calibration**: KBO Proto/Betman decimal odds, normalized to
   no-vig implied probabilities within the correct bet market, compared with
   real settled outcomes. Score the market, not our KBO model.
2. **CLEAN PRE challenger validation**: market-independent predictions
   timestamped and externally committed before game start, paired with the
   contemporaneously obtainable market price and settled outcomes. Compare
   model vs market on *identical* rows, then evaluate an ex ante
   predeclared EV/position-sizing strategy. No retrospective fitted
   probability may be called CLEAN PRE.

## Source registry (not yet imported)

- Betman / SportsToto official results and rules; baseball includes extra
  innings; baseball-first-half markets stop after 5 innings. Ordinary no-draw
  baseball draws refund (1.0), and abandoned/void games require the book's
  actual posted settlement. Official product-specific line/settlement governs.
- SharpPick /proto/ publicly describes opening-to-latest odds and final
  results by Proto round. **Its general web snippets are NOT a complete
  downloadable event-level KBO 2026 odds file.** Before use, archive a
  complete round, source URL/time/round and outcome semantics, licensing/
  permitted retrieval, and cross-check sample outcomes with official book.
- Existing Project PDF "실시간_경기_분석.pdf" has a screenshot/summary of
  conditional historical KBO odds searches; it is not a complete table of
  game/market odds, dates and settled outcomes.

## Input: normalized long-form CSV (one row per outcome of a market snapshot)

Mandatory fields:

```csv
game_pk,market_id,market_type,line,price_asof_kst,game_start_kst,settlement_status,settled_outcome,outcome,decimal_odds,model_probability,model_lock_kst,model_lock_ref
```

- Supported market_type: `FULL_ML2`, `F5_1X2`, `FULL_OU`, `F5_OU`,
  `FULL_HC2`, `FULL_HC3`. Keep distinct markets/handicap and total
  line values distinct: for example `FULL_OU@9.5` and `FULL_OU@10.5` need
  distinct `market_id`. Do **not** invent ties or push treatment.
- Options: ML2 HOME/AWAY; F5_1X2 HOME/DRAW/AWAY; OU OVER/UNDER;
  HC2 HOME/AWAY; HC3 HOME/DRAW/AWAY.
- `settlement_status`: SETTLED (then `settled_outcome` one of the offered
  choices), PUSH (ordinary 2-way draw or exact line, **refund**), or VOID
  (cancelled/invalid, **refund**). The last two must have blank
  `settled_outcome` and are not scored as binary losses.
- `price_asof_kst`, `game_start_kst`, and `model_lock_kst` need
  offset-aware ISO timestamps: e.g. `2026-09-15T08:00:00+09:00`.
  `price_asof_kst` must precede first pitch. Price snapshots are not
  independently scored games; the script scores the *earliest* snapshot
  supplied per market. An independent closing-odds study must select closing
  snapshots at ingestion without including post-start prices.
- Optional `model_probability` is required for **all** offered outcomes of
  a modeled market or none, summing to one. Model rows require a pregame
  model lock time and `model_lock_ref`. Having a ref string is not proof;
  separately audit original commit time/hash before calling the record
  prospective. A model with historical data missing must stay unavailable.
- For no-draw 2-way baseball markets with refunds on ties, the no-vig
  probabilities are conditional on the market producing a non-push outcome.
  Calibrate on SETTLED rows only and keep PUSH/VOID counts.

## Measures

```
q_i = 1 / displayed_decimal_odds_i
overround = sum(q_i) - 1
p_no_vig_i = q_i / sum(q_i)
synthetic_fair_odds_i = 1 / p_no_vig_i
```

Proportional margin removal is an explicit baseline, not the uniquely correct
probability estimate. Never backtest bets at synthetic fair odds. Model-vs-
market paired multiclass Brier and 0.1-width favorite reliability bins are
computed by distinct market; n, PUSH/VOID exclusions and quote versions are
recorded. Use log loss, game-cluster confidence intervals, rolling calibration,
and real-priced ROI in next stages **only after authentic long-form market
records and original CLEAN PRE model locks are present.**

Run:

```bash
python scripts/backtest_kbo_fair_odds.py --input path/to/verified_market_book.csv --output research/market_fair_odds/market_calibration.json
python -m unittest discover -s tests -p test_kbo_fair_odds_backtest.py -v
```

**Gate:** Not enough for investment/EV conclusion when no verified 2026 odds
or model snapshots are present. Do not replace missing input with inferred
odds, present-day odds, unlabeled close odds, or ex post adjusted μ.
