# KBO market-implied fair odds calibration / betting backtest

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
