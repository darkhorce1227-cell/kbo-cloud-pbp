# Reproducible baseline v0.1

Build one auditable team baseline and compare each additional information block
on the same games. This directory is research only and does not activate a
Champion, alter a historical PRE, or settle a betting ledger.

## Frozen design

`protocol.json` fixes all model features, shrinkage constants, temporal splits,
numerical distributions, metrics and source checksums before this run's scores
are inspected. Its initial SHA256 is
`8073998bb60401f6949713df13c6299c4267433b61fe845bb547bfcc836220ac`.
The historical seasons were previously inspected in other research. This is a
new retrospective comparison, not a pristine prospective holdout.

- League reference: prior 30-calendar-day scoring offset, shrunk to prior
  season scoring, plus learned home effect.
- Team baseline: add long-term team offense and opponent runs allowed, both
  shrunk toward that season's strictly prior league mean.
- Independent additions: opposing starter K-BB/HR, starter expected outs,
  own offense 7 days, own offense 14 days, opposing relief K-BB/HR,
  relief workload proxy, lineup on-base delta, lineup power delta.
- Every addition is compared to `baseline_team`, not to the previous addition.
  No post-result winning combination is built in this experiment.
- Refit monthly using earlier completed months. Feature history updates only
  after every game on a date has been prepared. No same-date lookahead.
- Two count models: F5 and nonnegative late runs. Full is their convolution,
  so a team cannot have Full expected runs below F5 expected runs.
- Primary losses: F5 W/D/L log loss and Full decided-game log loss.
  Also preserve Brier (sum over classes), count NLL, seven standard research
  market families, mean bias, monthly/yearly results and date-cluster paired CIs.
- Primary comparisons have Bonferroni-adjusted bootstrap intervals as well as
  unadjusted intervals. These condition on fitted forecasts; they are not a
  model-development uncertainty correction.

## Reproduce

Numerical versions: Python 3.12, numpy 2.3.5, pandas 2.2.3, scipy 1.17.0,
scikit-learn 1.8.0. No `kbo_pbp` network parser is required for this runner.

```bash
python -m unittest discover -s tests -p test_repro_baseline.py -v
python scripts/kbo_repro_baseline.py --inputs /path/kbo_pbp_2024_training.csv.gz /path/kbo_pbp_2025_training.csv.gz /path/kbo_pbp_2026_latest.csv.gz --out /tmp/baseline-results
```

One year can be supplied for a within-year diagnostic. Check `manifest.json`
before comparing runs: a one-year and a multiyear run have different prior
history, training data and eligible forecasts. Their metrics are not the same
experiment. Checksums reject changed Release data rather than silently reusing
an old source identity.

The workflow retrieves original 2024/25 files from existing Actions run
37073907739, artifact `kbo-mu-training-pbp-2024-2025`, and verifies their registered
hashes. It leaves those original archives untouched. 2026 is pinned to the
2026-11-06 validated CSV SHA even though the regular-season analysis ends Oct12.

## Saved audit trail

- `features.csv.gz`: one row per game, team and phase, with data cutoff and
  fallback/sample information. Features are stored in the workflow artifact.
- `predictions.csv`: every model/game prediction, target, loss and train cutoff.
- `fitted_models.json`: monthly features, coefficients, intercept and scaling.
- `manifest.json`: source hashes, excluded games, temporal coverage and versions.
- `metrics.json`: aggregate, year, month and paired differences with denominators.
- `REPORT.md`: readable comparison and interpretation constraints.

## Limits that matter

Actual starter identity and first nine distinct batter IDs are reconstructed
from postgame PBP. The player histories are prior-date, but these identities do
not establish pre-announcement availability. Lineup deltas are retrospective
proxies and cannot be treated as prospective pregame evidence. Relief workload
is prior PA count, not official pitch count or a confirmed active roster.
Starter outs are derived from first-reliever entry or terminal out states;
they are not independently verified official IP labels.

The late target includes any played extra innings. The model does not enforce
skipped home ninth innings, walkoffs or an exact overtime rule. Full winner
loss is conditional on a decided game, separately from F5 three-way loss.
Fixed F5 4.5 / Full 8.5 totals and home -1.5 lines are research questions,
not observed historical prices. No actual or hypothetical bankroll is changed.

2024/25 stop Sep30 and 2026 stops Oct12; do not describe this as every historical
regular-season game. New-player fallbacks are explicit prior shrinkage, not a
reason to drop only difficult games from one candidate. All evaluated models
must contain exactly the same game IDs.

Next prospective data must be captured before outcomes and with genuine
announcement timestamps. No result here replaces the original Operational
Champion or changes the recovered v0.9 BP30 promotion contract.
