# KBO reproducible μ-v1.2 research candidate

Status: `RESEARCH_CANDIDATE_NOT_CHAMPION`

## Goal

Replace the unrecovered/manual μ layer with a reproducible pregame mean model while keeping the existing team-score Negative Binomial distributions separate:

- F5 NB2 alpha = 0.54
- Full NB alpha = 0.322

No market prices are used.

## Data discipline

- Fixed 2024 PBP is used to create the initial training period. Training rows begin 2024-05-01 so March/April can initialize rolling history without future leakage.
- 2025 is the validation year used for regularization and temperature choices.
- 2026 through 2026-10-01 is untouched OOS evaluation.
- Same-date games receive features from information available before that date; same-day results do not leak into another game on that date.

Fixed training PBP is produced by the GitHub Actions workflow `KBO Mu Training PBP Export`.

## Architecture

The research showed that the best total-run calibration and the best team-side discrimination came from different submodels.

### F5

1. `simple_f5`: league F5 30D environment is a log offset; long/30D offense, park factor and home flag estimate the game scoring level.
2. `core_f5`: adds opposing starter long/recent K-BB, HR, H rates, expected-IP proxies and opposing bullpen long/30D RA9.
3. The game total F5 μ is preserved from the simple model.
4. The two team shares come from the core model and are expanded by a validation-selected temperature `gamma=5`.

### Full

Late scoring is modeled separately (`runs_late = full - F5`).

1. `simple_late`: late offense + park/home with league late-30D offset.
2. `core_late`: adds bullpen quality and starter-IP proxies, but validation shrinks this component heavily.
3. Full game total μ = simple F5 + simple late.
4. Team shares use core F5 + core late and validation-selected temperature `gamma=6`.

This gives the useful empirical separation found in research:

- simple model: better total/mean calibration;
- core model: better relative team ordering;
- hybrid: preserves game total while using matchup information for team allocation.

## Selected validation hyperparameters

- simple_f5 ridge alpha = 1
- core_f5 ridge alpha = 5
- simple_late ridge alpha = 1
- core_late ridge alpha = 300
- F5 share temperature gamma = 5
- Full share temperature gamma = 6

All were selected from 2025 validation before evaluating 2026 OOS.

## 2026 OOS result, 684 games / 1,368 team-games

### F5

| model | MAE | NB NLL | WDL Brier | WDL log loss |
|---|---:|---:|---:|---:|
| league baseline | 1.9563 | 2.12446 | 0.20325 | 0.99930 |
| simple | 1.9451 | 2.12205 | 0.20249 | 0.99619 |
| core | 1.9494 | 2.12293 | 0.20236 | 0.99607 |
| **hybrid** | **1.9310** | **2.11949** | **0.20187** | **0.99374** |

### Full

| model | MAE | NB NLL | WDL Brier | WDL log loss |
|---|---:|---:|---:|---:|
| league baseline | 2.7250 | 2.55792 | 0.17640 | 0.82519 |
| simple | 2.7130 | 2.55385 | 0.17602 | 0.82425 |
| core | 2.7214 | 2.55687 | 0.17537 | 0.82186 |
| **hybrid** | **2.7051** | **2.55299** | **0.17276** | **0.81352** |

The hybrid also selected the winning side in 2026 decided games at approximately 56.0% for F5 and 57.2% for Full. These figures are descriptive OOS diagnostics, not betting ROI or a market comparison.

## Important limitations

The candidate does not yet contain a historically reproducible representation for:

- confirmed lineup effects;
- weather;
- detailed defense;
- three-year PvB;
- Contact Quality / EV-LA;
- pitch-fit;
- same-day bullpen availability and manager news.

Therefore it should currently be treated as a reproducible mechanical baseline, with those modules remaining Challenger/manual audit layers. Do not silently promote it to the production Champion until integration and live snapshot comparison are approved.

See `model_spec.json` for exact fitted scaler/coefficients and `metrics.json` for full validation/OOS diagnostics.
