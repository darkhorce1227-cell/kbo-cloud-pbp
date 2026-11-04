# Cross-year lineup feature reproducibility (2026-11-04)

- Frozen original 2024 RS PBP artifact `kbo_pbp_2024_training.csv.gz` SHA256 `d9e4b59fc4c255990452c47d384977f689f338ce6b90988933906cb9e7874458`; 227,955 raw pitch rows / 736 total season games including postseason.
- 2025 counterpart SHA256 `017ae5f39e79b2aeaa05e269cf32aebe7cc410c75fedd6e928850a61730acb61`; 222,650 rows / 736 games including postseason.
- Source asset [GitHub Actions 37073907739](https://github.com/darkhorce1227-cell/kbo-cloud-pbp/actions/runs/37073907739), artifact `kbo-mu-training-pbp-2024-2025`, validated manifests [41809183736](https://github.com/darkhorce1227-cell/kbo-cloud-pbp/actions/runs/41809183736). Date filtering selects 2024 and 2025 **up to Sep30 only**, excluding PS games; 1432 games in source after filtering.
- Same features as frozen 2026 standalone lineup pilot, same 120PA pseudo-count, alpha=.10, past-date-only sequential batter/teams, retrospectively reconstructed first nine. **Do not call these independently timestamped announced lineups.**
- Train 2024 May1–Sep30: 561 games/1122 team-games. Evaluate 2025 May1–Sep30: 560 games/1120 team-games.
- Full game scoring model baseline NLL **2.8951081**, +lineup **2.8885938**; delta **−0.0065143**, game bootstrap 95% CI **[−0.0202595,+0.0072067]**. F5 baseline **2.4133512**, +lineup **2.4114641**, delta **−0.0018872**, CI **[−0.0109099,+0.0068105]**.
- F5 win/draw/lose logloss **1.0337194→1.0274063**; Full WDL **0.8869792→0.8876378**.
- Baseline full-run model's predicted mean 2025 **5.6626** vs actual **4.7339**, indicating broader cross-year environment calibration weakness, common to both candidates.
- Contrast independent 2026 first study: 2026 May–Jul train, Jul–Oct test 296 games; +lineup worsened F5 count NLL by **+0.0245764** [+.0011612,+.0484946] and Full by **+0.0482825** [+.0156048,+.0809123].
- **DECISION:** No stable validated gain for this particular lineup feature bundle. Accept read-only audit only. Original Champion and recovered v1.2 v0.9 unchanged. No market/price retrospective adjustment or bankroll effect.
- [2024train→2025test Actions 41809425461](https://github.com/darkhorce1227-cell/kbo-cloud-pbp/actions/runs/41809425461), machine JSON `lineup_shadow_2024train_2025test.json` in run artifacts.

`CHAMPION=UNCHANGED; NEW_LINEUP_NUMERIC_COEFFICIENT=REJECTED; LINEUP_SCOUTING_AUDIT=ACTIVE`.
