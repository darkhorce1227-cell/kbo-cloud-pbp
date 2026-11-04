# 2026-11-04 KT vs Samsung KS2 | MARKET-BLIND RESEARCH PRE FREEZE

**Frozen 2026-11-04T08:16Z / 17:16 KST**, before intentional Korean sportsbook price inspection.

- Game: 2026-11-04 KS G2, Samsung (away) Chris Paddack ID 56459 vs KT (home) Ko Young-pyo ID 64001, Suwon 18:30, KT series 1-0.
- Verified 2026 PBP source through 2026-11-03 (224,611 pitch/event rows, sha256 `32c70d31c89c83a9e8606254ff6ec084f599371b6debc6dde36066bf122c74e1`).
- Market odds were **not used**. Model parameters from frozen `research/mu-v1.2/model_spec.json` and `research/mu-v1.3/model_spec.json`; only available 2026 historical features. Both are **RESEARCH_CANDIDATE_NOT_CHAMPION**. The official operational Champion input is unavailable and cannot be replaced by either research model.
- Run: [success 41798174203](https://github.com/darkhorce1227-cell/kbo-cloud-pbp/actions/runs/41798174203). Run artifact `KS2-20261104-market-blind-research-pre` contains full available 10-family distributions, no HR×score joint. The program/code and model parameters were committed before the job run. **This artifact is not an Official LOCK**.

## Frozen summary, n=700,000 draws / model

| Segment | v1.2 KT μ | v1.2 SS μ | v1.3 KT μ | v1.3 SS μ |
|---|---:|---:|---:|---:|
| F5 | 3.61168 | 2.58231 | 4.47491 | 1.64713 |
| Full | 5.64406 | 4.40799 | 5.31012 | 4.74996 |
| 6th+ implied mean | 2.03238 | 1.82568 | 0.83521 | 3.10282 |

| Market | v1.2 | v1.3 |
|---|---|---|
| F5 KT/Draw/SS | 52.9089% / 12.7020% / 34.3891% | 69.1871% / 11.7083% / 19.1046% |
| Full KT/SS/15inn draw | 59.9141% / 39.9757% / 0.1101% | 48.9511% / 50.9879% / 0.0610% |
| Full 2+ KT / 1run-or-tie / 2+ SS | 46.7544% / 25.0880% / 28.1576% | 38.8430% / 24.3381% / 36.8189% |
| Full U9.5 | 49.7153% | 50.6456% |
| F5 U5.5 | 50.4047% | 52.2857% |
| Full KT -1.5 | 46.7544% | 38.8430% |
| F5 KT -1.5 | 41.0347% | 57.0454% |

## Expected IP shadow only
- RS start avg Paddack 6.1111IP; PS shift −.7598198773 => **5.3513 IP**; historical 284-residual approximation P(<5 IP) 36.32%.
- RS start avg Ko 5.9540IP; shift => **5.1942 IP**; empirical P(<5 IP) 40.18%.
- Championship operational EIP unavailable, so H4-A comparison not available today.
- Paddack last PO4 Oct30 5⅓ IP, 99 pitches; fresh analysis must consider 4-day recovery and PS usage. Ko last start Oct11, prolonged official-game gap; long rest is a confidence/usage consideration, not fixed skill penalty.

## Critical model disagreement
v1.3 proposes a huge late split: Samsung 3.10 vs KT 0.835 runs in inning6+ while F5 overwhelmingly KT (KT F5 69.2%). Existing KT bridge independent 2026 test shows adding named reliever quality/workload did NOT improve predictive performance, and KT's non-starter 6–8 run rate is similar to the rest of the league. **Do not treat v1.3's 6th+ split as proven.**
v1.2 gives moderate KT FG lean; do not average uncalibrated models into a pseudo-Champion.
All price edges/EV are unavailable until the **later** separate price check, and no retrospective change to this snapshot is allowed.

## PS 11-market audit status
- 1 Full W/L: vector frozen.
- 2 Full handicap: 0.5/1.5/2.5/3.5 lines frozen.
- 3 Full 승1패: vector frozen.
- 4 Full total: lines 3.5–18.5 frozen.
- 5 F5 W/D/L: vector frozen.
- 6 F5 handicap: lines 0.5/1.5/2.5/3.5 frozen.
- 7 F5 total: lines 2.5–14.5 frozen.
- 8 HR×score full joint: **UNAVAILABLE**, no calibrated HR/runs coupling.
- 9 14 final winning-margin bins plus residual 15-inning tie: frozen.
- 10 6 F5-final double combos plus residual draw: frozen.
- 11 exact total bins 1–19 and 20+, plus unquoted zero-run mass: frozen.

Critical limitation: 10–15 inning extra simulated independent Poisson inning increments without walkoff, so tail/exotic settlement is **RESEARCH PROVISIONAL** and not publishable as official priced prediction. No exact lineup, player-level HR model, prospectively validated first-follower mixture, or exact current active roster integrated.

`PS_PRE_SURFACE_AUDIT: games=1 | expected=11 | research_families=10 | unavailable=1 HR×score | official_champion=UNAVAILABLE | official_LOCK=NO | odds_seen=NO (at data generation)`
