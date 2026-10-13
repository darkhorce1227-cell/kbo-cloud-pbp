# Missing postseason games: manual reconstruction v2

This directory contains a **conservative manual reconstruction** for the seven historical KBO postseason games whose Naver relay payloads are missing.

## Files

- `games.csv`: game identity, final score, and inning line score.
- `lineups.csv`: reconstructed starting batting orders/positions.
- `known_events.csv`: v1 event anchors supported by cited contemporary/secondary sources.
- `pa_reconstruction.csv`: v2 base PA-level reconstruction. It mixes three explicitly flagged scopes:
  - `COMPLETE_GAME_PA_ORDER`: every recorded plate appearance in the game is represented in source order.
  - `COMPLETE_HALF_INNING_PA_ORDER`: every PA in that half-inning is represented in source order.
  - `PARTIAL_SEQUENCE_ANCHOR`: only the events actually described by the source are represented; missing PAs remain missing.
- `pa_reconstruction_additions.csv`: later source-backed PA rows discovered after the base v2 file was assembled.
- `pa_reconstruction_corrections.csv`: authoritative field overrides for rows where later audit found that the raw v2 row was too specific. `__BLANK__` means unknown/not attributable, not zero.
- `coverage.csv`: per-game effective PA-reconstruction coverage after additions.
- `superseded_known_events.csv`: v1 coarse rows that should no longer be used because v2 recovered finer PA detail.
- `sources.csv`: provenance and what each source supports.
- `part2_research_notes.md`: second-pass audit notes for the six games other than 2008 QP Game 1.

For analysis, do **not** consume the base CSV alone. Build the effective view with:

```bash
python scripts/build_manual_postseason_reconstruction.py
```

The builder unions `pa_reconstruction_additions.csv`, applies `pa_reconstruction_corrections.csv`, validates row counts and the full-game score audit, and writes `pa_reconstruction_effective.csv`.

## Major v2 upgrade

For `33331008SSLT0` (2008 준PO 1차전 삼성@롯데), a contemporary Sports Donga live-text article contains the game from 1회초 through 9회말 in batting order. The reconstruction therefore contains **84 ordered PA rows for the full game**.

This is a complete **PA outcome/order reconstruction**, not complete pitch-by-pitch data. Exact pitch counts are entered only where the article explicitly prints them. Missing pitch counts, pitch types, locations, count states, and unprinted pitch sequences are never imputed.

The article's introductory paragraph contains an obvious `10-3` typo, while its detailed live log and final line record the actual `12-3` result. `games.csv` and the PA reconstruction retain the verified 12-3 final.

Additional v2 coverage after the second-pass audit:
- `33331009SSLT0` (2008 준PO 2차전): scoring anchors plus additional 3·5·7회 PA anchors from contemporaneous OSEN photo/article pages; still partial.
- `33331011LTSS0` (2008 준PO 3차전): 3회초/3회말 complete PA order; 4회초 동점 득점은 동시대 사진기사 교차검증으로 이대호 안타→강민호 적시타까지 복구; other innings remain source-backed anchors.
- `33331011SKHT0` (2011 준PO 3차전): 2·4·6회 핵심 시퀀스에 더해 6회말 박희수 등판 뒤 나지완·이범호 삼진을 추가 확인.
- `77771026SKSS0` (2011 KS 2차전): **1회초, 6회초, 8회초 complete PA order**, plus 6회말 anchors.
- `33331011OBLT0` (2012 준PO 3차전): 주요 득점/주루 시퀀스는 partial anchor로 보존. 선발 타순은 2012-10-11 Newsis 기사로 재검증했으며 기존 `lineups.csv`가 맞다.
- `77771029SSOB0` (2013 KS 5차전): 1·2·3·5·8·9회 주요 득점/작전 앵커; 기사에서 개별 득점 귀속이 명시되지 않은 연속안타는 correction overlay로 공란 처리.

## What this is NOT

This reconstruction is **not equivalent to the Naver pitch-level master** and must not be concatenated into that master as if the rows had the same observation unit.

Unknown plate appearances and pitches are intentionally left unknown. No missing pitch is synthesized. For `PARTIAL_SEQUENCE_ANCHOR`, `sequence_order` means order among the events preserved from the source, not the literal PA number in the half-inning. Only rows with `order_semantics=COMPLETE_PA_ORDER` can be treated as contiguous PA order.

`known_events.csv` is retained as audit history. When `superseded_known_events.csv` marks a v1 row as `SUPERSEDED`, use the corresponding v2 effective rows instead and do not double-count both layers.

## Confidence

- `HIGH`: the PA/event is explicitly described in a contemporary source.
- `MEDIUM`: the sequence/result is strongly constrained by the source but one detail (for example exact RBI allocation or out subtype) is not explicitly stated.

## Analytical use

Suitable:
- restoring game-level inclusion in postseason samples;
- score/inning state and known scoring-event studies;
- lineup/start-player context;
- PA-level analysis **only within scopes flagged complete**;
- manual cross-checking if original PBP is later recovered.

Not suitable without additional source recovery:
- full-series or full-game PA denominators for games marked partial;
- BvP/PvB calculated as if every matchup were observed;
- pitch selection/count analysis;
- CSW, chase, whiff, velocity, location, or pitch-sequence studies.

When an original/authorized PBP source is recovered, preserve these files as provenance/audit material and replace the analytical use of manual rows with the recovered PBP.
