# Missing postseason games: manual reconstruction v3

This directory contains a **conservative manual reconstruction** for the seven historical KBO postseason games whose Naver relay payloads are missing.

## Files

- `games.csv`: game identity, final score, and inning line score.
- `lineups.csv`: reconstructed starting batting orders/positions.
- `known_events.csv`: v1 event anchors supported by cited contemporary/secondary sources.
- `pa_reconstruction.csv`: base PA-level reconstruction. It mixes three explicitly flagged scopes:
  - `COMPLETE_GAME_PA_ORDER`: every recorded plate appearance in the game is represented in source order.
  - `COMPLETE_HALF_INNING_PA_ORDER`: every PA in that half-inning is represented in source order.
  - `PARTIAL_SEQUENCE_ANCHOR`: only the events actually described by the source are represented; missing PAs remain missing.
- `pa_reconstruction_additions.csv`: later source-backed PA rows discovered after the base file was assembled.
- `pa_reconstruction_corrections.csv`: authoritative field overrides for rows where later audit found that the raw row was too specific or where a stronger source adds game-state detail. `__BLANK__` means unknown/not attributable, not zero.
- `coverage.csv`: per-game effective PA-reconstruction coverage after additions.
- `superseded_known_events.csv`: v1 coarse rows that should no longer be used because finer PA detail was recovered.
- `sources.csv`: provenance and what each source supports.
- `part2_research_notes.md`: second-pass audit notes for the six games other than 2008 QP Game 1.

For analysis, do **not** consume the base CSV alone. Build the effective view with:

```bash
python scripts/build_manual_postseason_reconstruction.py
```

The builder unions `pa_reconstruction_additions.csv`, applies `pa_reconstruction_corrections.csv`, validates row counts and key source-backed anchors, and writes `pa_reconstruction_effective.csv`.

## Major reconstruction gains

For `33331008SSLT0` (2008 준PO 1차전 삼성@롯데), a contemporary Sports Donga live-text article contains the game from 1회초 through 9회말 in batting order. The reconstruction therefore contains **84 ordered PA rows for the full game**.

This is a complete **PA outcome/order reconstruction**, not complete pitch-by-pitch data. Exact pitch counts are entered only where the article explicitly prints them. Missing pitch counts, pitch types, locations, count states, and unprinted pitch sequences are never imputed.

The article's introductory paragraph contains an obvious `10-3` typo, while its detailed live log and final line record the actual `12-3` result. `games.csv` and the PA reconstruction retain the verified 12-3 final.

Current additional coverage:
- `33331009SSLT0` (2008 준PO 2차전): **20 source-backed PA anchors**. Third-pass OSEN audit added 2회초 박진만 삼진 and 6회초 채태인 사구, and strengthened the state context for 7회초 박진만 결승 2루타 (1사 만루) and 9회말 김주찬 적시 2루타 (1사 1루). Still partial.
- `33331011LTSS0` (2008 준PO 3차전): **24 source-backed PA rows**, with 3회초/3회말 complete PA order. 4회초 동점 득점은 이대호 안타→강민호 적시타까지 확인했고, third pass에서 5회초 이대호 내야땅볼 때 이인구 홈아웃을 추가했다. 해당 노컷뉴스 사진기사는 `5회말`로 표기하지만 롯데 타석/선수 문맥상 5회초 장면으로 보아 `MEDIUM` confidence로 보존한다.
- `33331011SKHT0` (2011 준PO 3차전): 2·4·6회 핵심 시퀀스와 6회말 나지완·이범호 삼진까지 12개 anchor.
- `77771026SKSS0` (2011 KS 2차전): **1회초, 6회초, 8회초 complete PA order**, plus 6회말 anchors, 총 20 rows.
- `33331011OBLT0` (2012 준PO 3차전): 주요 득점/주루 시퀀스 15 anchors. 선발 타순은 2012-10-11 Newsis 기사로 재검증했으며 기존 `lineups.csv`가 맞다.
- `77771029SSOB0` (2013 KS 5차전): 18 anchors. 기사에서 개별 득점 귀속이 명시되지 않은 연속안타는 correction overlay로 공란 처리.

The effective manual layer currently contains **193 rows across the seven missing games**. Only the 84 rows for 2008 QP Game 1 are a complete-game PA sequence; all other completeness claims are restricted to explicitly flagged half-innings.

## What this is NOT

This reconstruction is **not equivalent to the Naver pitch-level master** and must not be concatenated into that master as if the rows had the same observation unit.

Unknown plate appearances and pitches are intentionally left unknown. No missing pitch is synthesized. For `PARTIAL_SEQUENCE_ANCHOR`, `sequence_order` means order among the events preserved from the source, not the literal PA number in the half-inning. A `sequence_order` of `0` is used only when a newly discovered event is known to precede an older partial anchor whose key is retained for provenance. Only rows with `order_semantics=COMPLETE_PA_ORDER` can be treated as contiguous PA order.

`known_events.csv` is retained as audit history. When `superseded_known_events.csv` marks a v1 row as `SUPERSEDED`, use the corresponding effective PA rows instead and do not double-count both layers.

## Confidence

- `HIGH`: the PA/event is explicitly described in a contemporary source.
- `MEDIUM`: the sequence/result is strongly constrained by the source but one detail (for example inning label, exact RBI allocation, or out subtype) is not fully reliable in the source text.

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
