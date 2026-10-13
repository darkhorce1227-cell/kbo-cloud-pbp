# Missing postseason games: manual reconstruction v1

This directory contains a **conservative manual reconstruction** for the seven historical KBO postseason games whose Naver relay payloads are missing.

## What this is

- `games.csv`: game identity, final score, and inning line score.
- `lineups.csv`: reconstructed starting batting orders/positions.
- `known_events.csv`: only events explicitly supported by the cited contemporary/secondary sources.
- `sources.csv`: provenance and what each source supports.

## What this is NOT

This is **not pitch-by-pitch data** and must not be concatenated into the pitch-level master as if it were equivalent to Naver relay output.

Unknown plate appearances and pitches are intentionally left unknown. No missing PA is synthesized from batting order, box score totals, or score arithmetic alone.

Quality flag: `PARTIAL_MANUAL_RECONSTRUCTION`.

`event_order_hint` means order among the events that this reconstruction records in that half-inning; it is **not** a complete PA sequence number.

Confidence:
- `HIGH`: the event is explicitly described in a contemporaneous news source.
- `MEDIUM`: inning total or grouped sequence is secure, but the exact PA-level allocation is not fully supported by the currently collected sources.

## Current usefulness

These rows are suitable for:
- restoring game-level inclusion in postseason samples;
- score/inning state and known scoring-event studies;
- lineup/start-player context;
- manual cross-checking if full PBP is later recovered.

They are not suitable for:
- pitch selection/count analysis;
- complete PA denominators;
- BvP/PvB calculated as if every matchup were observed;
- pitch-sequence, CSW, chase, whiff, or location studies.

When an original/authorized PBP source is recovered, preserve these files as provenance/audit material and replace the analytical use of these rows with the recovered PBP.
