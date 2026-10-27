# User-captured raw KBO PBP recovery

This directory is for an exact, original KBO/Naver nine-inning capture ZIP. Upload ONE file whose name matches `kbo_20261026_HTSS_raw_*.zip` through the GitHub file-upload UI. No renaming needed.

[Offline importer workflow](../../.github/workflows/import-manual-2026-10-26-pbp.yml) will independently verify SHA-256, source game, complete nine innings and the explicit Naver GAME_END(type=99) event. It then merges with the previous 724-game validated release, runs game-PK/date/score/duplicate checks, and only on PASS publishes the 725-game rolling release. No Naver HTTP access is required during import. The raw ZIP itself is retained as provenance.

Do not upload a modified ZIP or multiple matching captures. Other dates need separately validated importers.

For the 2026-10-26 game, the exact independently verified input is **`kbo_20261026_HTSS_raw_20261027_152005.zip`** with full ZIP SHA-256 **`abb9ba6656cd23ea80f18b8817d5da49f775cf317367608a345c2fe86ab7aec2`**. The importer enforces that hash before reconstruction. Navigate **Code → state → manual_pbp_raw → Add file → Upload files**, upload the unmodified ZIP and commit to `main`; its automated merge then runs without Naver network access.
