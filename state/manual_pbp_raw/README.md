# User-captured raw KBO PBP recovery

This directory is for an exact, original KBO/Naver nine-inning capture ZIP. Upload ONE file whose name matches `kbo_20261026_HTSS_raw_*.zip` through the GitHub file-upload UI. No renaming needed.

[Offline importer workflow](../../.github/workflows/import-manual-2026-10-26-pbp.yml) will independently verify SHA-256, source game, complete nine innings and the explicit Naver GAME_END(type=99) event. It then merges with the previous 724-game validated release, runs game-PK/date/score/duplicate checks, and only on PASS publishes the 725-game rolling release. No Naver HTTP access is required during import. The raw ZIP itself is retained as provenance.

Do not upload a modified ZIP or multiple matching captures. Other dates need separately validated importers.

For the 2026-10-26 game, the exact independently verified input is **`kbo_20261026_HTSS_raw_20261027_152005.zip`** with full ZIP SHA-256 **`abb9ba6656cd23ea80f18b8817d5da49f775cf317367608a345c2fe86ab7aec2`**. The importer enforces that hash before reconstruction. Navigate **Code → state → manual_pbp_raw → Add file → Upload files**, upload the unmodified ZIP and commit to `main`; its automated merge then runs without Naver network access.


## 2026-10-27: 삼성 8-3 KIA (playoff G2)

User-supplied original ZIP to upload unmodified:
`kbo_20261027_HTSS_raw_20261029_095724.zip`

SHA-256: `e122e245b09b1d346b31b1c8cc7573347e97d6115893e53f254b39131e74b8dd`

The separate workflow `.github/workflows/import-manual-2026-10-27-pbp.yml` triggers on that exact date prefix. It will merge the existing 725 verified games plus one real 10/27 game, require all 726 game IDs preserved, validate the binary score against Naver's original schedule (KIA 3, Samsung 8), recognize that the 9th bottom was not played, and only publish after a postgame binary audit PASS. The audit uses the schedule in the SAME uploaded ZIP; it is not a separate Naver fetch.

Upload page: https://github.com/darkhorce1227-cell/kbo-cloud-pbp/upload/main/state/manual_pbp_raw

Chat upload alone is not GitHub upload; to trigger merge, commit the exact original ZIP into this folder on `main`. See `research/pbp_recovery/2026_10_27_HTSS_original_zip_certificate.json` for its verified hash and ingestion-not-yet-complete status.
