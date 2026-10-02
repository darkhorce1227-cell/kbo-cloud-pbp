# KBO Cloud PBP Updater v0.2.0

`KBO Daily Analysis`용 네이버 Sports PBP 클라우드 현행화 모듈입니다.

## 동작

- PC 없이 GitHub Actions에서 실행합니다.
- 경기일에만 의미 있는 작업을 합니다.
- **14:00 경기가 하나라도 있으면 17:00 KST부터**, 그 외에는 **22:00 KST부터** 체크합니다. 필요하면 다음 날 05시대까지 재시도합니다.
- 이후 1시간 단위로 확인하되, 그날 대상 경기의 PBP 검증이 모두 통과하면 즉시 그날 작업을 끝냅니다.
- 아직 경기 종료 전이거나 네이버 PBP가 덜 올라왔으면 다음 시간에 다시 시도합니다.
- 수동 실행도 가능합니다.

GitHub scheduled workflow는 정확히 정각을 보장하지 않으므로 17:07, 18:07… 05:07 정도에 깨어나도록 설정합니다. 이미 PASS한 날짜는 preflight에서 즉시 종료합니다.

## 원자료

PBP parser는 공개 프로젝트 `slothman3878/kbo_pbp_naver_sports`를 사용합니다.

- Naver Sports schedule/relay API 기반
- 한 투구당 한 행
- upstream 자체 schema/validation 보유
- 공식 시즌 기록과 교차검증됨

## 승격 조건

candidate 데이터가 다음을 모두 만족해야 authoritative로 승격합니다.

1. 해당 날짜 정규시즌/포스트시즌 예정 경기(취소 제외)를 모두 식별
2. 해당 경기들이 모두 종료 상태
3. target date의 모든 expected game ID가 PBP에 존재
4. 모든 expected game의 row > 0
5. 핵심 컬럼 존재
6. `(game_pk, at_bat_number, pitch_number)` duplicate = 0
7. target date가 candidate PBP의 max date 이하가 아니라 실제로 포함됨
8. 직전 성공 dataset보다 설명 없는 row regression 없음

실패하면 기존 authoritative 데이터는 건드리지 않습니다.

## 산출물

성공 시 rolling GitHub Release tag `latest-pbp`에 다음을 올립니다.

- `kbo_pbp_<YEAR>_latest.csv.gz`
- `kbo_pbp_<YEAR>_latest.parquet`
- `manifest.json`

또한 repo의 `state/status.json`을 갱신합니다.

## 배포

공개 GitHub repository에 이 폴더를 올린 뒤 Actions write permission을 허용합니다.

수동 테스트:
1. Actions → `KBO PBP Cloud Update`
2. Run workflow
3. `target_date`를 이미 종료된 KBO 경기 날짜로 넣어 테스트 가능
4. PASS 시 Release `latest-pbp` 생성 여부 확인

## 주의

Naver Sports gateway는 비공식 공개 endpoint라 무공지 변경 가능성이 있습니다.
그 경우 이 시스템은 **잘못된 파일을 승격하는 대신 validation FAIL**하도록 설계했습니다.
