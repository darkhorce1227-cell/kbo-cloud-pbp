# KBO PBP 자동화 — Windows 집 PC 한 번만 연결하기

## 확인된 원인

2026-11-06 오전 7:39 KST의 기존 GitHub-hosted KBO PBP Cloud Update는
api-gw.sports.naver.com 날짜별 및 시즌별 일정 요청에서 각각 ReadTimeout(20초)을 기록했다.
GitHub Actions 자체, PBP 데이터 검증 및 Release 배포는 정상 실행된다.
네이버까지의 네트워크 경로가 응답하지 않는 것이 문제이며,
네트워크의 정확한 차단 지점까지는 아직 확인되지 않았다.

## 최초 1회 설정 (PC에서)

1. 네이버 수동 수집에 성공했던 Windows PC를 준비한다.
2. GitHub 저장소 Settings > Actions > Runners > New self-hosted runner를 연다.
3. Windows / x64 선택 후 **GitHub가 화면에 제공하는** runner 다운로드 및 구성 명령을
   해당 PC에서 실행한다. 등록용 token을 채팅/메일/로그에 공개하지 않는다.
4. 추가 label 입력 질문에 kbo-naver-local을 입력한다. 기본 labels
   self-hosted / Windows / X64는 runner가 자동 추가한다.
5. 처음 시험할 때 runner 폴더에서 .\run.cmd 실행 후 GitHub Runner 목록에서
   Online 상태를 확인한다.
6. 저장소 Settings > Actions > General > Workflow permissions에서
   Read and write permissions를 허용한다.
7. GitHub Actions > KBO Naver PBP Local Capture (Windows self-hosted) > Run workflow
   실행 후 수집 로그를 확인한다. 매번 GitHub에서 수동 실행할 필요는 없다.
8. PC가 항상 동작하도록 하려면 공식 Windows runner 설치 안내에 따라
   전용 최소권한 계정으로 Windows 서비스를 구성하고, 절전 설정도 확인한다.
   권한 없는 컴퓨터에서 관리자 권한으로 실행하지 않는다.

## 이후 자동 진행 과정

- GitHub schedule이 집 PC의 self-hosted runner에 작업을 전달한다.
- PC에서 네이버 날짜별 일정 확인 후 마지막 검증일 다음 날부터 어제(KST)까지의
  완료 경기를 찾아 투구별 원본 Relay를 수집한다.
- GAME_END(type=99), 일정상 경기 ID/최종점수, F5, 이닝별 득점 및 SHA-256 검증을
  통과해야만 state/auto_pbp_raw/에 원본 ZIP을 저장한다.
- runner가 ZIP을 GitHub에 push하고, 별도의 workflow_dispatch API로 Linux
  오프라인 import 워크플로를 시작한다. 이 명시적 호출이 필요한 이유는
  GITHUB_TOKEN으로 발생시킨 일반 push가 다른 Actions 워크플로를 유발하지 않기 때문이다.
- Linux GitHub runner는 네이버 API를 아예 호출하지 않는다.
  기존 검증된 Parquet + game_PK index에 원본 릴레이를 병합하고
  PBP 점수감사, 9~15이닝, 중복 0, 기존 게임 모두 유지, SHA를 검증한다.
- 하나라도 FAIL이면 상태 및 최신 Release를 변경하지 않는다.
  전부 PASS여야만 새 Release가 올라간다.

## 테스트와 운영 제한

- 2026-11-06 테스트: 실제 네이버 9이닝/12이닝 수집 ZIP 오프라인 재생 PASS.
- 2026-11-06 테스트: 실제 730경기 이전 백업 + 11/4 원본 12이닝
  364행을 새 자동병합기로 추가하여 731경기/224,975행 도출 PASS.
- 실제 PC live 자동 스케줄은 아직 runner 등록 이전이므로 검증하지 않았다.
- PC가 꺼져 있거나 네이버 API가 PC에서도 응답하지 않으면 새 PBP를 만들지 않는다.
  다음 번 연결 시 지난 일자를 다시 확인한다.
- 현재 자동 importer는 2026년 검증된 baseline에만 append하도록 제한했다.
  2027년 새 시즌에는 해당 시즌 baseline/index bootstrap이 추가로 필요하다.
- 공개 저장소에 self-hosted runner를 연결하면 PC에서 실행되는 코드의 보안 위험이 있다.
  개인 업무 계정 대신 전용 OS 계정/전용 PC를 추천한다.
