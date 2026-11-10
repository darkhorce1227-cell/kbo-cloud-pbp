# 재현 가능한 KBO 기준 모형과 개별 변수 추가 실험

모형 RS-BASE-v0.1-20261110 / 상태: 회고적 연구, Champion 승격 없음.
프로토콜 SHA256: `8073998bb60401f6949713df13c6299c4267433b61fe845bb547bfcc836220ac`

같은 경기에서 팀 기준 모형에 각 정보 묶음을 독립적으로 추가했다. 낮은 log loss가 좋다.

|모형|경기 수 F5/Full승패|F5 3방향 log loss|Full 조건부 승패 log loss|F5 득점 NLL|Full 득점 NLL|
|---|---:|---:|---:|---:|---:|
|add_bullpen_skill|1457/1422|1.00476|0.69635|2.15851|2.56886|
|add_expected_sp_outs|1457/1422|0.99582|0.68585|2.15167|2.56169|
|add_lineup_onbase_delta|1457/1422|1.00252|0.69344|2.15784|2.56681|
|add_lineup_power_delta|1457/1422|1.00227|0.69298|2.15736|2.56670|
|add_offense14|1457/1422|1.00225|0.69303|2.15778|2.56680|
|add_offense7|1457/1422|1.00208|0.69348|2.15758|2.56701|
|add_relief_workload|1457/1422|1.00205|0.69287|2.15761|2.56628|
|add_starter_skill|1457/1422|0.98557|0.67651|2.14329|2.55355|
|baseline_team|1457/1422|1.00208|0.69297|2.15746|2.56648|
|league_reference|1457/1422|1.00258|0.69483|2.15873|2.56874|

## 팀 기준 대비 차이

음수는 개선이다. 날짜 군집 95% 구간과 주 지표 전체 비교에 대한 Bonferroni 구간을 함께 보존했다.

|추가 변수|F5 Δ log loss (95% 구간)|Full Δ log loss (95% 구간)|
|---|---:|---:|
|add_bullpen_skill|+0.00268 [+0.00033, +0.00504]|+0.00338 [+0.00139, +0.00548]|
|add_expected_sp_outs|-0.00626 [-0.01155, -0.00105]|-0.00712 [-0.01119, -0.00296]|
|add_lineup_onbase_delta|+0.00045 [-0.00030, +0.00117]|+0.00047 [-0.00028, +0.00120]|
|add_lineup_power_delta|+0.00019 [-0.00031, +0.00072]|+0.00000 [-0.00068, +0.00068]|
|add_offense14|+0.00017 [-0.00068, +0.00106]|+0.00005 [-0.00161, +0.00174]|
|add_offense7|+0.00001 [-0.00136, +0.00137]|+0.00050 [-0.00095, +0.00195]|
|add_relief_workload|-0.00002 [-0.00029, +0.00026]|-0.00011 [-0.00055, +0.00032]|
|add_starter_skill|-0.01651 [-0.02343, -0.00959]|-0.01646 [-0.02264, -0.01023]|
|league_reference|+0.00051 [-0.00133, +0.00237]|+0.00186 [-0.00098, +0.00466]|

## 해석

- 모든 모델은 같은 경기와 같은 월별 학습 마감에서 비교했다.
- Full 승패는 실제 무승부를 제외한 조건부 두 방향이다. F5는 무승부를 포함한 세 방향이다.
- F5와 후반은 비음수 득점 분포이고 Full은 두 분포의 합성이다. 각 팀 Full 평균은 항상 F5 이상이다.
- 이미 연구자가 관찰한 역사 자료를 사용했다. 새 독립 전향 성능이나 당시 실제 운용 성적이 아니다.
- 좋은 결과가 나온 변수들을 사후 조합하지 않았다. 실제 베팅 가격·수익률을 계산하지 않았다.
- starter/lineup identity는 경기 후 PBP 재구성이다. 독립적으로 보존된 발표 시각이 없으므로 실전 사전예측 자격을 주장하지 않는다.
- 2024·2025는 9월 30일까지, 2026은 10월 12일까지다. 모든 시즌 전체 경기를 포함한다는 의미는 아니다.
- 차기 검증은 다음 미관찰 시즌의 실제 발표 라인업·선발·입력 시각을 동결해 진행한다. 기존 BP30 게이트는 별개로 유지한다.

## 입력

```json
[
  {
    "file": "kbo_pbp_2024_training.csv.gz",
    "sha256": "d9e4b59fc4c255990452c47d384977f689f338ce6b90988933906cb9e7874458",
    "year": 2024,
    "source_rows": 227955,
    "included_rows": 223015,
    "included_games": 719,
    "calendar_cutoff": "2024-09-30"
  },
  {
    "file": "kbo_pbp_2025_training.csv.gz",
    "sha256": "017ae5f39e79b2aeaa05e269cf32aebe7cc410c75fedd6e928850a61730acb61",
    "year": 2025,
    "source_rows": 222650,
    "included_rows": 215737,
    "included_games": 713,
    "calendar_cutoff": "2025-09-30"
  },
  {
    "file": "kbo_pbp_2026_latest.csv.gz",
    "sha256": "e854326bd1709bef884521f4e617d51fdc2e87bc3b59e9f04c28ce6517735615",
    "year": 2026,
    "source_rows": 225254,
    "included_rows": 221603,
    "included_games": 720,
    "calendar_cutoff": "2026-10-12"
  }
]
```

## 월별 범위

```json
[
  {
    "month": "2024-05",
    "status": "warmup",
    "prior_training_games": 0
  },
  {
    "month": "2024-06",
    "status": "warmup",
    "prior_training_games": 122
  },
  {
    "month": "2024-07",
    "status": "evaluated",
    "train_games": 247,
    "test_games": 97
  },
  {
    "month": "2024-08",
    "status": "evaluated",
    "train_games": 344,
    "test_games": 119
  },
  {
    "month": "2024-09",
    "status": "evaluated",
    "train_games": 463,
    "test_games": 98
  },
  {
    "month": "2025-05",
    "status": "evaluated",
    "train_games": 561,
    "test_games": 131
  },
  {
    "month": "2025-06",
    "status": "evaluated",
    "train_games": 692,
    "test_games": 111
  },
  {
    "month": "2025-07",
    "status": "evaluated",
    "train_games": 803,
    "test_games": 100
  },
  {
    "month": "2025-08",
    "status": "evaluated",
    "train_games": 903,
    "test_games": 127
  },
  {
    "month": "2025-09",
    "status": "evaluated",
    "train_games": 1030,
    "test_games": 91
  },
  {
    "month": "2026-05",
    "status": "evaluated",
    "train_games": 1121,
    "test_games": 128
  },
  {
    "month": "2026-06",
    "status": "evaluated",
    "train_games": 1249,
    "test_games": 123
  },
  {
    "month": "2026-07",
    "status": "evaluated",
    "train_games": 1372,
    "test_games": 102
  },
  {
    "month": "2026-08",
    "status": "evaluated",
    "train_games": 1474,
    "test_games": 86
  },
  {
    "month": "2026-09",
    "status": "evaluated",
    "train_games": 1560,
    "test_games": 104
  },
  {
    "month": "2026-10",
    "status": "evaluated",
    "train_games": 1664,
    "test_games": 40
  }
]
```
