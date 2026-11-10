"""Read-only diagnostics after the frozen run: reliability and archived market.

This does not refit models or choose feature combinations. The market comparison
was added after the first run as a descriptive stronger-benchmark audit.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from kbo_repro_baseline import paired_ci, probability_losses

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "league_reference": "리그 평균 기준", "baseline_team": "팀 장기 전력 기준",
    "add_starter_skill": "+ 상대 선발 K-BB·HR", "add_expected_sp_outs": "+ 상대 선발 예상 이닝",
    "add_offense7": "+ 최근 7일 공격력", "add_offense14": "+ 최근 14일 공격력",
    "add_bullpen_skill": "+ 상대 불펜 K-BB·HR", "add_relief_workload": "+ 직전 3일 불펜 PA량",
    "add_lineup_onbase_delta": "+ 평소 대비 타선 출루 차이", "add_lineup_power_delta": "+ 평소 대비 타선 장타 차이",
}


def calibration(pred):
    rows = []
    for name, g in pred.groupby("model"):
        decided = g[g.full_home_actual != g.full_away_actual]
        items = [("full_decided", "HOME", decided.full_p_home_decided.to_numpy(),
                  (decided.full_home_actual > decided.full_away_actual).to_numpy())]
        for side, col in [("AWAY", "away"), ("DRAW", "draw"), ("HOME", "home")]:
            a, h = g.f5_away_actual, g.f5_home_actual
            y = a > h if side == "AWAY" else a == h if side == "DRAW" else h > a
            items.append(("f5_wdl", side, g["f5_p_" + col].to_numpy(), y.to_numpy()))
        for market, side, p, y in items:
            assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1
            bins = np.minimum((p * 10).astype(int), 9)
            n_total = 0
            for i in range(10):
                mask = bins == i
                n = int(mask.sum())
                if not n:
                    continue
                n_total += n
                rows.append({"model": name, "market": market, "class": side,
                             "bin_low": i / 10, "bin_high": (i + 1) / 10,
                             "n": n, "mean_probability": float(p[mask].mean()),
                             "observed_frequency": float(y[mask].mean()),
                             "gap_observed_minus_predicted": float(y[mask].mean() - p[mask].mean())})
            assert n_total == len(p)
    return pd.DataFrame(rows)


def market_benchmark(pred, quotes_path, cfg):
    quotes = pd.read_csv(quotes_path)
    quotes = quotes[quotes.market_type == "FULL_ML2"]
    assert not quotes.game_pk.duplicated().any()
    rows = []
    for q in quotes.itertuples():
        if q.home_score == q.away_score:
            continue
        odds = json.loads(q.odds_json)
        assert odds["HOME"] > 1 and odds["AWAY"] > 1
        ih, ia = 1 / odds["HOME"], 1 / odds["AWAY"]
        ph = ih / (ih + ia)
        y = int(q.home_score > q.away_score)
        ll, bs = probability_losses([1 - ph, ph], y)
        rows.append({"game_pk": q.game_pk, "date": q.game_date, "market_probability_home": ph,
                     "market_logloss": ll, "market_brier": bs, "y": y,
                     "home_actual": q.home_score, "away_actual": q.away_score})
    market = pd.DataFrame(rows)
    common = set(market.game_pk) & set(pred.game_pk)
    market = market[market.game_pk.isin(common)].set_index("game_pk")
    result = {"status": "POST_RUN_DESCRIPTIVE_INFORMATION_BENCHMARK_ONLY", "games": len(market),
              "price_temporality": "Historical displayed prices, captured retrospectively; pregame/close timing unverified",
              "source_sha256": hashlib.sha256(Path(quotes_path).read_bytes()).hexdigest(),
              "market": {"logloss": float(market.market_logloss.mean()), "brier": float(market.market_brier.mean()),
                         "hit_rate": float(((market.market_probability_home > .5) == market.y).mean())},
              "models": {}}
    for name, g in pred.groupby("model"):
        g = g[g.game_pk.isin(common)].set_index("game_pk").reindex(market.index)
        assert (g.full_home_actual == market.home_actual).all()
        assert (g.full_away_actual == market.away_actual).all()
        result["models"][name] = {"logloss": float(g.full_decided_logloss.mean()),
                                  "brier": float(g.full_decided_brier.mean()),
                                  "hit_rate": float(g.full_decided_hit.mean()),
                                  "delta_vs_market": paired_ci((g.full_decided_logloss - market.market_logloss).to_numpy(),
                                                                 market.date.to_numpy(), cfg)}
    return result


def coefficient_replay(pred, fits_path, features_path):
    features = pd.read_csv(features_path)
    features["month"] = features.date.str[:7]
    fits = json.loads(Path(fits_path).read_text())
    max_error, count = 0., 0
    for key, fit in fits.items():
        month, model, phase = key.split("/")
        f = features[(features.month == month) & (features.phase == phase)]
        pp = pred[pred.model == model].set_index("game_pk")
        f = f[f.game_pk.isin(pp.index)]
        x = (f[fit['features']].to_numpy(float) - fit['scaler_mean']) / fit['scaler_scale']
        mu = f.offset.to_numpy() * np.exp(fit['intercept'] + x @ np.array(fit['coef']))
        expected = np.array([pp.loc[pk, phase + ('_home_mu' if home else '_away_mu')]
                             for pk, home in zip(f.game_pk, f.is_home)])
        if len(mu):
            max_error = max(max_error, float(np.max(np.abs(mu - expected))))
            count += len(mu)
    assert count == len(pred) * 4 and max_error < 1e-7
    return {"phase_team_forecasts": count, "max_abs_mu_error": max_error,
            "tolerance": 1e-7, "note": "Saved prediction CSV uses 10 significant digits"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "research/repro_baseline/results"))
    ap.add_argument("--quotes", default=str(ROOT / "research/market_fair_odds/sharppick_2026_proto_r105_r122_combined_kbo.csv"))
    ap.add_argument("--replay-features")
    ap.add_argument("--independent-predictions")
    args = ap.parse_args()
    out = Path(args.results)
    cfg = json.loads((ROOT / "research/repro_baseline/protocol.json").read_text())
    pred = pd.read_csv(out / "predictions.csv")
    metrics = json.loads((out / "metrics.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    assert not pred.duplicated(["game_pk", "model"]).any()
    assert (pd.to_datetime(pred.train_max_date) < pd.to_datetime(pred.date)).all()
    assert (pred.full_away_mu >= pred.f5_away_mu).all() and (pred.full_home_mu >= pred.f5_home_mu).all()
    assert pred.groupby("model").game_pk.nunique().nunique() == 1
    cal = calibration(pred)
    cal.to_csv(out / "calibration.csv", index=False, float_format="%.10g")
    market = market_benchmark(pred, args.quotes, cfg)
    (out / "market_information_benchmark.json").write_text(json.dumps(market, ensure_ascii=False, indent=2) + "\n")
    audit = {"validation": "PASS", "games": manifest["eligible_games"], "forecast_rows": len(pred),
             "model_count": pred.model.nunique(), "calibration_rows": len(cal),
             "code_sha256": hashlib.sha256((ROOT / "scripts/kbo_repro_baseline.py").read_bytes()).hexdigest(),
             "review_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "protocol_sha256": hashlib.sha256((ROOT / "research/repro_baseline/protocol.json").read_bytes()).hexdigest(),
             "checks": ["one row per model/game", "same game counts", "training cutoff before game", "Full mu >= F5 mu",
                        "calibration bin denominators equal source counts", "archived market score matches PBP target"],
             "calibration_note": "10 fixed-width probability bins, descriptive only; no calibration function fitted"}
    assert audit["protocol_sha256"] == manifest["protocol_sha256"]
    if args.replay_features:
        audit['coefficient_replay'] = coefficient_replay(pred, out / 'fitted_models.json', args.replay_features)
        audit['feature_snapshot_sha256'] = hashlib.sha256(Path(args.replay_features).read_bytes()).hexdigest()
    if args.independent_predictions:
        other = pd.read_csv(args.independent_predictions).sort_values(['model', 'game_pk']).reset_index(drop=True)
        current = pred.sort_values(['model', 'game_pk']).reset_index(drop=True)
        assert current[['model', 'game_pk']].equals(other[['model', 'game_pk']])
        numeric = [c for c in current if pd.api.types.is_numeric_dtype(current[c])]
        error = float(np.nanmax(np.abs(current[numeric].to_numpy() - other[numeric].to_numpy())))
        assert error < 1e-7
        audit['independent_replay'] = {'forecast_rows': len(pred), 'max_abs_numeric_difference': error}
    (out / "review_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    assert {s['year'] for s in manifest['sources']} == {2024, 2025, 2026}, "This review describes the frozen multiyear run"
    base = metrics['aggregate']['baseline_team']
    sp = metrics['aggregate']['add_starter_skill']
    lines = ["# 1차 검증 결과와 다음 판단", "", "## 확인한 개선", "",
             f"평가 {manifest['eligible_games']:,}경기, Full 승패는 무승부 제외 {base['decided_games']:,}경기다. 전체 {manifest['model_count']}개 모형에 같은 경기와 같은 월별 학습 마감을 적용했다.",
             "선발 K-BB·HR 추가는 2024·2025·2026 각 연도에서 두 주 지표를 개선했다. 16개 주 비교에 대한 보정 구간도 두 지표 모두 0 아래였다.",
             "선발 예상 이닝 추가는 Full 승패에서 보정 후 개선이 남았지만, F5는 보정 구간이 0을 포함했다. 두 유망 정보를 이번 결과를 보고 조합하지 않았다.", "",
             "|추가 정보|F5 Δ log loss|Full Δ log loss|판단|", "|---|---:|---:|---|"]
    for name in cfg["additions_to_baseline_team_individually"]:
        q = metrics["paired_vs_team"][name]
        a, b = q["f5_wdl_logloss"], q["full_decided_logloss"]
        verdict = "두 지표 개선 근거" if a['familywise_ci'][1] < 0 and b['familywise_ci'][1] < 0 else "Full 개선, F5 불확실" if b['familywise_ci'][1] < 0 else "Full 악화 근거" if b['familywise_ci'][0] > 0 else "추가 개선 확인 못함"
        lines.append(f"|{LABELS[name]}|{a['delta']:+.5f}|{b['delta']:+.5f}|{verdict}|")
    lines += ["", "낮은 값이 좋다. 위 판단의 개선·악화 근거는 주 지표 전체 다중 비교 보정 구간을 기준으로 한다. 정확한 구간은 metrics.json에 있다.",
              "", "## 기준선의 실용성도 따로 확인", "",
              f"Full 승패 적중률은 팀 기준 약 {base['full_decided_hit']*100:.1f}%, 선발 능력 추가 약 {sp['full_decided_hit']*100:.1f}%였다. 이것은 새 연구 모형의 같은 표본 비교다. 과거 수동 Champion의 73.1%와 같은 기록이 아니다.",
              "단순 기준선을 이긴 결과만으로 실전 경쟁력을 주장하지 않기 위해, 원래 실험 실행 뒤 별도의 과거 표시 배당 기준도 비교했다. 추가 변수나 계수를 이 결과에 맞춰 바꾸지 않았다.", "",
              f"공통 {market['games']}경기에서 시장 무마진 Full 승패 log loss {market['market']['logloss']:.5f}.", "",
              "|모형|동일 경기 log loss|모형−시장 차이와 날짜 군집 95% 구간|", "|---|---:|---:|"]
    for name in ["baseline_team", "add_starter_skill", "add_expected_sp_outs"]:
        r = market["models"][name]
        d = r["delta_vs_market"]
        lines.append(f"|{LABELS[name]}|{r['logloss']:.5f}|{d['delta']:+.5f} [{d['ci95'][0]:+.5f}, {d['ci95'][1]:+.5f}]|")
    lines += ["", "선발 능력 후보가 이 시장 표본을 이겼다는 근거는 없다. 배당은 사후 수집된 과거 표시값이므로 같은 정보 시점의 전향 시장 비교·CLV·실행 수익 검증도 아니다.",
              "", "## 남긴 결과와 제한", "",
              "- 월별 계수·스케일링·입력 해시·경기별 예측·연도별 오차·신뢰구간을 보존했다.",
              "- 확률대별 실제 빈도는 calibration.csv에 있다. 작은 구간의 적중률을 보정 계수로 재사용하지 않았다.",
              "- 원자료 구성에서 2025-04-19 NC–한화 1경기는 F5 완료 상태를 확인하지 못해 공통 제외했다. 2024/25는 Sep30, 2026은 Oct12까지만 포함했다.",
              "- 선발 신원과 타선은 경기 후 PBP proxy다. 독립적인 발표 시각이 없는 한 실제 사전예측이라고 부르지 않는다.",
              "- 과거 시즌의 결과는 이미 관찰된 개발 자료다. 이번 비교의 선택 편향을 bootstrap이나 다중 비교 보정이 없애지는 못한다.",
              "- 이 단계의 산출물은 다음 검증에 쓸 재현 가능한 연구 기준선이다. 공식 Champion이나 시장 대비 우위를 확보한 베팅 모형으로 승격하지 않는다.",
              "", "## 다음 시험", "",
              "새 독립 표본에서 팀+선발 능력 후보를 기준으로 고정하고, 예상 이닝/교체 경로를 한 단계 추가하는 시험을 우선한다. 실제 발표 선발·라인업·불펜 가용성을 경기 전에 저장해야 한다. 기존 v0.9 BP30 시험과 공식 장부는 별개로 유지한다.", ""]
    if 'coefficient_replay' in audit:
        q = audit['coefficient_replay']
        lines += ["## 재현 검사", "", f"저장 계수·스케일링·입력에서 {q['phase_team_forecasts']:,}개 팀/구간 예측을 다시 계산했다. 저장된 CSV 평균과 최대 차이는 {q['max_abs_mu_error']:.3g}점이며 CSV 반올림 허용치 안이다."]
    if 'independent_replay' in audit:
        lines += [f"별도 실행 환경에서 계산한 {len(pred):,}개 모형/경기 행은 저장된 GitHub 실행 결과와 수치 차이 {audit['independent_replay']['max_abs_numeric_difference']:.3g}이었다.", ""]
    (out / "INTERPRETATION.md").write_text("\n".join(lines))
    print(json.dumps({"audit": audit, "market_games": market["games"], "market_ll": market["market"]["logloss"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
