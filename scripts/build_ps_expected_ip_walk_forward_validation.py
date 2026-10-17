#!/usr/bin/env python3
"""Step 5: validate leakage-free postseason Expected-IP simple-shift challenger.

For target season Y (2017-2025):
  baseline_eip = same-season player RS starter-average IP
  candidate_eip = baseline_eip + expanding historical PS mean shift for Y

The expanding shift is produced in Step 4 using only postseason seasons < Y.
Rows with no same-season RS starter history remain UNAVAILABLE and are not imputed.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE_CI_NONBINDING = [-0.859, -0.312]


def rmse(err):
    s = pd.Series(err, dtype=float)
    return float(math.sqrt((s * s).mean()))


def bootstrap_mean_ci(values, reps=20000, seed=20261017):
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(x)
    if n == 0:
        return [None, None]
    # chunk to avoid a large permanent allocation
    out = np.empty(reps, dtype=float)
    chunk = 1000
    for start in range(0, reps, chunk):
        k = min(chunk, reps - start)
        idx = rng.integers(0, n, size=(k, n))
        out[start:start+k] = x[idx].mean(axis=1)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return [float(lo), float(hi)]


def cluster_bootstrap_mean_ci(frame, value_col, cluster_col, reps=20000, seed=20261017):
    clusters = frame[cluster_col].drop_duplicates().tolist()
    grouped = {k: frame.loc[frame[cluster_col] == k, value_col].to_numpy(dtype=float) for k in clusters}
    rng = np.random.default_rng(seed)
    out = np.empty(reps, dtype=float)
    m = len(clusters)
    for r in range(reps):
        sampled = rng.integers(0, m, size=m)
        vals = np.concatenate([grouped[clusters[i]] for i in sampled])
        out[r] = vals.mean()
    lo, hi = np.percentile(out, [2.5, 97.5])
    return [float(lo), float(hi)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--joined-input', default='research/postseason_expected_ip/ps_rs_starter_avg_ip.csv')
    ap.add_argument('--step3-validation', default='research/postseason_expected_ip/ps_rs_starter_avg_ip_validation.json')
    ap.add_argument('--shift-input', default='research/postseason_expected_ip/ps_expected_ip_expanding_shift_by_target_season.csv')
    ap.add_argument('--step4-validation', default='research/postseason_expected_ip/ps_expected_ip_expanding_shift_validation.json')
    ap.add_argument('--frozen-reference', default='research/postseason_expected_ip/frozen_2026_prior.json')
    ap.add_argument('--output-dir', default='research/postseason_expected_ip')
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    joined = pd.read_csv(args.joined_input, dtype={'game_pk': str, 'starter_id': str})
    shifts = pd.read_csv(args.shift_input)
    step3 = json.loads(Path(args.step3_validation).read_text(encoding='utf-8'))
    step4 = json.loads(Path(args.step4_validation).read_text(encoding='utf-8'))
    frozen = json.loads(Path(args.frozen_reference).read_text(encoding='utf-8'))
    reference = frozen['historical_validation_reference']

    eval_all = joined[joined['season'].between(2017, 2025)].copy()
    eval_all = eval_all.merge(
        shifts[['target_season', 'historical_ps_mean_ip_shift']],
        left_on='season', right_on='target_season', how='left', validate='many_to_one'
    )

    if eval_all['historical_ps_mean_ip_shift'].isna().any():
        raise ValueError('missing Step-4 shift for an evaluation season')
    if (eval_all['target_season'] != eval_all['season']).any():
        raise ValueError('target-season shift merge mismatch')

    eligible = eval_all[eval_all['rs_link_status'] == 'MATCHED'].copy()
    unavailable = eval_all[eval_all['rs_link_status'] != 'MATCHED'].copy()

    eligible['baseline_rs_eip'] = eligible['rs_starter_avg_ip']
    eligible['candidate_ps_eip'] = eligible['rs_starter_avg_ip'] + eligible['historical_ps_mean_ip_shift']
    eligible['baseline_error'] = eligible['baseline_rs_eip'] - eligible['actual_ip']
    eligible['candidate_error'] = eligible['candidate_ps_eip'] - eligible['actual_ip']
    eligible['baseline_sq_error'] = eligible['baseline_error'] ** 2
    eligible['candidate_sq_error'] = eligible['candidate_error'] ** 2
    eligible['mse_delta_candidate_minus_baseline'] = eligible['candidate_sq_error'] - eligible['baseline_sq_error']

    # Keep all 258 evaluation appearances in the published prediction table.
    pred = eval_all.copy()
    for col in [
        'baseline_rs_eip', 'candidate_ps_eip', 'baseline_error', 'candidate_error',
        'baseline_sq_error', 'candidate_sq_error', 'mse_delta_candidate_minus_baseline'
    ]:
        pred[col] = np.nan
    pred.loc[eligible.index, [
        'baseline_rs_eip', 'candidate_ps_eip', 'baseline_error', 'candidate_error',
        'baseline_sq_error', 'candidate_sq_error', 'mse_delta_candidate_minus_baseline'
    ]] = eligible[[
        'baseline_rs_eip', 'candidate_ps_eip', 'baseline_error', 'candidate_error',
        'baseline_sq_error', 'candidate_sq_error', 'mse_delta_candidate_minus_baseline'
    ]]
    pred['prediction_status'] = np.where(pred['rs_link_status'].eq('MATCHED'), 'EVALUATED', 'UNAVAILABLE')

    season_rows = []
    for season in range(2017, 2026):
        all_s = eval_all[eval_all['season'] == season]
        g = eligible[eligible['season'] == season]
        b = rmse(g['baseline_error'])
        c = rmse(g['candidate_error'])
        season_rows.append({
            'season': season,
            'games_total': int(all_s['game_pk'].nunique()),
            'starter_appearances_total': int(len(all_s)),
            'eligible_appearances': int(len(g)),
            'unavailable_appearances': int(len(all_s) - len(g)),
            'historical_shift': float(all_s['historical_ps_mean_ip_shift'].iloc[0]),
            'baseline_rs_ip_rmse': b,
            'baseline_rs_ip_mae': float(g['baseline_error'].abs().mean()),
            'candidate_rmse': c,
            'candidate_mae': float(g['candidate_error'].abs().mean()),
            'candidate_beats_baseline_rmse': bool(c < b),
            'mse_delta_candidate_minus_baseline': float(g['mse_delta_candidate_minus_baseline'].mean()),
        })
    season_df = pd.DataFrame(season_rows)

    mse_delta = eligible['mse_delta_candidate_minus_baseline'].to_numpy(dtype=float)
    mean_delta = float(mse_delta.mean())
    se_delta = float(mse_delta.std(ddof=1) / math.sqrt(len(mse_delta)))
    normal_ci = [float(mean_delta - 1.96 * se_delta), float(mean_delta + 1.96 * se_delta)]
    appearance_boot_ci = bootstrap_mean_ci(mse_delta)
    game_boot_ci = cluster_bootstrap_mean_ci(
        eligible, 'mse_delta_candidate_minus_baseline', 'game_pk'
    )

    metrics = {
        'status': 'PASS',
        'step': 5,
        'definition': '2017-2025 leakage-free walk-forward evaluation of RS-IP baseline vs historical-PS-shift challenger',
        'formula': 'candidate_ps_eip = same-season RS starter avg IP + Step-4 shift for target season',
        'policy': {
            'target_or_future_ps_data': 'FORBIDDEN',
            'unavailable_rs_baseline_imputation': 'FORBIDDEN',
            'evaluation_metrics_population': 'MATCHED rows only',
            'all_258_rows_retained_in_prediction_artifact': True,
        },
        'coverage': {
            'evaluation_seasons': list(range(2017, 2026)),
            'games_total': int(eval_all['game_pk'].nunique()),
            'starter_appearances_total': int(len(eval_all)),
            'eligible_evaluated_appearances': int(len(eligible)),
            'unavailable_appearances': int(len(unavailable)),
        },
        'rebuilt_metrics_256_eligible': {
            'baseline_rs_ip_rmse': rmse(eligible['baseline_error']),
            'baseline_rs_ip_mae': float(eligible['baseline_error'].abs().mean()),
            'candidate_rmse': rmse(eligible['candidate_error']),
            'candidate_mae': float(eligible['candidate_error'].abs().mean()),
            'candidate_rmse_season_wins': int(season_df['candidate_beats_baseline_rmse'].sum()),
            'candidate_rmse_seasons_evaluated': int(len(season_df)),
            'mse_delta_candidate_minus_baseline': mean_delta,
            'mse_delta_normal_95ci': normal_ci,
            'mse_delta_appearance_bootstrap_95ci': appearance_boot_ci,
            'mse_delta_game_cluster_bootstrap_95ci': game_boot_ci,
            'bootstrap_reps': 20000,
            'bootstrap_seed': 20261017,
        },
        'old_phase2_reference_258_claimed': {
            'starter_appearances': int(reference['evaluation_starter_appearances']),
            'baseline_rs_ip_rmse': float(reference['rs_ip_rmse']),
            'baseline_rs_ip_mae': float(reference['rs_ip_mae']),
            'candidate_rmse': float(reference['ps_intercept_rmse']),
            'candidate_mae': float(reference['ps_intercept_mae']),
            'mse_delta_candidate_minus_baseline': float(reference['rs_vs_candidate_mse_delta']),
            'candidate_rmse_season_wins': 9,
            'candidate_rmse_seasons_evaluated': 9,
            'mse_delta_95ci_nonbinding_memory_reference': REFERENCE_CI_NONBINDING,
        },
        'reference_delta_rebuilt_minus_old': {},
        'unavailable_details': unavailable[[
            'season','game_date','game_pk','team','starter_id','starter_name','actual_ip','rs_link_status'
        ]].rename(columns={'actual_ip': 'actual_ps_ip'}).to_dict(orient='records'),
        'by_season': season_df.to_dict(orient='records'),
        'checks': {},
        'inputs': {
            'joined_step3': args.joined_input,
            'step3_validation': args.step3_validation,
            'expanding_shift_step4': args.shift_input,
            'step4_validation': args.step4_validation,
            'frozen_reference': args.frozen_reference,
        },
        'outputs': {
            'predictions': str(out / 'ps_expected_ip_walk_forward_predictions_2017_2025.csv'),
            'by_season': str(out / 'ps_expected_ip_walk_forward_validation_by_season.csv'),
            'validation': str(out / 'ps_expected_ip_walk_forward_validation.json'),
        }
    }

    r = metrics['rebuilt_metrics_256_eligible']
    old = metrics['old_phase2_reference_258_claimed']
    for key in [
        'baseline_rs_ip_rmse','baseline_rs_ip_mae','candidate_rmse','candidate_mae',
        'mse_delta_candidate_minus_baseline','candidate_rmse_season_wins'
    ]:
        metrics['reference_delta_rebuilt_minus_old'][key] = float(r[key] - old[key])
    metrics['reference_delta_rebuilt_minus_old']['starter_appearances'] = int(len(eligible) - old['starter_appearances'])

    expected_unavailable = {
        (2018, '33331023HHWO02018', '68703'),
        (2022, '77771105SKWO02022', '67603'),
    }
    got_unavailable = set(zip(
        unavailable['season'].astype(int), unavailable['game_pk'].astype(str), unavailable['starter_id'].astype(str)
    ))

    checks = {
        'step3_validation_pass': step3.get('status') == 'PASS',
        'step4_validation_pass': step4.get('status') == 'PASS',
        'eval_seasons_exactly_2017_2025': sorted(eval_all['season'].astype(int).unique().tolist()) == list(range(2017, 2026)),
        'eval_games_exactly_129': int(eval_all['game_pk'].nunique()) == 129,
        'eval_rows_exactly_258': len(eval_all) == 258,
        'eligible_rows_exactly_256': len(eligible) == 256,
        'unavailable_rows_exactly_2': len(unavailable) == 2,
        'unavailable_identities_exact': got_unavailable == expected_unavailable,
        'all_eval_rows_have_target_shift': bool(eval_all['historical_ps_mean_ip_shift'].notna().all()),
        'candidate_formula_exact': bool(np.allclose(
            eligible['candidate_ps_eip'].to_numpy(),
            eligible['rs_starter_avg_ip'].to_numpy() + eligible['historical_ps_mean_ip_shift'].to_numpy(),
            rtol=0, atol=1e-12
        )),
        'baseline_formula_exact': bool(np.allclose(
            eligible['baseline_rs_eip'].to_numpy(), eligible['rs_starter_avg_ip'].to_numpy(), rtol=0, atol=1e-12
        )),
        'prediction_artifact_retains_258_rows': len(pred) == 258,
        'nine_seasons_evaluated': len(season_df) == 9,
    }
    metrics['checks'] = checks
    if not all(checks.values()):
        metrics['status'] = 'FAIL'

    pred = pred.sort_values(['season','game_date','game_pk','home_away'])
    pred.to_csv(out / 'ps_expected_ip_walk_forward_predictions_2017_2025.csv', index=False)
    season_df.to_csv(out / 'ps_expected_ip_walk_forward_validation_by_season.csv', index=False)
    (out / 'ps_expected_ip_walk_forward_validation.json').write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(json.dumps(metrics, ensure_ascii=False), flush=True)
    if metrics['status'] != 'PASS':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
