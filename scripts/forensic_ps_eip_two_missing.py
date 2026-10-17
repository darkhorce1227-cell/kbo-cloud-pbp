#!/usr/bin/env python3
"""Forensic audit of the two 2017-2025 PS starters with zero same-season RS starts.

Goals:
1) Infer the two missing baseline-error magnitudes forced by the old Phase-2
   aggregate RS baseline RMSE + MAE (258 claimed rows).
2) Enumerate assignment/sign combinations and rebuild the full 258-row
   leakage-free walk-forward candidate to see which combination best matches
   the old candidate RMSE/MAE/MSE delta and frozen 2026 mean shift.
3) Compare inferred baseline values with simple reproducible fallback hypotheses
   from the audited RS starter table (team/league starter means/medians, PS-team
   peer RS averages) plus documented all-appearance means for the two pitchers.

This is forensic only. It MUST NOT be used as a live fallback rule.
"""
import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def rmse(x):
    a = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(a * a)))


def solve_two_abs(sum_abs, sum_sq):
    # u+v=S, u^2+v^2=Q -> uv=(S^2-Q)/2
    prod = (sum_abs * sum_abs - sum_sq) / 2.0
    disc = sum_abs * sum_abs - 4.0 * prod
    if disc < -1e-10:
        raise ValueError(f'impossible magnitude system: S={sum_abs} Q={sum_sq} disc={disc}')
    disc = max(disc, 0.0)
    root = math.sqrt(disc)
    return sorted([(sum_abs + root) / 2.0, (sum_abs - root) / 2.0], reverse=True)


def build_258(joined, inferred_rows):
    x = joined[joined['season'].between(2017, 2025)].copy()
    x['forensic_baseline_ip'] = x['rs_starter_avg_ip']
    x['forensic_delta'] = x['delta_ps_minus_rs_ip']

    by_key = {(int(r['season']), str(r['game_pk']), str(r['starter_id'])): r for r in inferred_rows}
    for idx, row in x[x['rs_link_status'] != 'MATCHED'].iterrows():
        key = (int(row['season']), str(row['game_pk']), str(row['starter_id']))
        inf = by_key[key]
        x.at[idx, 'forensic_baseline_ip'] = inf['baseline_ip']
        x.at[idx, 'forensic_delta'] = inf['residual_actual_minus_baseline']

    # Refit each target shift from all PS seasons < target, including the two
    # inferred residuals when they are historically available.
    full_hist = joined.copy()
    full_hist['forensic_delta'] = full_hist['delta_ps_minus_rs_ip']
    for idx, row in full_hist[full_hist['rs_link_status'] != 'MATCHED'].iterrows():
        key = (int(row['season']), str(row['game_pk']), str(row['starter_id']))
        if key in by_key:
            full_hist.at[idx, 'forensic_delta'] = by_key[key]['residual_actual_minus_baseline']

    shifts = {}
    for target in range(2017, 2027):
        h = full_hist[(full_hist['season'] < target) & full_hist['forensic_delta'].notna()]
        shifts[target] = float(h['forensic_delta'].mean())

    x['forensic_shift'] = x['season'].astype(int).map(shifts)
    x['baseline_error'] = x['forensic_baseline_ip'] - x['actual_ip']
    x['candidate_ip'] = x['forensic_baseline_ip'] + x['forensic_shift']
    x['candidate_error'] = x['candidate_ip'] - x['actual_ip']
    x['mse_delta'] = x['candidate_error'] ** 2 - x['baseline_error'] ** 2

    wins = 0
    season_metrics = []
    for s, g in x.groupby('season'):
        b = rmse(g['baseline_error'])
        c = rmse(g['candidate_error'])
        wins += int(c < b)
        season_metrics.append({
            'season': int(s),
            'baseline_rmse': b,
            'candidate_rmse': c,
            'candidate_beats_baseline': bool(c < b),
            'forensic_shift': float(g['forensic_shift'].iloc[0]),
        })

    return {
        'baseline_rmse': rmse(x['baseline_error']),
        'baseline_mae': float(x['baseline_error'].abs().mean()),
        'candidate_rmse': rmse(x['candidate_error']),
        'candidate_mae': float(x['candidate_error'].abs().mean()),
        'mse_delta_candidate_minus_baseline': float(x['mse_delta'].mean()),
        'season_wins': int(wins),
        'shift_2026': float(shifts[2026]),
        'season_metrics': season_metrics,
    }


def natural_fallbacks(row, rs_apps, joined):
    season = int(row['season'])
    team = str(row['team'])
    rsy = rs_apps[rs_apps['season'] == season].copy()
    rst = rsy[rsy['team_code'] == team].copy()
    peers = joined[
        (joined['season'] == season)
        & (joined['team'] == team)
        & (joined['rs_link_status'] == 'MATCHED')
    ].copy()

    out = {
        'team_rs_starter_appearance_mean_ip': float(rst['actual_ip'].mean()),
        'team_rs_starter_appearance_median_ip': float(rst['actual_ip'].median()),
        'league_rs_starter_appearance_mean_ip': float(rsy['actual_ip'].mean()),
        'league_rs_starter_appearance_median_ip': float(rsy['actual_ip'].median()),
        'ps_team_peer_rs_avg_ip_appearance_weighted': float(peers['rs_starter_avg_ip'].mean()) if len(peers) else None,
        'ps_team_peer_rs_avg_ip_unique_pitcher': float(peers.drop_duplicates('starter_id')['rs_starter_avg_ip'].mean()) if len(peers) else None,
    }
    # Public record facts, used only as a rejected-hypothesis benchmark.
    if season == 2018 and str(row['starter_id']) == '68703':
        out['documented_all_appearance_mean_ip'] = (18.0 + 2.0/3.0) / 22.0
        out['documented_all_appearance_mean_source'] = 'KBO regular-season line: 22 G, 18 2/3 IP; contemporary report says all 22 were relief'
    elif season == 2022 and str(row['starter_id']) == '67603':
        out['documented_all_appearance_mean_ip'] = (50.0 + 1.0/3.0) / 53.0
        out['documented_all_appearance_mean_source'] = 'KBO regular-season line: 53 G, 50 1/3 IP; contemporary report says all 53 were relief'
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--joined-input', default='research/postseason_expected_ip/ps_rs_starter_avg_ip.csv')
    ap.add_argument('--step5-validation', default='research/postseason_expected_ip/ps_expected_ip_walk_forward_validation.json')
    ap.add_argument('--rs-appearances', default='historical_rs/rs_starter_appearances_2016_2025.csv')
    ap.add_argument('--frozen-reference', default='research/postseason_expected_ip/frozen_2026_prior.json')
    ap.add_argument('--output-dir', default='research/postseason_expected_ip')
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    joined = pd.read_csv(args.joined_input, dtype={'game_pk': str, 'starter_id': str})
    rs_apps = pd.read_csv(args.rs_appearances, dtype={'game_id': str, 'starter_pcode': str})
    step5 = json.loads(Path(args.step5_validation).read_text(encoding='utf-8'))
    frozen = json.loads(Path(args.frozen_reference).read_text(encoding='utf-8'))
    old = step5['old_phase2_reference_258_claimed']

    ev = joined[joined['season'].between(2017, 2025)].copy()
    known = ev[ev['rs_link_status'] == 'MATCHED'].copy()
    missing = ev[ev['rs_link_status'] != 'MATCHED'].copy().sort_values(['season','game_pk'])
    if len(known) != 256 or len(missing) != 2:
        raise ValueError(f'unexpected coverage known={len(known)} missing={len(missing)}')

    # Known baseline error = RS avg - actual = -delta.
    known_err = -known['delta_ps_minus_rs_ip'].astype(float)
    missing_abs_sum = float(old['starter_appearances'] * old['baseline_rs_ip_mae'] - known_err.abs().sum())
    missing_sq_sum = float(old['starter_appearances'] * (old['baseline_rs_ip_rmse'] ** 2) - (known_err ** 2).sum())
    mags = solve_two_abs(missing_abs_sum, missing_sq_sum)

    target = {
        'candidate_rmse': float(old['candidate_rmse']),
        'candidate_mae': float(old['candidate_mae']),
        'mse_delta_candidate_minus_baseline': float(old['mse_delta_candidate_minus_baseline']),
        'shift_2026': float(frozen['research_basis']['mean_shift_ip']),
        'season_wins': int(old['candidate_rmse_season_wins']),
    }

    missing_records = missing[['season','game_date','game_pk','team','starter_id','starter_name','actual_ip']].to_dict(orient='records')
    candidates = []
    # Two assignments of magnitudes x four sign pairs.
    for perm in set(itertools.permutations(mags, 2)):
        for signs in itertools.product([-1.0, 1.0], repeat=2):
            inferred = []
            for rec, mag, sign in zip(missing_records, perm, signs):
                residual = float(sign * mag)  # actual - baseline
                baseline = float(rec['actual_ip'] - residual)
                inferred.append({
                    **rec,
                    'residual_actual_minus_baseline': residual,
                    'baseline_ip': baseline,
                    'abs_residual': abs(residual),
                })
            m = build_258(joined, inferred)
            # Scale-free score; old values were rounded, so use small practical scales.
            score = (
                ((m['candidate_rmse'] - target['candidate_rmse']) / 0.005) ** 2
                + ((m['candidate_mae'] - target['candidate_mae']) / 0.005) ** 2
                + ((m['mse_delta_candidate_minus_baseline'] - target['mse_delta_candidate_minus_baseline']) / 0.005) ** 2
                + ((m['shift_2026'] - target['shift_2026']) / 0.002) ** 2
                + ((m['season_wins'] - target['season_wins']) / 1.0) ** 2
            )
            candidates.append({
                'assignment': inferred,
                'metrics': m,
                'delta_vs_old': {
                    'candidate_rmse': m['candidate_rmse'] - target['candidate_rmse'],
                    'candidate_mae': m['candidate_mae'] - target['candidate_mae'],
                    'mse_delta_candidate_minus_baseline': m['mse_delta_candidate_minus_baseline'] - target['mse_delta_candidate_minus_baseline'],
                    'shift_2026': m['shift_2026'] - target['shift_2026'],
                    'season_wins': m['season_wins'] - target['season_wins'],
                },
                'fit_score': float(score),
            })
    candidates.sort(key=lambda z: z['fit_score'])

    # Natural fallback hypotheses computed from audited RS starter appearances.
    natural = []
    best_assignment = candidates[0]['assignment']
    inferred_by_key = {(int(r['season']), str(r['game_pk']), str(r['starter_id'])): r for r in best_assignment}
    for _, row in missing.iterrows():
        key = (int(row['season']), str(row['game_pk']), str(row['starter_id']))
        inf = inferred_by_key[key]
        fb = natural_fallbacks(row, rs_apps, joined)
        vals = []
        for name, value in fb.items():
            if name.endswith('_source') or value is None:
                continue
            vals.append({
                'fallback': name,
                'baseline_ip': float(value),
                'distance_to_inferred_baseline_ip': float(abs(value - inf['baseline_ip'])),
                'implied_residual_actual_minus_baseline': float(row['actual_ip'] - value),
            })
        vals.sort(key=lambda q: q['distance_to_inferred_baseline_ip'])
        natural.append({
            'season': int(row['season']),
            'game_pk': str(row['game_pk']),
            'starter_id': str(row['starter_id']),
            'starter_name': row['starter_name'],
            'actual_ps_ip': float(row['actual_ip']),
            'best_inferred_baseline_ip': float(inf['baseline_ip']),
            'best_inferred_residual': float(inf['residual_actual_minus_baseline']),
            'fallbacks_ranked_by_distance': vals,
            'documented_all_appearance_mean_source': fb.get('documented_all_appearance_mean_source'),
        })

    best = candidates[0]
    second = candidates[1]
    both_negative = all(r['residual_actual_minus_baseline'] < 0 for r in best['assignment'])
    result = {
        'status': 'PASS',
        'forensic_scope': 'two zero-RS-start postseason starters only; no live imputation rule is authorized',
        'old_phase2_reference': target,
        'baseline_metric_inverse_solution': {
            'missing_abs_error_sum_implied_by_old_mae': missing_abs_sum,
            'missing_sq_error_sum_implied_by_old_rmse': missing_sq_sum,
            'two_abs_error_magnitudes_ip': mags,
            'note': 'Old aggregate RMSE+MAE determine the two absolute missing baseline errors up to assignment and sign, subject to reported rounding.'
        },
        'missing_rows': missing_records,
        'best_enumerated_solution': best,
        'second_best_enumerated_solution': second,
        'all_enumerated_candidates': candidates,
        'natural_fallback_comparison': natural,
        'checks': {
            'known_rows_256': len(known) == 256,
            'missing_rows_2': len(missing) == 2,
            'best_solution_both_residuals_negative': bool(both_negative),
            'best_solution_preserves_9_of_9_season_wins': best['metrics']['season_wins'] == 9,
            'baseline_rmse_reconstructs_old_to_reported_precision': abs(best['metrics']['baseline_rmse'] - old['baseline_rs_ip_rmse']) < 1e-10,
            'baseline_mae_reconstructs_old_to_reported_precision': abs(best['metrics']['baseline_mae'] - old['baseline_rs_ip_mae']) < 1e-10,
        },
        'interpretation': {
            'relief_mean_fallback_status': 'REJECT_IF_FAR_FROM_INFERRED_BASELINE',
            'team_or_league_simple_fallback_status': 'COMPARE_DISTANCES_ONLY; do not promote without original code evidence',
            'exact_original_fallback_identified': False,
            'why_not_exact': 'Original Phase-2 code/appearance table is absent at the fixed repository reference commit; aggregate metrics constrain but do not uniquely prove the generating fallback rule.'
        },
        'inputs': {
            'joined': args.joined_input,
            'step5_validation': args.step5_validation,
            'rs_appearances': args.rs_appearances,
            'frozen_reference': args.frozen_reference,
        }
    }

    (outdir / 'ps_expected_ip_two_missing_forensic.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    flat = []
    for rank, c in enumerate(candidates, 1):
        row = {
            'rank': rank,
            'fit_score': c['fit_score'],
            'baseline_rmse': c['metrics']['baseline_rmse'],
            'baseline_mae': c['metrics']['baseline_mae'],
            'candidate_rmse': c['metrics']['candidate_rmse'],
            'candidate_mae': c['metrics']['candidate_mae'],
            'mse_delta': c['metrics']['mse_delta_candidate_minus_baseline'],
            'shift_2026': c['metrics']['shift_2026'],
            'season_wins': c['metrics']['season_wins'],
        }
        for i, a in enumerate(c['assignment'], 1):
            row[f'missing{i}_season'] = a['season']
            row[f'missing{i}_starter_id'] = a['starter_id']
            row[f'missing{i}_starter_name'] = a['starter_name']
            row[f'missing{i}_baseline_ip'] = a['baseline_ip']
            row[f'missing{i}_residual_actual_minus_baseline'] = a['residual_actual_minus_baseline']
        flat.append(row)
    pd.DataFrame(flat).to_csv(outdir / 'ps_expected_ip_two_missing_forensic_candidates.csv', index=False)

    print(json.dumps({
        'status': result['status'],
        'magnitudes': mags,
        'best': best,
        'natural_fallback_comparison': natural,
        'checks': result['checks'],
    }, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
