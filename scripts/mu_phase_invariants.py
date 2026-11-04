"""Pure physical consistency gate for paired F5/Full KBO forecast means.

This is an independent validation helper, not a score model or Champion coefficient.
"""
from __future__ import annotations
import math
from typing import Mapping

_KEYS=('f5_away','f5_home','full_away','full_home','f5_total','full_total')

def phase_issues(mu: Mapping[str,float], tolerance: float = 1e-7) -> list[str]:
    issues=[]
    if not all(k in mu for k in _KEYS):
        return ['MISSING_FIELDS:'+','.join(k for k in _KEYS if k not in mu)]
    try:
        x={k:float(mu[k]) for k in _KEYS}
    except (ValueError,TypeError):
        return ['NON_NUMERIC_FIELD']
    if not all(math.isfinite(v) and v>=0 for v in x.values()):
        issues.append('NONFINITE_OR_NEGATIVE_MEAN')
    if not all(math.isfinite(v) for v in x.values()):
        return issues
    if abs(x['f5_away']+x['f5_home']-x['f5_total'])>tolerance:
        issues.append('F5_TOTAL_MARGINAL_MISMATCH')
    if abs(x['full_away']+x['full_home']-x['full_total'])>tolerance:
        issues.append('FULL_TOTAL_MARGINAL_MISMATCH')
    for side in ('away','home'):
        if x['full_'+side]+tolerance<x['f5_'+side]:
            issues.append('IMPOSSIBLE_NEGATIVE_6PLUS_RUNS_'+side.upper())
    if x['full_total']+tolerance<x['f5_total']:
        issues.append('IMPOSSIBLE_NEGATIVE_TOTAL_6PLUS_RUNS')
    return issues

def assert_phase_consistent(mu: Mapping[str,float]) -> None:
    issues=phase_issues(mu)
    if issues:
        raise ValueError('DISTRIBUTION_INCONSISTENT: '+';'.join(issues))
