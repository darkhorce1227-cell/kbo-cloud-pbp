from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np

import refine_mu_v1_2_hierarchical_priors as h
import reverse_mu_v1_2_final_development as base

ROOT=Path(__file__).resolve().parents[1]


def best1(vals, fn):
    out=[]
    for x in vals:
        feat=fn(float(x)); loss,detail=base.score(feat); out.append({'x':float(x),'loss':loss,'detail':detail})
    return sorted(out,key=lambda z:z['loss'])[:20]

def best2(A,B,fn):
    out=[]
    for a in A:
      for b in B:
        feat=fn(float(a),float(b)); loss,detail=base.score(feat); out.append({'a':float(a),'b':float(b),'loss':loss,'detail':detail})
    return sorted(out,key=lambda z:z['loss'])[:20]

def single_off(raw, feature, p):
    ph='f5' if 'f5' in feature else 'late'; recent='30d' in feature; vals=[]
    for r in raw.itertuples(index=False):
        if recent:
            lg=getattr(r,f'lg_{ph}_30'); s=getattr(r,f'{ph}_sum_30'); n=r.team_n_30
        else:
            lg=getattr(r,f'lg_{ph}_long'); s=getattr(r,f'{ph}_sum_long'); n=r.team_n_long
        rate=base.shrink(s,n,lg,p); vals.append(rate/lg if lg else 1.)
    return {feature:np.asarray(vals)}
def ip_feat(raw,pl,pr):
    return h.sp_ip_hier(raw,pl,pr,'shrunk_long')
def park_feat(raw,p): return base.park(raw,p)

def bp_variant(raw, scope, p_long, p30, denom_mode):
    valsL=[];vals30=[]; pr='by' if scope=='season_reset' else 'bc'
    for r in raw.itertuples(index=False):
        tr=getattr(r,f'{pr}L_tr'); to=getattr(r,f'{pr}L_to'); lr=getattr(r,f'{pr}L_lr'); lo=getattr(r,f'{pr}L_lo')
        lgL=lr*27/lo if lo else 4.5; teamL=(tr+p_long*lgL/27)*27/(to+p_long) if to+p_long>0 else lgL
        tr30=r.bc30_tr; to30=r.bc30_to; lr30=r.bc30_lr; lo30=r.bc30_lo; lg30=lr30*27/lo30 if lo30 else lgL
        team30=(tr30+p30*teamL/27)*27/(to30+p30) if to30+p30>0 else teamL
        valsL.append(teamL/lgL if lgL else 1.)
        if denom_mode=='league30': d=lg30
        elif denom_mode=='league_long': d=lgL
        elif denom_mode=='team_long': d=teamL
        else: raise ValueError(denom_mode)
        vals30.append(team30/d if d else 1.)
    return {'opp_bp_ra9_long_rel':np.asarray(valsL),'opp_bp_ra9_30d_rel':np.asarray(vals30)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--training-dir',default='out_training'); ap.add_argument('--out',default='state/mu_v1_2_remaining_feature_refine.json'); z=ap.parse_args()
    raw=h.prep(ROOT/z.training_dir); assert len(raw)==2628
    fine=np.arange(0,30.001,.1)
    off={f:best1(fine,lambda p,ff=f:single_off(raw,ff,p)) for f in ['off_f5_long_ratio','off_f5_30d_ratio','off_late_long_ratio','off_late_30d_ratio']}
    ip=best2(np.arange(3.5,6.501,.05),np.arange(3.5,6.501,.05),lambda a,b:ip_feat(raw,a,b))
    park=best1(np.arange(3,7.001,.05),lambda p:park_feat(raw,p))
    bp=[]
    for scope in ['continuous','season_reset']:
      for denom in ['league30','league_long','team_long']:
       for pl in range(240,551,5):
        for p30 in range(0,91,3):
          feat=bp_variant(raw,scope,float(pl),float(p30),denom); loss,detail=base.score(feat); bp.append({'scope':scope,'denom30':denom,'p_long':float(pl),'p_30d':float(p30),'loss':loss,'detail':detail})
    bp=sorted(bp,key=lambda x:x['loss'])[:30]
    payload={'version':'mu-v1.2-remaining-feature-refine-v0.1','rows':len(raw),'official_shadow_allowed':False,
             'best':{'offense_each':{k:v[0] for k,v in off.items()},'starter_ip':ip[0],'park':park[0],'bullpen':bp[0]},
             'top':{'offense_each':off,'starter_ip':ip,'park':park,'bullpen':bp},
             'confirmed_exact':{'starter_rates':'all PA; K-BB-HBP; long prior=league at 150 PA; recent5 prior=shrunk long at 150 PA'},
             'note':'Fine diagnostic for remaining features; do not infer production rules from non-natural decimal optima without semantic support.'}
    out=ROOT/z.out; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(payload,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
