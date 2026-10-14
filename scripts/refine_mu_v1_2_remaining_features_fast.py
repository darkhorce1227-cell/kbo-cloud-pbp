from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np

import refine_mu_v1_2_hierarchical_priors as h
import reverse_mu_v1_2_final_development as base

ROOT=Path(__file__).resolve().parents[1]


def merr(a, feat):
    a=np.asarray(a,float); fm=base.FROZEN[feat]['mean']; fs=base.FROZEN[feat]['scale']
    m=float(np.nanmean(a)); s=float(np.nanstd(a))
    return {'mean':m,'scale':s,'frozen_mean':fm,'frozen_scale':fs,'loss':abs(m-fm)/fs+abs(s-fs)/fs}

def one_score(a,feat):
    d=merr(a,feat); return d['loss'],d

def two_score(a,fa,b,fb):
    da=merr(a,fa); db=merr(b,fb); return da['loss']+db['loss'],{fa:da,fb:db}

def grid_single(vals, fn, feat):
    z=[]
    for p in vals:
        arr=fn(float(p)); loss,d=one_score(arr,feat); z.append({'x':float(p),'loss':loss,'detail':{feat:d}})
    return sorted(z,key=lambda x:x['loss'])[:20]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--training-dir',default='out_training'); ap.add_argument('--out',default='state/mu_v1_2_remaining_feature_refine_fast.json'); z=ap.parse_args()
    raw=h.prep(ROOT/z.training_dir); assert len(raw)==2628
    def A(c): return raw[c].to_numpy(float)
    off={}
    for feat,ph,recent in [('off_f5_long_ratio','f5',False),('off_f5_30d_ratio','f5',True),('off_late_long_ratio','late',False),('off_late_30d_ratio','late',True)]:
        if recent: s=A(f'{ph}_sum_30'); n=A('team_n_30'); lg=A(f'lg_{ph}_30')
        else: s=A(f'{ph}_sum_long'); n=A('team_n_long'); lg=A(f'lg_{ph}_long')
        off[feat]=grid_single(np.arange(0,30.001,.1),lambda p,s=s,n=n,lg=lg:(s+p*lg)/(n+p)/lg,feat)
    league=np.divide(A('cP_ip'),A('cP_n'),out=np.full(len(raw),5.0),where=A('cP_n')>0)
    ip=[]
    for pl in np.arange(3.5,6.501,.05):
        lv=(A('cL_ip')+pl*league)/(A('cL_n')+pl)
        for pr in np.arange(3.5,6.501,.05):
            rv=(A('c5_ip')+pr*lv)/(A('c5_n')+pr)
            loss,d=two_score(lv,'sp_ip_long',rv,'sp_ip_recent5'); ip.append({'p_long':float(pl),'p_recent5':float(pr),'loss':loss,'detail':d})
    ip=sorted(ip,key=lambda x:x['loss'])[:20]
    lg=A('lg_full_long'); n=A('park_games'); runs=A('park_runs')
    park=grid_single(np.arange(3,7.001,.05),lambda p:(runs+2*p*lg)/(2*(n+p))/lg,'park_factor')
    bp=[]
    for scope,prx in [('continuous','bc'),('season_reset','by')]:
        tr=A(f'{prx}L_tr'); to=A(f'{prx}L_to'); lr=A(f'{prx}L_lr'); lo=A(f'{prx}L_lo'); lgL=np.divide(lr*27,lo,out=np.full(len(raw),4.5),where=lo>0)
        tr30=A('bc30_tr'); to30=A('bc30_to'); lr30=A('bc30_lr'); lo30=A('bc30_lo'); lg30=np.divide(lr30*27,lo30,out=lgL.copy(),where=lo30>0)
        for pl in range(240,551,5):
            teamL=(tr+pl*lgL/27)*27/(to+pl)
            fL=teamL/lgL
            for p30 in range(0,91,3):
                team30=np.divide((tr30+p30*teamL/27)*27,to30+p30,out=teamL.copy(),where=(to30+p30)>0)
                for mode in ['league30','league_long','team_long']:
                    den={'league30':lg30,'league_long':lgL,'team_long':teamL}[mode]; f30=team30/den
                    loss,d=two_score(fL,'opp_bp_ra9_long_rel',f30,'opp_bp_ra9_30d_rel'); bp.append({'scope':scope,'denom30':mode,'p_long':float(pl),'p_30d':float(p30),'loss':loss,'detail':d})
    bp=sorted(bp,key=lambda x:x['loss'])[:30]
    payload={'version':'mu-v1.2-remaining-feature-refine-fast-v0.1','rows':len(raw),'official_shadow_allowed':False,
             'best':{'offense_each':{k:v[0] for k,v in off.items()},'starter_ip':ip[0],'park':park[0],'bullpen':bp[0]},
             'top':{'offense_each':off,'starter_ip':ip,'park':park,'bullpen':bp},
             'confirmed_exact':{'starter_rates':'all PA; K-BB-HBP; long->league 150 PA; recent5->shrunk long 150 PA'}}
    out=ROOT/z.out; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(payload,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
