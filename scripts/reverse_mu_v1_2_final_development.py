from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

import build_mu_v1_2_features as mu
from run_mu_v1_2_pipeline import prepare_pa_fixed
from reverse_mu_v1_2_preprocessing import FROZEN, TRAIN_START
from diagnose_mu_v1_2_sp_bp_definitions import starter_game_table

ROOT=Path(__file__).resolve().parents[1]
SHA24="d9e4b59fc4c255990452c47d384977f689f338ce6b90988933906cb9e7874458"
SHA25="017ae5f39e79b2aeaa05e269cf32aebe7cc410c75fedd6e928850a61730acb61"


def moment_error(vals, feat):
    a=np.asarray(vals,float); a=a[np.isfinite(a)]
    m=float(a.mean()); s=float(a.std(ddof=0)); fm=float(FROZEN[feat]['mean']); fs=float(FROZEN[feat]['scale'])
    return {'mean':m,'scale':s,'frozen_mean':fm,'frozen_scale':fs,
            'z_mean_abs_error':abs(m-fm)/max(fs,1e-12),
            'relative_scale_error':abs(s-fs)/max(fs,1e-12),
            'loss':abs(m-fm)/max(fs,1e-12)+abs(s-fs)/max(fs,1e-12)}

def score(features):
    d={k:moment_error(v,k) for k,v in features.items()}; return float(sum(x['loss'] for x in d.values())),d

def shrink(sum_x,n,prior,p):
    return (sum_x+p*prior)/(n+p) if n+p>0 else prior

def dev_games(tg):
    d=tg.game_date
    return tg[((d.dt.year==2024)&(d>=TRAIN_START))|(d.dt.year==2025)].sort_values(['game_date','game_pk','is_home']).reset_index(drop=True)


def build_sufficient(tg,stc,sta,bp):
    smap=stc.set_index(['game_pk','pitching_team'])['starter'].to_dict()
    ipmap=stc.set_index(['game_pk','pitching_team','starter'])['ip'].to_dict()
    sta=sta.copy(); sta['ip']=[ipmap.get((r.game_pk,r.pitching_team,r.starter),np.nan) for r in sta.itertuples(index=False)]
    rows=[]
    for r in dev_games(tg).itertuples(index=False):
        d=pd.Timestamp(r.game_date)
        pre=tg[tg.game_date<d]; p30=pre[pre.game_date>=d-pd.Timedelta(days=30)]
        if p30.empty: p30=pre
        my=pre[pre.team.astype(str).eq(str(r.team))]; my30=p30[p30.team.astype(str).eq(str(r.team))]
        if my.empty: my=pre
        if my30.empty: my30=my
        lg={
          'f5_long':float(pre.runs_f5.mean()), 'f5_30':float(p30.runs_f5.mean()),
          'late_long':float(pre.runs_late.mean()), 'late_30':float(p30.runs_late.mean()),
          'full_long':float(pre.runs_full.mean())
        }
        sid=smap.get((r.game_pk,r.opponent))
        sc=stc[stc.game_date<d]; sa=sta[sta.game_date<d]
        pc=sc[sc.starter.astype(str).eq(str(sid))].sort_values('game_date') if sid is not None else sc.iloc[0:0]
        pa=sa[sa.starter.astype(str).eq(str(sid))].sort_values('game_date') if sid is not None else sa.iloc[0:0]
        def ss(x,den):
            return {'den':float(x[den].sum()),'k':float(x.k.sum()),'bb':float(x.bb.sum()),'hbp':float(x.hbp.sum()),'hr':float(x.hr.sum()),'h':float(x.h.sum()),'n':float(len(x)),'ip':float(x.ip.sum())}
        def sprior(x,den):
            dd=float(x[den].sum()); nn=float(len(x))
            return {'den':dd,'k':float(x.k.sum()),'bb':float(x.bb.sum()),'hbp':float(x.hbp.sum()),'hr':float(x.hr.sum()),'h':float(x.h.sum()),'n':nn,'ip':float(x.ip.sum())}
        cL,c5,cP=ss(pc,'pa'),ss(pc.tail(5),'pa'),sprior(sc,'pa')
        aL,a5,aP=ss(pa,'pa_all'),ss(pa.tail(5),'pa_all'),sprior(sa,'pa_all')
        bpre=bp[bp.game_date<d]; b30=bpre[bpre.game_date>=d-pd.Timedelta(days=30)]
        if b30.empty: b30=bpre
        ypre=bpre[bpre.game_date.dt.year==d.year]; y30=b30
        if ypre.empty: ypre=bpre
        def bs(pool):
            team=pool[pool.pitching_team.astype(str).eq(str(r.opponent))]
            return float(team.bp_runs.sum()),float(team.bp_outs.sum()),float(pool.bp_runs.sum()),float(pool.bp_outs.sum())
        bcL=bs(bpre); bc30=bs(b30); byL=bs(ypre); by30=bs(y30)
        home=r.team if int(r.is_home) else r.opponent
        pgids=pre[(pre.team.astype(str).eq(str(home)))&(pre.is_home==1)].game_pk.unique(); pg=pre[pre.game_pk.isin(pgids)]
        rows.append({
          'game_pk':r.game_pk,'game_date':str(d.date()),'team':r.team,'is_home':float(r.is_home),
          'team_n_long':float(len(my)),'team_n_30':float(len(my30)),
          'f5_sum_long':float(my.runs_f5.sum()),'f5_sum_30':float(my30.runs_f5.sum()),
          'late_sum_long':float(my.runs_late.sum()),'late_sum_30':float(my30.runs_late.sum()),
          **{f'lg_{k}':v for k,v in lg.items()},
          **{f'cL_{k}':v for k,v in cL.items()},**{f'c5_{k}':v for k,v in c5.items()},**{f'cP_{k}':v for k,v in cP.items()},
          **{f'aL_{k}':v for k,v in aL.items()},**{f'a5_{k}':v for k,v in a5.items()},**{f'aP_{k}':v for k,v in aP.items()},
          'bcL_tr':bcL[0],'bcL_to':bcL[1],'bcL_lr':bcL[2],'bcL_lo':bcL[3],
          'bc30_tr':bc30[0],'bc30_to':bc30[1],'bc30_lr':bc30[2],'bc30_lo':bc30[3],
          'byL_tr':byL[0],'byL_to':byL[1],'byL_lr':byL[2],'byL_lo':byL[3],
          'by30_tr':by30[0],'by30_to':by30[1],'by30_lr':by30[2],'by30_lo':by30[3],
          'park_runs':float(pg.runs_full.sum()),'park_games':float(len(pgids))
        })
    return pd.DataFrame(rows)


def offense(raw,pl,p30):
    out={}
    for ph in ['f5','late']:
      vl=[]; v30=[]
      for r in raw.itertuples(index=False):
        ll=getattr(r,f'lg_{ph}_long'); l30=getattr(r,f'lg_{ph}_30')
        ml=shrink(getattr(r,f'{ph}_sum_long'),r.team_n_long,ll,pl); m30=shrink(getattr(r,f'{ph}_sum_30'),r.team_n_30,l30,p30)
        vl.append(ml/ll if ll else 1.0); v30.append(m30/l30 if l30 else 1.0)
      out[f'off_{ph}_long_ratio']=np.asarray(vl); out[f'off_{ph}_30d_ratio']=np.asarray(v30)
    return out


def starter_rates(raw,mode,pl,p5):
    pref='a' if mode.startswith('all') else 'c'; use_hbp=mode=='all_pa_hbp'
    out={k:[] for k in ['sp_kbb_long','sp_kbb_recent5','sp_hr_long','sp_hr_recent5','sp_h_long','sp_h_recent5']}
    for r in raw.itertuples(index=False):
      for tag,p,suf in [('L',pl,'long'),('5',p5,'recent5')]:
        den=getattr(r,f'{pref}{tag}_den'); pden=getattr(r,f'{pref}P_den')
        hbp=getattr(r,f'{pref}{tag}_hbp') if use_hbp else 0.; phbp=getattr(r,f'{pref}P_hbp') if use_hbp else 0.
        nums={'kbb':getattr(r,f'{pref}{tag}_k')-getattr(r,f'{pref}{tag}_bb')-hbp,'hr':getattr(r,f'{pref}{tag}_hr'),'h':getattr(r,f'{pref}{tag}_h')}
        pri={'kbb':(getattr(r,f'{pref}P_k')-getattr(r,f'{pref}P_bb')-phbp)/pden if pden else 0.,'hr':getattr(r,f'{pref}P_hr')/pden if pden else 0.,'h':getattr(r,f'{pref}P_h')/pden if pden else 0.}
        for stat in ['kbb','hr','h']:
          out[f'sp_{stat}_{suf}'].append(shrink(nums[stat],den,pri[stat],p))
    return {k:np.asarray(v) for k,v in out.items()}


def starter_ip(raw,pl,p5):
    a=[];b=[]
    for r in raw.itertuples(index=False):
      prior=r.cP_ip/r.cP_n if r.cP_n else 5.0
      a.append(shrink(r.cL_ip,r.cL_n,prior,pl)); b.append(shrink(r.c5_ip,r.c5_n,prior,p5))
    return {'sp_ip_long':np.asarray(a),'sp_ip_recent5':np.asarray(b)}


def bullpen(raw,scope,pl,p30):
    a=[];b=[]; pr='bc' if scope=='continuous' else 'by'
    for r in raw.itertuples(index=False):
      def one(tag,p):
        tr=getattr(r,f'{pr}{tag}_tr'); to=getattr(r,f'{pr}{tag}_to'); lr=getattr(r,f'{pr}{tag}_lr'); lo=getattr(r,f'{pr}{tag}_lo')
        lg=lr*27/lo if lo else 4.5; val=(tr+p*lg/27)*27/(to+p) if to+p>0 else lg
        return val/lg if lg else 1.0
      a.append(one('L',pl)); b.append(one('30',p30))
    return {'opp_bp_ra9_long_rel':np.asarray(a),'opp_bp_ra9_30d_rel':np.asarray(b)}


def park(raw,p):
    v=[]
    for r in raw.itertuples(index=False):
      lg=r.lg_full_long; n=r.park_games; x=(r.park_runs+2*p*lg)/(2*(n+p)) if n+p>0 else lg; v.append(x/lg if lg else 1.)
    return {'park_factor':np.asarray(v)}

def grid2(A,B,fn):
    top=[]
    for a in A:
      for b in B:
        loss,detail=score(fn(float(a),float(b))); top.append({'a':float(a),'b':float(b),'loss':loss,'detail':detail})
    return sorted(top,key=lambda x:x['loss'])[:12]
def grid1(A,fn):
    top=[]
    for a in A:
      loss,detail=score(fn(float(a))); top.append({'x':float(a),'loss':loss,'detail':detail})
    return sorted(top,key=lambda x:x['loss'])[:12]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--training-dir',default='out_training'); ap.add_argument('--out',default='state/mu_v1_2_final_development_reverse.json'); z=ap.parse_args()
    p24=ROOT/z.training_dir/'kbo_pbp_2024_training.csv.gz'; p25=ROOT/z.training_dir/'kbo_pbp_2025_training.csv.gz'
    assert hashlib.sha256(p24.read_bytes()).hexdigest()==SHA24 and hashlib.sha256(p25.read_bytes()).hexdigest()==SHA25
    df=pd.concat([pd.read_csv(p24,low_memory=False),pd.read_csv(p25,low_memory=False)],ignore_index=True); pa=prepare_pa_fixed(df); tg,stc,bp=mu.game_tables(pa); sta=starter_game_table(pa)
    raw=build_sufficient(tg,stc,sta,bp); assert len(raw)==2628
    off=grid2([0,1,2,3,4,5,7,10,12,15,20,25,30,40,50],[0,1,2,3,4,5,7,10,12,15,20],lambda a,b:offense(raw,a,b))
    spr={}
    for mode in ['classified_pa','all_pa','all_pa_hbp']:
      spr[mode]=grid2([0,25,50,75,100,125,150,200,250,300,400],[0,20,40,60,80,100,125,150,200],lambda a,b,m=mode:starter_rates(raw,m,a,b))
    spi=grid2([0,.5,1,1.5,2,3,4,5,7,10,15,20,30],[0,.5,1,1.5,2,3,4,5,7,10,15],lambda a,b:starter_ip(raw,a,b))
    bpres={}
    for sc in ['continuous','season_reset']:
      bpres[sc]=grid2([0,9,18,27,36,54,81,108,135,180,270],[0,9,18,27,36,54,81,108,135],lambda a,b,s=sc:bullpen(raw,s,a,b))
    pk=grid1([0,1,2,3,4,5,7,10,12,15,20,25,30],lambda a:park(raw,a))
    payload={'version':'mu-v1.2-final-development-reverse-v0.1','rows':len(raw),'official_shadow_allowed':False,
             'best':{'offense':off[0],'starter_rates':{k:v[0] for k,v in spr.items()},'starter_ip':spi[0],'bullpen':{k:v[0] for k,v in bpres.items()},'park':pk[0]},
             'top':{'offense':off,'starter_rates':spr,'starter_ip':spi,'bullpen':bpres,'park':pk},
             'note':'Reverse-engineering diagnostic on exact 2628-row final-development sample. No promotion without semantic confirmation and parity audit.'}
    out=ROOT/z.out; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(payload,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
