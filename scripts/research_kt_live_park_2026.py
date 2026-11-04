#!/usr/bin/env python3
"""Experimental live-at-first-bridge Park appearance prior, not a pregame/Champion input."""
import json,os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
src=Path(os.environ.get('KT_BRIDGE_CSV','/mnt/data/kt_bridge/extracted/results/kt_144_game_bridge.csv'))
out=Path(os.environ.get('KT_PARK_RESULT','/mnt/data/kt_bridge/research/kt_live_park_conditioned_2026.json'))
df=pd.read_csv(src)
df=df[df.valid.astype(bool)&df.bridge_used.astype(bool)].copy()
df.date=pd.to_datetime(df.date)
assert len(df)==142
assert df.game_pk.is_unique
df['score_state']=np.select([df.bridge_entry_lead<=0,df.bridge_entry_lead.between(1,3),df.bridge_entry_lead>=4],['NOT_LEADING','LEAD_1_TO_3','LEAD_4PLUS'],default='UNKNOWN')
assert 'UNKNOWN' not in set(df.score_state)
pre=df[df.date<'2026-07-16'].copy();post=df[df.date>='2026-07-16'].copy()
assert len(pre)==82 and len(post)==60
p0=float((pre.park_used.sum()+1)/(len(pre)+2))
calibration=8.
group=pre.groupby('score_state').park_used.agg(['sum','count'])
p_by={str(k):float((v['sum']+calibration*p0)/(v['count']+calibration)) for k,v in group.iterrows()}
y=post.park_used.astype(int).to_numpy()
p=np.array([p_by[b] for b in post.score_state]);baseline=np.full(len(post),p0)
np.clip(p,1e-6,1-1e-6,out=p)
def brier(y,p):return float(np.mean((p-y)**2))
def ll(y,p):return float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p)))
rawdiff=(p-y)**2-(baseline-y)**2
rawll=(-y*np.log(p)-(1-y)*np.log1p(-p))-(-y*np.log(baseline)-(1-y)*np.log1p(-baseline))
rng=np.random.default_rng(20261104);indices=rng.integers(0,len(post),size=(12000,len(post)))
lead=df.score_state.eq('LEAD_1_TO_3')
a=int(df.loc[lead,'park_used'].sum());b=int(lead.sum()-a)
c=int(df.loc[~lead,'park_used'].sum());d=int((~lead).sum()-c)
res={
 'status':'EXPLORATORY_LIVE_AT_BRIDGE_NOT_PREGAME_NOT_CHAMPION',
 'training':{'n':len(pre),'end':'2026-07-15','base_park_use_probability':p0,'smoothed_conditional_probability':p_by,'pseudo_observations_per_score_bin':calibration},
 'holdout':{'n':len(post),'start':'2026-07-16','baseline_brier':brier(y,baseline),'state_conditional_brier':brier(y,p),'delta_brier':float(rawdiff.mean()),'delta_brier_ci95':[float(v) for v in np.quantile(rawdiff[indices].mean(1),[.025,.975])],
 'baseline_logloss':ll(y,baseline),'state_conditional_logloss':ll(y,p),'delta_logloss':float(rawll.mean()),'delta_logloss_ci95':[float(v) for v in np.quantile(rawll[indices].mean(1),[.025,.975])]},
 'all_states':{s:{'n':int(len(z)),'park_used':int(z.park_used.sum()),'rate':float(z.park_used.mean())} for s,z in df.groupby('score_state')},
 'holdout_states':{s:{'n':int(len(z)),'park_used':int(z.park_used.sum()),'rate':float(z.park_used.mean())} for s,z in post.groupby('score_state')},
 'association':{'fisher_oddsratio_lead1to3_vs_other':float(fisher_exact([[a,b],[c,d]]).statistic),'fisher_p':float(fisher_exact([[a,b],[c,d]]).pvalue),'table':[[a,b],[c,d]]},
 'data_exclusions':['20260604LGKT02026 role attribution ambiguous','2026-07-16+ one KT game with no bridge entered'],
 'limitations':['Score at actual bridge entry is observable in-game only, not pregame; cannot use it in CLEAN PRE.','Park availability/roster and inning not modeled; 2026 result known during exploratory choice.','Brier and log-loss improvement CIs include zero; not sufficient for production promotion.','First-bridge entrance score-state depends on manager decisions and actual starter usage; no causal interpretation.']}
out.parent.mkdir(exist_ok=True,parents=True)
out.write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n')
print('KT_PARK_LIVE_RESULT='+json.dumps(res,ensure_ascii=False))