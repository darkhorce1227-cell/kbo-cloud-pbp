"""2026-11-03 KS1 market-blind frozen research baseline. NOT Champion.
Frozen 2024-25 fitted mu-v1.3 coefficients, inputs as-of 2026-11-01.
Data limitation: 2026 only history; 2024-25 priors unavailable in this fast path.
No betting/market odds used or read.
"""
from pathlib import Path
import sys, json, hashlib
import pandas as pd
import numpy as np
sys.path.insert(0,str(Path.cwd()/"scripts"))
import mu_v1_3 as m
import build_mu_v1_2_features as base
from scipy.stats import nbinom

SRC=Path("/tmp/ks1/kbo_pbp_2026_latest.parquet")
STATUS=Path("state/status.json")
OUT=Path("/tmp/ks1/ks1_20261103_shadow.json")
source_hash=hashlib.sha256(SRC.read_bytes()).hexdigest()
status=json.loads(STATUS.read_text())
assert status['last_success_date']=='2026-11-01',status['last_success_date']
assert status['sha256_parquet']==source_hash,(source_hash,status['sha256_parquet'])
raw=pd.read_parquet(SRC).copy()
assert len(raw)==224290,len(raw)
raw['game_date']=pd.to_datetime(raw['game_date'],errors='raise')
assert raw['game_date'].max()<pd.Timestamp("2026-11-03")
def load_only_2026(last_year):
 assert last_year==2026
 return raw.copy()
base.load_seasons=load_only_2026
team,starts,bp=m.load_all(2026)
cfg=m.config()
model=json.loads(m.MODEL_PATH.read_text())
assert model['status']=="RESEARCH_CANDIDATE_NOT_CHAMPION"
cutoff=pd.Timestamp('2026-11-03')
away=m.build_features(team,starts,bp,d=cutoff,team='SS',opponent='KT',opposing_starter='56002',is_home=0,cfg=cfg)
home=m.build_features(team,starts,bp,d=cutoff,team='KT',opponent='SS',opposing_starter='65320',is_home=1,cfg=cfg)
mu=m.score(model,away,home)
for side in ('home','away'):
 assert mu['full_'+side]>mu['f5_'+side],mu

n=1_000_000
rng=np.random.default_rng(20261103)
def draw_nb(mean,alpha):
 return rng.negative_binomial(1/alpha,1/(1+alpha*mean),size=n).astype(np.int16)
fa=draw_nb(mu['f5_away'],0.54); fh=draw_nb(mu['f5_home'],0.54)
la=draw_nb(mu['full_away']-mu['f5_away'],0.322)
lh=draw_nb(mu['full_home']-mu['f5_home'],0.322)
away_full=fa+la;home_full=fh+lh
pre_extra_tie=(away_full==home_full).mean()
# Overtime 10–15 inning: simple two-sided Poisson per 9-inning late rate, with home walkoff approx
for inning in range(10,16):
 tied=np.flatnonzero(away_full==home_full)
 if not len(tied):break
 rna=rng.poisson((mu['full_away']-mu['f5_away'])/4,len(tied)).astype(np.int16)
 rnh=rng.poisson((mu['full_home']-mu['f5_home'])/4,len(tied)).astype(np.int16)
 away_full[tied]+=rna;home_full[tied]+=rnh
f5margin=fh-fa
fullmargin=home_full-away_full
total_f5=fh+fa
total_full=home_full+away_full
p=lambda x:float(np.mean(x))
f5_wdl={'KT_win':p(f5margin>0),'draw':p(f5margin==0),'Samsung_win':p(f5margin<0)}
full_wdl={'KT_win':p(fullmargin>0),'draw_after_15':p(fullmargin==0),'Samsung_win':p(fullmargin<0),'tie_after_9':float(pre_extra_tie)}
totals={'F5':{str(k+.5):{'under':p(total_f5<=k),'over':p(total_f5>k)} for k in range(2,15)},
'Full':{str(k+.5):{'under':p(total_full<=k),'over':p(total_full>k)} for k in range(3,20)}}
full_hcap={str(k+.5):{'KT_minus':p(fullmargin>k+.5),'Samsung_plus':p(fullmargin<k+.5)} for k in (0,1,2,3)}
f5_hcap={str(k+.5):{'KT_minus':p(f5margin>k+.5),'Samsung_plus':p(f5margin<k+.5)} for k in (0,1,2,3)}
win1loss={'KT_by_2plus':p(fullmargin>=2),'within_1_or_tie':p(abs(fullmargin)<=1),'Samsung_by_2plus':p(fullmargin<=-2)}
margin={'KT':{str(k):p(fullmargin==k) for k in range(1,7)},'Samsung':{str(k):p(fullmargin==-k) for k in range(1,7)}}
margin['KT']['7+']=p(fullmargin>=7);margin['Samsung']['7+']=p(fullmargin<=-7)
double={}
for side,sign in [('KT',1),('draw',0),('Samsung',-1)]:
 for side_full,signfull in [('KT',1),('Samsung',-1),('draw',0)]:
  e=(f5margin>0 if sign==1 else f5margin<0 if sign==-1 else f5margin==0)&(fullmargin>0 if signfull==1 else fullmargin<0 if signfull==-1 else fullmargin==0)
  double[side+'_'+side_full]=p(e)
exact={str(i):p(total_full==i) for i in range(20)}
exact['20+']=p(total_full>=20)
stats={}
for code in ['KT','SS']:
 g=team[(team.team==code)&(team.game_date<cutoff)]
 g30=g[g.game_date>=cutoff-pd.Timedelta(days=30)]
 stats[code]={'games':len(g),'avg_runs_full':round(float(g.runs_full.mean()),3),'avg_runs_f5':round(float(g.runs_f5.mean()),3),'recent30_games':len(g30),'recent30_runs':round(float(g30.runs_full.mean()),3) if len(g30) else None}
for pid in ['56002','65320']:
 sp=starts[(starts.starter.astype(str)==pid)&(starts.game_date<cutoff)]
 stats[pid]={'starts':len(sp),'ip_mean':round(float(sp.ip.mean()),3) if len(sp) else None,'k_per_pa':round(float(sp.k.sum()/sp.pa.sum()),4) if sp.pa.sum() else None,'bb_per_pa':round(float(sp.bb.sum()/sp.pa.sum()),4) if sp.pa.sum() else None,'hr_per_pa':round(float(sp.hr.sum()/sp.pa.sum()),4) if sp.pa.sum() else None,'last_date':str(sp.game_date.max().date()) if len(sp) else None,'recent5_ip':round(float(sp.sort_values('game_date').tail(5).ip.mean()),3) if len(sp) else None}
ans={
 'game':{'date':'2026-11-03','game_pk':'77771103SSKT02026','venue':'Suwon KT Wiz Park','away':'Samsung (SS)','home':'KT','starter_away':'Choi Won-tae 65320','starter_home':'Davis Daniel 56002'},
 'created_with_market_unseen':True,'status':'RESEARCH_PROVISIONAL_NOT_CHAMPION','model':'mu-v1.3 frozen 2024-25 fitted coefficients; 2026 ONLY feature history; no PS adaptation or lineup patch',
 'source_sha256':source_hash,'source_rows':len(raw),'target':'CLEAN_PRE_BASELINE_RESEARCH_NOT_OFFICIAL_LOCK',
 'n':n,'simulation_seed':20261103,'alpha_f5':0.54,'alpha_late_ASSUMED':0.322,
 'mu':mu,'stats':stats,'features':{'Samsung_away':away,'KT_home':home},
 'f5':f5_wdl,'full':full_wdl,'total':totals,'full_handicap':full_hcap,'f5_handicap':f5_hcap,'win1loss':win1loss,'margin':margin,'double':double,'exact_total':exact,
 'pool_hr_score_144':{'status':'UNAVAILABLE_NO_VALIDATED_JOINT_HR_MODEL'},
 'audit':{'F5_sum':sum(f5_wdl.values()),'Full_sum':sum(full_wdl[k] for k in ['KT_win','Samsung_win','draw_after_15']),
 'run_monotonic':bool(np.all(home_full>=fh) and np.all(away_full>=fa)), 'win1loss_sum':sum(win1loss.values()), 'exact_sum':sum(exact.values()),'double_sum':sum(double.values()),'margin_sum':sum(margin['KT'].values())+sum(margin['Samsung'].values())+full_wdl['draw_after_15']},
 'limitations':['2024/2025 historical PBP not loaded so shrinkage is 2026-only (research fast path).','No confirmed lineups, current park-weather model coefficient or projected first-follower team quality.','PS series fatigue and long rest not treated as hard runs penalties.','OT runs simulated via independent Poisson events, no walkoff handling.','No player-level HR×score joint model: pool unavailable.','No execution odds inspected; no official price board/EV.']
}
assert all(abs(ans['audit'][k]-1)<.00005 for k in ['F5_sum','Full_sum','win1loss_sum','exact_sum','double_sum','margin_sum']),ans['audit']
assert ans['audit']['run_monotonic']
OUT.write_text(json.dumps(ans,ensure_ascii=False,indent=2))
print('KS1_PRE_RESULT='+json.dumps({k:ans[k] for k in ['status','source_sha256','source_rows','mu','stats','f5','full','win1loss','audit']},ensure_ascii=False))
print('KS1_TOTALS='+json.dumps({'full_8.5':totals['Full']['8.5'],'full_9.5':totals['Full']['9.5'],'f5_4.5':totals['F5']['4.5'],'f5_5.5':totals['F5']['5.5'],'margin':margin,'double':double},ensure_ascii=False))
