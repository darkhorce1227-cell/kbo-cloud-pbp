#!/usr/bin/env python3
"""KT/league bridge candidate pregame-only 2026 chronological test.
No actual same-game role or outcome is used in features. Target: 6-8 runs conceded.
Research / not champion. Deliberately low-dimensional regularized Poisson.
"""
from __future__ import annotations
import hashlib, json, math, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import gammaln

ROOT=Path(os.environ.get('KT_RESEARCH_ROOT','/tmp/kt-bridge-verify'))
SRC=ROOT/'kbo_pbp_2026_latest.parquet'
OUT=ROOT/'kt_bridge_oos_2026.json'
EXPECTED_SHA='32c70d31c89c83a9e8606254ff6ec084f599371b6debc6dde36066bf122c74e1'
assert hashlib.sha256(SRC.read_bytes()).hexdigest()==EXPECTED_SHA
raw=pd.read_parquet(SRC)
assert len(raw)==224611
assert not raw.duplicated(['game_pk','at_bat_number','pitch_number']).any()
raw=raw[raw.game_pk.astype(str).str.startswith('2026')].copy()
assert raw.game_pk.nunique()==720
raw['date']=pd.to_datetime(raw.game_date).dt.normalize()
raw['half']=np.where(raw.inning_topbot.astype(str).str.lower().str.startswith('top'),'top','bot')
raw['pitching_team']=np.where(raw.half=='top',raw.home_team,raw.away_team)
raw['batting_team']=np.where(raw.half=='top',raw.away_team,raw.home_team)
raw['pitcher']=raw.pitcher.astype(str)
raw=raw.sort_values(['game_pk','at_bat_number','pitch_number'],kind='stable')
starters=raw.drop_duplicates(['game_pk','pitching_team'],keep='first')[['game_pk','pitching_team','pitcher']].rename(columns={'pitcher':'starter'})
raw=raw.merge(starters,on=['game_pk','pitching_team'],validate='many_to_one')
raw['is_starter']=raw.pitcher.eq(raw.starter)
first=raw.drop_duplicates(['game_pk','at_bat_number'],keep='first').set_index(['game_pk','at_bat_number'])
pa=raw.drop_duplicates(['game_pk','at_bat_number'],keep='last').copy().reset_index(drop=True)
idx=pd.MultiIndex.from_frame(pa[['game_pk','at_bat_number']])
pa['first_outs']=idx.map(first.outs_when_up).astype(int)
for c in ['on_1b','on_2b','on_3b']:pa['first_'+c]=idx.map(first[c])
pa['prev_home']=pa.groupby('game_pk').post_home_score.shift().fillna(0).astype(int)
pa['prev_away']=pa.groupby('game_pk').post_away_score.shift().fillna(0).astype(int)
pa['runs']=(pa.post_home_score+pa.post_away_score-pa.prev_home-pa.prev_away).astype(int)
assert pa.runs.ge(0).all()
pa['bf']=pa.events.notna().astype(int)
for ev,new in [('strikeout','k'),('walk','bb'),('home_run','hr')]:pa[new]=pa.events.eq(ev).astype(int)
pa['outs']=pa.post_outs-pa.first_outs
pa['r68']=np.where(pa.inning.between(6,8),pa.runs,0)
pa['in68']=pa.inning.between(6,8)
pa['is_reliever']=~pa.is_starter

# All team-game totals including zero-run innings, and raw bullpen availability history.
base=pa.groupby(['game_pk','date','batting_team','pitching_team'],as_index=False).agg(r68=('r68','sum'))
assert len(base)==1440
# Enrich home/away from raw first pitch without current-game outcome.
games=raw.drop_duplicates('game_pk')[['game_pk','date','home_team','away_team']]
base=base.merge(games,on=['game_pk','date'],validate='many_to_one')
base['is_home']=base.batting_team.eq(base.home_team).astype(int)
# prior pitcher quality stats at PA granularity; late reliever prior appearances are role proxy.
rel=pa[pa.is_reliever].groupby(['date','pitching_team','pitcher'],as_index=False).agg(bf=('bf','sum'),k=('k','sum'),bb=('bb','sum'),hr=('hr','sum'),pa=('bf','size'),pa68=('in68','sum'))
# count actual pitcher raw pitch rows for preceding days (no same-day).
pt=raw.groupby(['date','pitching_team','pitcher'],as_index=False).size().rename(columns={'size':'pitches'})
rel=rel.merge(pt,on=['date','pitching_team','pitcher'],how='left',validate='one_to_one')
rel['kbb']=rel.k-rel.bb
rel['hr_count']=rel.hr

# All-time prior league reliever rates for a safe pregame prior; static fallback is descriptive training prior not target outcome.
# Use 2026 season prior each date, with conservative public-math fallback, not fitted from future.
BASE_KBB=.08;BASE_HR=.025
# 30D offensive/defensive team rate, smoothed to pregame league mean.
rows=[]
for d in sorted(base.date.unique()):
  hist=base[base.date<d]
  if len(hist)<160:continue
  hist30=hist[hist.date>=d-pd.Timedelta(days=30)]
  if hist30.empty:hist30=hist
  league=float(hist.r68.mean())
  league30=float(hist30.r68.mean())
  # league prior relief rates only prior-date; no target future.
  rh=rel[rel.date<d]
  totalbf=float(rh.bf.sum())
  lkbb=float((rh.k.sum()-rh.bb.sum())/totalbf) if totalbf else BASE_KBB
  lhr=float(rh.hr.sum()/totalbf) if totalbf else BASE_HR
  for _,r in base.loc[base.date==d].iterrows():
    off=r.batting_team;de=r.pitching_team
    offh=hist[hist.batting_team==off];defh=hist[hist.pitching_team==de]
    off30=hist30[hist30.batting_team==off];de30=hist30[hist30.pitching_team==de]
    def rate(z,anchor,pseudo):return float((z.r68.sum()+pseudo*anchor)/(len(z)+pseudo))
    # Candidate bridge roles chosen with ONLY earlier game outings in innings 6-8.
    reliefhist=rh[rh.pitching_team==de]
    r30=reliefhist[reliefhist.date>=d-pd.Timedelta(days=35)]
    # 35 day appearance counts, long-term quality. Exclude pitchers who did not relief in last 35 days.
    roles=r30.groupby('pitcher',as_index=False).agg(pa68=('pa68','sum'),pitches=('pitches','sum'))
    roles=roles[roles.pa68>=2].sort_values(['pa68','pitcher'],ascending=[False,True]).head(4)
    qual=[];wt=[];busy=[];hrqs=[]
    for q in roles.itertuples():
      person=reliefhist[reliefhist.pitcher==q.pitcher]
      bf=float(person.bf.sum())
      kbb=(float(person.kbb.sum())+100*lkbb)/(bf+100)
      hrq=(float(person.hr_count.sum())+180*lhr)/(bf+180)
      two=float(person[person.date>=d-pd.Timedelta(days=2)].pitches.sum())
      # Relative role priority from earlier 35day inning6–8 appearances; freshness only candidate availability.
      w=float(q.pa68)
      qual.append(kbb);hrqs.append(hrq);wt.append(w);busy.append(float(two>=40))
    if wt:
      bridge_kbb=float(np.average(qual,weights=wt));bridge_hr=float(np.average(hrqs,weights=wt))
      workload_busy=float(np.average(busy,weights=wt));coverage=int(len(wt))
    else:
      bridge_kbb=lkbb;bridge_hr=lhr;workload_busy=0.0;coverage=0
    rows.append({
      'game_pk':r.game_pk,'date':str(pd.Timestamp(d).date()),'batting_team':off,'pitching_team':de,'is_home':int(r.is_home),'y':int(r.r68),
      'league':league,'league30':league30,
      'off_season':rate(offh,league,16),'off30':rate(off30,league30,12),
      'def_season':rate(defh,league,16),'def30':rate(de30,league30,12),
      'bridge_kbb':bridge_kbb,'bridge_hr':bridge_hr,'league_kbb':lkbb,'league_hr':lhr,
      'role_coverage':coverage,'role_busy40':workload_busy,
      'def_prior_games':len(defh),'opp_prior_games':len(offh)
    })
df=pd.DataFrame(rows).sort_values(['date','game_pk','batting_team'])
assert not df.empty and df.date.min()>'2026-03-28'
# Train to first half, evaluate second half. No hyperparameter tuning on test.
train=df[(df.date<'2026-07-16')&(df.date>='2026-05-01')].copy()
test=df[df.date>='2026-07-16'].copy()
assert len(train)>300 and len(test)>200,(len(train),len(test))
for z in [train,test]:
 z['off_rel']=z.off30/z.league30
 z['def_rel']=z.def30/z.league30
 z['kbb_delta']=z.bridge_kbb-z.league_kbb
 z['hr_delta']=z.bridge_hr-z.league_hr
 z['busy']=z.role_busy40
 z['role_count']=z.role_coverage

# Approximate regularized Poisson with an offset: control existing 6-8 runs histories.
def fit(cols,alpha=30.):
 x=train[cols].to_numpy(float)
 mean=x.mean(0);scale=x.std(0);scale[scale<1e-8]=1
 design=np.column_stack((np.ones(len(x)),(x-mean)/scale))
 y=train.y.to_numpy(float)
 offset=np.log(np.clip(train.league.to_numpy(float),.1,None))
 theta=np.zeros(design.shape[1]);theta[0]=np.log((np.mean(y)+1e-8)/(np.mean(np.exp(offset))+1e-8))
 pen=np.ones(len(theta))*alpha;pen[0]=0
 for _ in range(100):
  mu=np.exp(np.clip(offset+design@theta,-12,12))
  grad=design.T@(y-mu)-pen*theta
  info=design.T@(design*mu[:,None])+np.diag(pen)+np.eye(len(theta))*1e-8
  delta=np.linalg.solve(info,grad)
  theta+=delta
  if np.max(np.abs(delta))<1e-9:break
 def predict(data):
  z=data[cols].to_numpy(float)
  des=np.column_stack((np.ones(len(z)),(z-mean)/scale))
  return np.exp(np.clip(np.log(np.clip(data.league.to_numpy(float),.1,None))+des@theta,-10,10))
 return predict,dict(coef={n:float(v) for n,v in zip(['intercept']+cols,theta)},train_n=len(train),alpha=alpha)

specs={
 'league_only':[],
 'prior_off_def':['off_rel','def_rel'],
 'bridge_quality':['off_rel','def_rel','kbb_delta','hr_delta','role_count'],
 'bridge_quality_workload':['off_rel','def_rel','kbb_delta','hr_delta','role_count','busy']
}
fits={};pred={};
for name,features in specs.items():
 if name=='league_only':
  fitted=lambda s: s.league.to_numpy(float)
  fits[name]={'coef':{},'train_n':len(train)}
 else:fitted,fits[name]=fit(features)
 pred[name]=np.clip(fitted(test),.0001,30)

def scores(data,p):
 y=data.y.to_numpy(float)
 loss=p-y*np.log(p)+gammaln(y+1)
 prob=1-np.exp(-p)*(1+p)
 brier=(prob-(y>=2))**2
 mae=np.abs(y-p)
 return {'nll':float(loss.mean()),'brier_2plus':float(brier.mean()),'mae':float(mae.mean()),'mean_mu':float(p.mean()),'mean_actual':float(y.mean())}, {'nll':loss,'brier_2plus':brier,'mae':mae}
res={};rawscore={}
for name,p in pred.items():res[name],rawscore[name]=scores(test,p)
# paired game-cluster bootstrap with sample games; no team-games as independent samples.
ids=np.array(sorted(test.game_pk.unique()));grp=test.groupby('game_pk').indices
rng=np.random.default_rng(20261104);B=6000
boot={}
for name in ('bridge_quality','bridge_quality_workload'):
 boot[name]={}
 for metric in ('nll','brier_2plus','mae'):
  dif=rawscore[name][metric]-rawscore['prior_off_def'][metric]
  by=np.array([float(np.mean(dif[grp[k]])) for k in ids])
  ind=rng.integers(0,len(ids),size=(B,len(ids)))
  draws=by[ind].mean(1)
  boot[name][metric]={'delta':float(dif.mean()),'ci95':[float(x) for x in np.quantile(draws,[.025,.975])]}
# Per-team diagnostic: test team score when defending.
perteam={}
for team,z in test.groupby('pitching_team'):
 inds=z.index.to_numpy();loc=test.index.get_indexer(inds)
 perteam[team]={'n':len(z),'realized_runs68':float(z.y.mean()),'prior_base_nll':scores(z,pred['prior_off_def'][loc])[0]['nll'],'quality_nll':scores(z,pred['bridge_quality'][loc])[0]['nll']}
kt=test[test.pitching_team.eq('KT')];ki=test.index.get_indexer(kt.index)
# Correlation and baseline composition is descriptive only.
result={'status':'RESEARCH_CHRONOLOGICAL_OOS_NOT_CHAMPION','dataset_sha256':EXPECTED_SHA,'source_games':720,'team_games':len(df),
 'train':{'start':'2026-05-01','end':'2026-07-15','n':len(train),'unique_games':train.game_pk.nunique()},
 'test':{'start':'2026-07-16','end':str(test.date.max()),'n':len(test),'unique_games':test.game_pk.nunique()},
 'models':res,'coefficients':fits,'bootstrap_6000_game_cluster':boot,
 'test_KT':{'games':len(kt),'observed_mean_6to8':float(kt.y.mean()),'base':scores(kt,pred['prior_off_def'][ki])[0],
            'quality':scores(kt,pred['bridge_quality'][ki])[0], 'quality_workload':scores(kt,pred['bridge_quality_workload'][ki])[0],
            'average_busy_share':float(kt.role_busy40.mean()),'avg_candidates':float(kt.role_coverage.mean())},
 'team_diagnostics':perteam,
 'limitations':['Target is 6th-8th innings team scored runs: includes starter continuing, not only bridge relievers.',
 'Only one 2026 season; test is subsequent 2026 games, not future-season or true independent preregistration.',
 'Pregame candidate set is based on past 35d 6-8 inning PA and no current active roster confirmation.',
 'Pitch-count proxy uses prior raw pitch rows; exact KBO official pitch counts and availability restrictions not captured.',
 'Quality uses K-BB and HR on prior season reliever PA and shrinks to pregame league rates; score-state and inherited runners not adjusted.',
 '2026 OOS observations were known at the time of exploratory rule construction. No Champion promotion based on this result.',
 'Same-date outcomes excluded from all pregame features and reliever history.']}
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('KT_BRIDGE_OOS_RESULT='+json.dumps({k:result[k] for k in ['status','source_games','train','test','models','bootstrap_6000_game_cluster','test_KT']},ensure_ascii=False))