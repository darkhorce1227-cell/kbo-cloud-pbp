"""Research only: can PRIOR-POSTSEASON bullpen evidence improve pregame score forecasting?
2016-20 fit, 2021-25 calendar-after split. Same-day or later games are never used
as reliever-history features. The actual first-follower ID of target is NEVER used.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson, skellam

SRC=Path('/tmp/ps-work/postseason_screen_pass.parquet')
START=Path('historical_ps/ps_starter_appearances.csv')
LINK=Path('research/postseason/ps_rs_starter_avg_ip.csv')
OUT=Path('/tmp/ps-work/ps_stage6_bullpen_pregame.json')
raw=pd.read_parquet(SRC).reset_index(drop=True)
st=pd.read_csv(START,dtype={'game_pk':str,'starter_id':str})
link=pd.read_csv(LINK,dtype={'game_pk':str,'starter_id':str})
keys=set(st.game_pk.astype(str));assert len(keys)==143
raw['game_pk']=raw.game_pk.astype(str);raw=raw[raw.game_pk.isin(keys)].copy()
raw['_row']=np.arange(len(raw));raw['half']=raw.inning_topbot.astype(str).str.lower().replace({'bottom':'bot','b':'bot','t':'top'})
raw['team']=np.where(raw.half=='top',raw.home_team,raw.away_team)
raw['pitcher_id']=raw.pitcher.astype('string').str.strip()
raw['game_date']=pd.to_datetime(raw.game_date)
raw['inning']=raw.inning.astype(int);raw['at_bat_number']=raw.at_bat_number.astype(int)
raw['post_home_score']=raw.post_home_score.astype(int);raw['post_away_score']=raw.post_away_score.astype(int)
first=raw.drop_duplicates(['game_pk','team','pitcher_id'],keep='first').sort_values('_row').copy()
first['order']=first.groupby(['game_pk','team']).cumcount()
first['role']=np.select([first.order==0,first.order==1],['SP','FIRST'],default='LATER')
first['year']=first.game_date.dt.year
pitch_counts=raw.groupby(['game_pk','team','pitcher_id']).size().rename('pitches').reset_index()
first=first.merge(pitch_counts,on=['game_pk','team','pitcher_id'],validate='one_to_one')
pa=raw.drop_duplicates(['game_pk','at_bat_number'],keep='last').sort_values('_row').copy()
for side in ('home','away'):
 c='post_'+side+'_score';pa['scored_'+side]=pa[c]-pa.groupby('game_pk')[c].shift().fillna(0)
assert (pa[['scored_home','scored_away']]>=0).all().all()
pa['runs']=np.where(pa.half=='top',pa.scored_away,pa.scored_home)
pa=pa.merge(first[['game_pk','team','pitcher_id','role']],on=['game_pk','team','pitcher_id'],validate='many_to_one')
pa['runs_f5']=np.where(pa.inning<=5,pa.runs,0)
pa_summary=pa.groupby(['game_pk','team','pitcher_id']).agg(pa=('runs','size'),runs=('runs','sum'),f5=('runs_f5','sum')).reset_index()
first=first.merge(pa_summary,on=['game_pk','team','pitcher_id'],how='left',validate='one_to_one').fillna({'pa':0,'runs':0,'f5':0})
for k in ('pa','runs','f5'):first[k]=first[k].astype(int)
team=first.groupby(['game_pk','team'],as_index=False).agg(season=('year','first'),game_date=('game_date','first'),runs=('runs','sum'),f5=('f5','sum'))
team=team.merge(st[['game_pk','team','home_away']],on=['game_pk','team'],validate='one_to_one')
link['rs_starter_avg_ip']=pd.to_numeric(link.rs_starter_avg_ip,errors='coerce')
team=team.merge(link[['game_pk','team','rs_starter_avg_ip','rs_link_status']],on=['game_pk','team'],validate='one_to_one')
assert len(team)==286
assert int(team.runs.sum())==int(pa.runs.sum())
# A constant learned from training window, NOT from later postseason years.
train_rel=first[(first.year<=2020)&(first.role!='SP')]
prior_global_rate=float(train_rel.runs.sum()/train_rel.pa.sum())
assert .04<prior_global_rate<.35

def get_pregame_features(row):
 history=first[(first.year==row.season)&(first.team==row.team)&(first.game_date<row.game_date)&(first.role!='SP')]
 # Historical reliever performance prior to this team's first PS game is missing.
 # Shrink toward a training-only league PS prior with 30 PA pseudo-observations.
 npa=int(history.pa.sum()); nr=int(history.runs.sum())
 team_rate=(nr+30*prior_global_rate)/(npa+30)
 candidates={}
 for h in history.itertuples():
  c=candidates.setdefault(str(h.pitcher_id),{'first':0,'other':0,'pa':0,'runs':0,'recent2d':0,'last':pd.Timestamp('1900-01-01')})
  c['first']+=int(h.role=='FIRST');c['other']+=int(h.role=='LATER')
  c['pa']+=int(h.pa);c['runs']+=int(h.runs);c['last']=max(c['last'],h.game_date)
  if (row.game_date-h.game_date).days<=2:c['recent2d']+=int(h.pitches)
 if candidates:
  expected_id=max(candidates,key=lambda x:(2*candidates[x]['first']+.5*candidates[x]['other'],candidates[x]['last'],x))
  f=candidates[expected_id]
  first_rate=(f['runs']+30*prior_global_rate)/(f['pa']+30)
  first_recent_2d=f['recent2d']
 else:
  first_rate=prior_global_rate;first_recent_2d=0
 return pd.Series({'prior_ps_relief_pa':npa,'bp_rate':team_rate,'pred_first_rate':first_rate,
                   'pred_first_last2_pitches':first_recent_2d,'candidate_n':len(candidates)})

feat=team.apply(get_pregame_features,axis=1)
team=pd.concat([team,feat],axis=1)
team=team[(team.rs_link_status=='MATCHED')&team.rs_starter_avg_ip.notna()].copy()
train=team[team.season<=2020].copy();test=team[team.season>=2021].copy()
assert len(train)==139 and len(test)==145

# Predict defensive team runs conceded, not scored. All features refer to DEFENSIVE team's history.
# Fit two separate Poisson means; 10-unit L2 on standardized slopes, unpenalized intercept.
# Starter-only baseline has identical form to previous Stage4, except all slopes standardized.
features={
 'direct_rsip':['rs_starter_avg_ip'],
 'rsip_bp_history':['rs_starter_avg_ip','bp_rate'],
 'rsip_first_history':['rs_starter_avg_ip','pred_first_rate'],
 'rsip_bp_and_first':['rs_starter_avg_ip','bp_rate','pred_first_rate'],
 'rsip_bp_first_recent_pitches':['rs_starter_avg_ip','bp_rate','pred_first_rate','pred_first_last2_pitches']}
# Fixed normalization based on TRAIN only and fixed prior; avoids target-year feature statistics.
transform={'rs_starter_avg_ip':lambda f:(f-5.5),
           'bp_rate':lambda f:(f-prior_global_rate)*10,
           'pred_first_rate':lambda f:(f-prior_global_rate)*10,
           'pred_first_last2_pitches':lambda f:f/50}

fit_output={};pred={}
for name,cols in features.items():
 X=np.column_stack([transform[c](train[c].to_numpy(float)) for c in cols]);
 Z=np.column_stack([transform[c](test[c].to_numpy(float)) for c in cols]);
 pred[name]={}
 for target in ('runs','f5'):
  y=train[target].to_numpy(float)
  def objective(t):
   eta=np.clip(t[0]+X@t[1:],-6,6)
   return float(np.sum(np.exp(eta)-y*eta)+10*np.sum(t[1:]**2))
  def gradient(t):
   eta=np.clip(t[0]+X@t[1:],-6,6)
   err=np.exp(eta)-y
   return np.r_[err.sum(), X.T@err+20*t[1:]]
  fit=minimize(objective,np.r_[np.log(max(y.mean(),.01)),np.zeros(X.shape[1])],jac=gradient,method='BFGS')
  mu=np.exp(np.clip(fit.x[0]+Z@fit.x[1:],-6,6))
  fit_output[name+'_'+target]={'success':bool(fit.success),'intercept':round(float(fit.x[0]),6),
       'slopes':dict(zip(cols,[round(float(x),6) for x in fit.x[1:]]))}
  pred[name][target]=dict(zip(zip(test.game_pk,test.team),mu))

complete=[]
for g,sub in test.groupby('game_pk'):
 if len(sub)!=2:continue
 away=sub[sub.home_away=='away'].iloc[0];home=sub[sub.home_away=='home'].iloc[0]
 complete.append(dict(game=g,season=int(away.season),away_team=away.team,home_team=home.team,
                     score_home=int(away.runs),score_away=int(home.runs),f5_home=int(away.f5),f5_away=int(home.f5),
                     observed_prior_relief_pa_home=int(away.prior_ps_relief_pa),
                     observed_prior_relief_pa_away=int(home.prior_ps_relief_pa)))
assert len(complete)==72

def loss(name):
 output={}
 for scope in ('runs','f5'):
  h=np.array([pred[name][scope][(g['game'],g['away_team'])] for g in complete]);
  a=np.array([pred[name][scope][(g['game'],g['home_team'])] for g in complete]);
  y_h=np.array([g['score_home' if scope=='runs' else 'f5_home'] for g in complete]);
  y_a=np.array([g['score_away' if scope=='runs' else 'f5_away'] for g in complete]);
  y_win=(y_h>y_a).astype(int);p_win=skellam.sf(0,h,a)
  y_over=((y_h+y_a)>(8 if scope=='runs' else 4)).astype(int)
  p_over=poisson.sf(8 if scope=='runs' else 4,h+a)
  # Valid 3-class F5 as an additional strict output: home/draw/away
  if scope=='f5':
   p_tie=skellam.pmf(0,h,a)
   y_nll_3=-(y_win*np.log(np.clip(p_win,1e-10,1)) + (1-y_win-(y_h<y_a).astype(int))*np.log(np.clip(p_tie,1e-10,1)) + (y_h<y_a)*np.log(np.clip(1-p_win-p_tie,1e-10,1)))
   output['f5_3way_logloss']=y_nll_3
  output[scope+'_score_nll']=-(poisson.logpmf(y_h,h)+poisson.logpmf(y_a,a))
  output[scope+'_home_win_brier']=(p_win-y_win)**2
  output[scope+'_over_brier']=(p_over-y_over)**2
 return output
ll={k:loss(k) for k in features}
metrics={k:{metric:round(float(x.mean()),6) for metric,x in arr.items()} for k,arr in ll.items()}
boot_rng=np.random.default_rng(20261103)
indices=boot_rng.integers(0,72,size=(10000,72))
comparison={}
for k in features:
 if k=='direct_rsip':continue
 comparison[k]={}
 for metric in metrics['direct_rsip']:
  d=ll[k][metric]-ll['direct_rsip'][metric]
  draws=d[indices].mean(axis=1)
  comparison[k][metric]={'delta':round(float(d.mean()),6),
                          'ci95':[round(float(x),6) for x in np.quantile(draws,[.025,.975])]}

result={'status':'RESEARCH_PS_PREGAME_BULLPEN_HISTORY_TEMPORAL_TEST','source':'historical-ps-master/postseason_screen_pass.parquet',
 'train_rows':len(train),'test_rows':len(test),'complete_test_games':len(complete),
 'train_global_PS_relief_scored_runs_per_PA':round(prior_global_rate,6),
 'test_prior_relief_observations':{'two_teams_have_prior':sum(g['observed_prior_relief_pa_home']>0 and g['observed_prior_relief_pa_away']>0 for g in complete),
                                   'no_prior_one_or_both':sum(g['observed_prior_relief_pa_home']==0 or g['observed_prior_relief_pa_away']==0 for g in complete)},
 'features':list(features),'fits':fit_output,'metrics_2021_2025':metrics,
 'paired_game_bootstrap_10000_vs_direct_rsip':comparison,
 'limits':['Historical PS reliever scored-runs per PA is NOT earned-run talent or pitching independent of opponent and context.',
           'The simulated pregame first reliever is ranked by same-season earlier PS usage; actual future selection never enters features.',
           'RS IP data are a same-year season summary; regular-season figures could reflect games after early PS in edge cases, although RS ended before postseason.',
           'No per-pitch velocity, batter ability, inherited runner state, roster confirmations or actual bookmaker prices are included.',
           'Model/feature selection was exploratory, not prospectively registered. These are weak historical benchmarks, not Champion.',
           'Same-day source observations are deliberately withheld.']}
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('STAGE6_RESULT_JSON='+json.dumps(result,ensure_ascii=False))