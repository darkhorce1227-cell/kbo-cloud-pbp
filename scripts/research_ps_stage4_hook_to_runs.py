from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import poisson, skellam

SOURCE=Path('/tmp/ps-work/postseason_screen_pass.parquet')
STARTERS=Path('historical_ps/ps_starter_appearances.csv')
JOIN=Path('research/postseason/ps_rs_starter_avg_ip.csv')
OUTPUT=Path('/tmp/ps-work/stage4_results.json')

raw=pd.read_parquet(SOURCE).reset_index(drop=True)
st=pd.read_csv(STARTERS,dtype={'game_pk':str,'starter_id':str})
join=pd.read_csv(JOIN,dtype={'game_pk':str,'starter_id':str})
keys=set(st.game_pk.astype(str))
assert len(keys)==143 and len(st)==286,(len(keys),len(st))
req={'season','game_pk','home_team','away_team','inning','inning_topbot','at_bat_number','pitcher','post_home_score','post_away_score'}
assert not req-set(raw.columns), req-set(raw.columns)
raw['game_pk']=raw.game_pk.astype(str)
raw=raw[raw.game_pk.isin(keys)].copy()
raw['_row']=np.arange(len(raw))
raw['half']=raw.inning_topbot.astype(str).str.lower().replace({'bottom':'bot','b':'bot','t':'top'})
assert set(raw.half.unique())=={'top','bot'},raw.half.unique()
raw['def_team']=np.where(raw.half=='top',raw.home_team,raw.away_team)
raw['pitcher_id']=raw.pitcher.astype('string').str.strip()
assert raw.pitcher_id.notna().all()
for c in ['inning','at_bat_number','post_home_score','post_away_score']:
 raw[c]=pd.to_numeric(raw[c],errors='raise').astype(int)
raw=raw.sort_values(['game_pk','_row'],kind='stable')
seq=raw.drop_duplicates(['game_pk','def_team','pitcher_id'],keep='first').copy()
seq['role_order']=seq.groupby(['game_pk','def_team']).cumcount()
roles=seq[['game_pk','def_team','pitcher_id','role_order']]
pa=raw.drop_duplicates(['game_pk','at_bat_number'],keep='last').copy()
pa=pa.sort_values(['game_pk','_row'],kind='stable')
for side in ['home','away']:
 c='post_'+side+'_score'
 pa['runs_'+side]=(pa[c]-pa.groupby('game_pk',sort=False)[c].shift().fillna(0)).astype(int)
assert (pa[['runs_home','runs_away']]>=0).all().all(),"negative score delta"
pa['runs']=np.where(pa.half=='top',pa.runs_away,pa.runs_home)
pa['other_runs']=np.where(pa.half=='top',pa.runs_home,pa.runs_away)
assert pa.other_runs.eq(0).all(),"score change during opposite team at bat"
pa=pa.merge(roles,on=['game_pk','def_team','pitcher_id'],validate='many_to_one')
pa['role']=np.select([pa.role_order==0,pa.role_order==1],['SP','FIRST'],default='LATER')
pa['f5_runs']=np.where(pa.inning<=5,pa.runs,0)
pa['late_runs']=np.where(pa.inning>=6,pa.runs,0)
panel=pa.groupby(['game_pk','def_team'],as_index=False).agg(runs=('runs','sum'),f5=('f5_runs','sum'),late=('late_runs','sum'))
role=pa.groupby(['game_pk','def_team','role'],as_index=False).agg(runs=('runs','sum'),f5=('f5_runs','sum'),late=('late_runs','sum'),pa=('runs','size'))
wide=role.pivot(index=['game_pk','def_team'],columns='role',values=['runs','f5','late','pa']).fillna(0)
wide.columns=[f'{v}_{r}' for v,r in wide.columns]
wide=wide.reset_index()
panel=panel.merge(wide,on=['game_pk','def_team'],validate='one_to_one')
panel=panel.merge(st[['game_pk','team','actual_outs','starter_name','season','home_away']],left_on=['game_pk','def_team'],right_on=['game_pk','team'],validate='one_to_one')
panel['starter_ip']=panel.actual_outs/3
assert len(panel)==286
assert panel.runs.sum()==panel.filter(regex='^runs_(SP|FIRST|LATER)$').sum().sum()
assert np.all(panel.runs==panel.f5+panel.late)
assert (panel.runs==panel[['runs_SP','runs_FIRST','runs_LATER']].sum(axis=1)).all()
assert (panel.f5==panel[['f5_SP','f5_FIRST','f5_LATER']].sum(axis=1)).all()
assert (panel.late==panel[['late_SP','late_FIRST','late_LATER']].sum(axis=1)).all()
# Cross-check final game score for both sides.
last=pa.groupby('game_pk').tail(1).set_index('game_pk')
for g,t in panel.groupby('game_pk'):
 assert t.loc[t.home_away=='away','runs'].iloc[0]==int(last.loc[g,'post_home_score'])
 assert t.loc[t.home_away=='home','runs'].iloc[0]==int(last.loc[g,'post_away_score'])

panel['branch']=pd.cut(panel.starter_ip,[-.01,3-1e-9,5-1e-9,100],labels=['LT3','3TO5','GE5'])
def agg(r):
 return dict(n=int(len(r)),full_mean=round(float(r.runs.mean()),3),f5_mean=round(float(r.f5.mean()),3),
    starter_runs_mean=round(float(r.runs_SP.mean()),3),first_runs_mean=round(float(r.runs_FIRST.mean()),3),
    later_runs_mean=round(float(r.runs_LATER.mean()),3),
    first_runs_per100PA=round(float(100*r.runs_FIRST.sum()/r.pa_FIRST.sum()),3) if r.pa_FIRST.sum()>0 else None,
    first_pa=int(r.pa_FIRST.sum()),first_present=int((r.pa_FIRST>0).sum()))
summary={str(k):agg(g) for k,g in panel.groupby('branch',observed=True)}
summary['ALL']=agg(panel)

# Train before 2021; test 2021–2025. NO future current-game outcomes as feature.
join['game_pk']=join.game_pk.astype(str)
join['rs_starter_avg_ip']=pd.to_numeric(join.rs_starter_avg_ip,errors='coerce')
rows=panel.merge(join[['game_pk','team','rs_starter_avg_ip','rs_link_status']],left_on=['game_pk','def_team'],right_on=['game_pk','team'],validate='one_to_one',suffixes=('','_join'))
rows=rows[(rows.rs_link_status=='MATCHED') & rows.rs_starter_avg_ip.notna()].copy()
train=rows[rows.season<=2020].copy(); test=rows[rows.season>=2021].copy()
assert len(train)>=120 and len(test)>=120
residual=(train.starter_ip-train.rs_starter_avg_ip).to_numpy()
prior=agg(train)
means={b:agg(train[train.branch==b]) for b in ['LT3','3TO5','GE5']}
# Conservative empirical Bayes shrinks all branch means toward overall historical mean,
# using a fixed 15-team-game pseudo count.
branch_n={b:len(train[train.branch==b]) for b in means}
for b in means:
 for c,basefield in [('full_mean','runs'),('f5_mean','f5')]:
  sub=train[train.branch==b][basefield]
  means[b][c+'_shrunk']=(float(sub.sum())+15*float(train[basefield].mean()))/(len(sub)+15)

from collections import defaultdict
rates={}
for z in test.itertuples():
 x=float(z.rs_starter_avg_ip)
 # prior-posterior smoothing (0.5 per branch)
 prior_bins=np.array([sum(x+residual<3),sum((x+residual>=3)&(x+residual<5)),sum(x+residual>=5)],float)
 w=(prior_bins+0.5)/(len(residual)+1.5)
 rate_full=float(sum(w[i]*means[b]['full_mean_shrunk'] for i,b in enumerate(['LT3','3TO5','GE5'])))
 rate_f5=float(sum(w[i]*means[b]['f5_mean_shrunk'] for i,b in enumerate(['LT3','3TO5','GE5'])))
 rates[(z.game_pk,z.def_team)]={'branch_weights':w.tolist(),'full':rate_full,'f5':rate_f5}

# Evaluate game-level totals and home-win, conditional on only pregame pitcher RS average IP.
# Direct, non-branch RS-IP scoring baseline to test whether branching adds value.
from scipy.optimize import minimize

def fit_direct(col):
 x=(train.rs_starter_avg_ip.to_numpy(float)-5.5)
 y=train[col].to_numpy(float)
 def objective(t):
  eta=np.clip(t[0]+t[1]*x,-5,5)
  mu=np.exp(eta)
  return float(np.sum(mu-y*eta)+10*t[1]**2)
 result=minimize(objective,np.array([np.log(max(y.mean(),.01)),0.]),method='BFGS')
 return result.x
fit_f5=fit_direct('f5');fit_full=fit_direct('runs')
def direct_mu(z,fit):
 return float(np.exp(np.clip(fit[0]+fit[1]*(z.rs_starter_avg_ip-5.5),-5,5)))

actual=[]; pred_flat=[];pred_branch=[];pred_direct=[]
for g,t in test.groupby('game_pk'):
 if len(t)!=2:continue
 away=t[t.home_away=='away'].iloc[0];home=t[t.home_away=='home'].iloc[0]
 # Forecast scoring: home offense faces away defensive starter and vice versa.
 score_home=int(away.runs);score_away=int(home.runs)
 f5_home=int(away.f5);f5_away=int(home.f5)
 target={'game':g,'season':int(away.season),'home_runs':score_home,'away_runs':score_away,'home_f5':f5_home,'away_f5':f5_away}
 actual.append(target)
 pred_flat.append({'game':g,'home':float(train.runs.mean()),'away':float(train.runs.mean()),'home_f5':float(train.f5.mean()),'away_f5':float(train.f5.mean())})
 pred_branch.append({'game':g,'home':rates[(g,away.def_team)]['full'],'away':rates[(g,home.def_team)]['full'],
              'home_f5':rates[(g,away.def_team)]['f5'],'away_f5':rates[(g,home.def_team)]['f5']})
 pred_direct.append({'game':g,'home':direct_mu(away,fit_full),'away':direct_mu(home,fit_full),
              'home_f5':direct_mu(away,fit_f5),'away_f5':direct_mu(home,fit_f5)})

def test_metrics(pred):
 mm={};n=len(actual)
 for scope in ['full','f5']:
  fn=(lambda x:('home','away')) if scope=='full' else (lambda x:('home_f5','away_f5'))
  a,b=fn(None)
  obs_h=np.array([z['home_runs' if scope=='full' else 'home_f5'] for z in actual]);obs_a=np.array([z['away_runs' if scope=='full' else 'away_f5'] for z in actual]);
  mu_h=np.array([p[a] for p in pred]);mu_a=np.array([p[b] for p in pred]);total=obs_h+obs_a
  p_home=skellam.sf(0,mu_h,mu_a)
  y_home=(obs_h>obs_a).astype(int)
  threshold=8 if scope=='full' else 4
  p_over=poisson.sf(threshold,mu_h+mu_a);y_over=(total>threshold).astype(int)
  score_ll=float(np.mean(-(poisson.logpmf(obs_h,mu_h)+poisson.logpmf(obs_a,mu_a))))
  mm[scope]={'games':n,'team_score_poisson_nll':round(score_ll,5),
    'home_win_brier':round(float(np.mean((p_home-y_home)**2)),6),
    'over_'+str(threshold+0.5)+'_brier':round(float(np.mean((p_over-y_over)**2)),6),
    'total_mean_pred':round(float((mu_h+mu_a).mean()),3),'total_mean_actual':round(float(total.mean()),3)}
 return mm
def game_loss_arrays(pred):
 out={}
 for scope in ['full','f5']:
  a,b=('home','away') if scope=='full' else ('home_f5','away_f5')
  y_h=np.asarray([z['home_runs' if scope=='full' else 'home_f5'] for z in actual]);y_a=np.asarray([z['away_runs' if scope=='full' else 'away_f5'] for z in actual]);
  mu_h=np.asarray([z[a] for z in pred]);mu_a=np.asarray([z[b] for z in pred]);
  p_home=skellam.sf(0,mu_h,mu_a)
  p_over=poisson.sf(8 if scope=='full' else 4,mu_h+mu_a)
  out[scope+'_score_nll']=-(poisson.logpmf(y_h,mu_h)+poisson.logpmf(y_a,mu_a))
  out[scope+'_win_brier']=(p_home-(y_h>y_a))**2
  out[scope+'_over_brier']=(p_over-((y_h+y_a)>(8 if scope=='full' else 4)))**2
 return out

flat_loss=game_loss_arrays(pred_flat)
branch_loss=game_loss_arrays(pred_branch)
direct_loss=game_loss_arrays(pred_direct)
rng=np.random.default_rng(20261102)
bootidx=rng.integers(0,len(actual),size=(12000,len(actual)))
bootstrap={}
for metric in flat_loss:
 bootstrap[metric]={}
 for ref_name,ref in [('vs_pooled',flat_loss),('vs_direct_rsip',direct_loss)]:
  delta=branch_loss[metric]-ref[metric]
  bootmeans=delta[bootidx].mean(axis=1)
  bootstrap[metric][ref_name]={'delta':round(float(delta.mean()),6),
     'ci95':[round(float(x),6) for x in np.quantile(bootmeans,[.025,.975])]}

result={'status':'PBP_PATH_PASSES_SCORE_AUDIT','source':'historical-ps-master/postseason_screen_pass.parquet','sample_games':143,
 'pitch_rows':len(raw),'pa_rows':len(pa),'team_game_rows':len(panel),'by_actual_starter_hook':summary,
 'validation':{'all_games_two_starters':len(panel)==286,'score_partition':True,'f5_late_partition':True,'game_final_crosscheck':True},
 'temporal_test':{'train_team_starts':len(train),'test_team_starts':len(test),'test_complete_games':len(actual),
    'feature':'pregame same-year RS starter avg IP only, historical PS residual mixture (LT3 / 3to5 / GE5)',
    'baseline':'2016-2020 pooled PS runs/team','branch_train':means,
    'baseline_metrics':test_metrics(pred_flat),'direct_rsip_metrics':test_metrics(pred_direct),'branch_mixture_metrics':test_metrics(pred_branch),
    'paired_game_bootstrap12000':bootstrap,'test_seasons':{str(y):len([z for z in actual if z['season']==y]) for y in range(2021,2026)}},
 'limitations':['Runs are scoreboard events while active pitcher was on watch, not official R/ER or causal pitching quality.',
   'Starter hook occurs after runs and score-state; descriptive conditionings are selected on outcomes.',
   'Historical pooled scorer forecasts omit batter, park, weather and bullpen quality, so are not Champion and do not establish Champion gain.',
   'Team-innings run events are attributed to final pitcher of PA when a mid-PA pitching change occurs.',
   'No betting ROI or EV without contemporaneous posted bookmaker odds.']}
OUTPUT.parent.mkdir(parents=True,exist_ok=True)
OUTPUT.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=float))
print('STAGE4_RESULT_JSON='+json.dumps(result,ensure_ascii=False,default=float))