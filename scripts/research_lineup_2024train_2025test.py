"""KBO 2026 lineup quality shadow: sequential 2024→2025 external-season test, no target-day outcomes in features.

IMPORTANT: Historical starting 9 is reconstructed RETROSPECTIVELY from PBP's first
9 distinct batters (by 4th inning). This tests a *proxy* for announced lineup,
not proof those names were timestamped and published before betting markets.
This is not the official Champion and must not be promoted by itself.
"""
from __future__ import annotations
import json, math, hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import poisson, skellam
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

SRC=Path('/tmp/kbo-multiyear/kbo_rs_2024_2025.parquet')
OUT=Path('/tmp/kbo-multiyear/lineup_shadow_2024train_2025test.json')
EXPECTED_SHA='COMBINED_ASSET_SHA_AT_RUNTIME'  # Each original season file SHA checked against immutable manifest by workflow
assert SRC.is_file() and SRC.stat().st_size>10_000_000, 'validated training source is unavailable'
COMBINED_SHA=hashlib.sha256(SRC.read_bytes()).hexdigest()
x=pd.read_parquet(SRC)
assert len(x)>400_000,len(x)
x=x[(x.game_date.astype(str).between('2024-03-23','2024-09-30')) | (x.game_date.astype(str).between('2025-03-22','2025-09-30'))].copy()  # excludes PS
x['game_date']=pd.to_datetime(x.game_date)
x=x.sort_values(['game_date','game_pk','at_bat_number','pitch_number'],kind='stable')
assert x.game_pk.nunique()>1050,x.game_pk.nunique()
pa=x.drop_duplicates(['game_pk','at_bat_number'],keep='last').copy()
pa=pa.sort_values(['game_date','game_pk','at_bat_number'],kind='stable')
pa['batting_team']=np.where(pa.inning_topbot.eq('top'),pa.away_team,pa.home_team)
pa['event']=pa.events.fillna('').astype(str)
pa['hit']=pa.event.isin(['single','double','triple','home_run']).astype('int8')
pa['tb']=pa.event.map({'single':1,'double':2,'triple':3,'home_run':4}).fillna(0).astype('int8')
pa['hr']=pa.event.eq('home_run').astype('int8')
pa['walk']=pa.event.isin(['walk','intent_walk','hit_by_pitch']).astype('int8')
pa['k']=pa.event.isin(['strikeout','strikeout_double_play']).astype('int8')
cols=['pa','hit','tb','hr','walk','k']
profiles={}
player_records=defaultdict(lambda:np.zeros(6,dtype=float))
league=np.zeros(6,dtype=float)
team_hist=defaultdict(list)
# All future game plate appearances are held until after every same-date prediction.
record=[]
summary={'recovered_lineups':0,'eligible_games':0,'missing_lineups':0,'bad_f5':0,'substitution_risk':0}
# Grouping by game avoids source-row target contamination.
games=[]
for pk,q in pa.groupby('game_pk',sort=False):
 q=q.sort_values('at_bat_number',kind='stable')
 date=pd.Timestamp(q.game_date.iloc[0]); home=str(q.home_team.iloc[0]);away=str(q.away_team.iloc[0]);
 # Genuine F5 full inning reconciliation requires PA through inning 5.
 f5=q.loc[q.inning.astype(int)<=5]
 if len(f5)==0 or f5.inning.astype(int).max()!=5:
  summary['bad_f5']+=1;continue
 f5last=f5.iloc[-1];full=q.iloc[-1]
 score_f5={'home':int(f5last.post_home_score),'away':int(f5last.post_away_score)}
 score_full={'home':int(full.post_home_score),'away':int(full.post_away_score)}
 # Pre-announced starter lineup PROXY: first distinct batters appear across first four innings.
 lineups={};left_fraction={}
 for team in [away,home]:
  qq=q[q.batting_team.eq(team)].drop_duplicates('batter',keep='first')
  nn=qq.head(9)
  if len(nn)!=9 or (nn.inning.astype(int)>4).any():
   lineups[team]=None;continue
  ids=nn.batter.astype(str).tolist()
  lineups[team]=ids
  left_fraction[team]=float(nn.stand.eq('L').mean())
 if all(lineups.get(t) is not None for t in [away,home]):summary['recovered_lineups']+=1
 else:summary['missing_lineups']+=1
 games.append(dict(pk=str(pk),date=date,away=away,home=home,score_f5=score_f5,score_full=score_full,lineups=lineups,left_fraction=left_fraction,pa=q))
games.sort(key=lambda g:(g['date'],g['pk']))
summary['eligible_games']=len(games)
assert len(games)>1050,(len(games),summary)
weights=np.array([1.12,1.09,1.06,1.04,1.02,1.00,.96,.92,.89])
weights/=weights.sum()

def season_mean(t,which,kind):
 v=team_hist.get(t,[])
 if not v:return None
 return np.mean([r[kind][which] for r in v])

def prior_team(t,d,metric,ag,lgmean):
 hist=team_hist[t]
 v=[float(k[metric][ag]) for k in hist]
 recent=[float(k[metric][ag]) for k in hist if k['date']>=d-pd.Timedelta(days=30)]
 # independent coarse, conventional 10-game pseudo-count; fixed without tuning
 long=(sum(v)+12*lgmean)/(len(v)+12)
 short=(sum(recent)+12*lgmean)/(len(recent)+12)
 return long,short

def hitter_features(ids):
 if ids is None:return None
 # fixed within-year 120 PA pseudo sample. Prior rates are strictly pre-date.
 rate=(league[1:]+np.array([38,58,6,28,70],float))/(league[0]+np.array([120,120,120,120,120],float))
 # Prior defaults only affect early season; checked as baseline in dates Jul+.
 hp=[]; unknown=0
 for p in ids:
  v=player_records.get(p)
  if v is None: v=np.zeros(6,dtype=float)
  if v[0]<30:unknown+=1
  hp.append((v[1:]+120*rate)/(v[0]+120))
 a=np.average(np.asarray(hp),weights=weights,axis=0)
 return a,unknown

# Historical league team scores from previous days only; do not recalc target scores until date batch complete.
for date,daygames in pd.Series(games).groupby(lambda i:games[i]['date']):
 # previous-game global team-run league environment; fixed fallback only before May 1
 historical=np.array([entry[m][side] for t in team_hist for entry in team_hist[t] for m in ('f5','full') for side in ('for',)],dtype=float)
 past_f5=[ent['f5']['for'] for t in team_hist for ent in team_hist[t]]
 past_full=[ent['full']['for'] for t in team_hist for ent in team_hist[t]]
 lf=float(np.mean(past_f5)) if past_f5 else 2.4
 lfull=float(np.mean(past_full)) if past_full else 4.8
 for g in daygames:
  for team,opp,is_home in [(g['away'],g['home'],0),(g['home'],g['away'],1)]:
   ids=g['lineups'].get(team)
   if ids is None:continue
   hp=hitter_features(ids)
   assert hp is not None
   rates,unseen=hp
   # use league prior in hitter rate, but also expose league-mean-corrected lineup features
   cur_league=(league[1:]+np.array([38,58,6,28,70],float))/(league[0]+120.)
   ml=[float(v) for v in rates-cur_league]
   for scope,lg in [('f5',lf),('full',lfull)]:
    a,b=prior_team(team,date,scope,'for',lg)
    c,d=prior_team(opp,date,scope,'against',lg)
    record.append({'date':date,'pk':g['pk'],'team':team,'is_home':is_home,
        'scope':scope,'target':g['score_f5' if scope=='f5' else 'score_full']['home' if is_home else 'away'],
        'f_off_long':a/lg,'f_off30':b/lg,'f_opp_d_long':c/lg,'f_opp_d30':d/lg,
        'lineup_hit_delta':ml[0], 'lineup_tb_delta':ml[1],
        'lineup_hr_delta':ml[2], 'lineup_walk_delta':ml[3],
        'lineup_k_delta':ml[4], 'unseen':unseen,
        'left_frac':g['left_fraction'].get(team,None),
        'source':'postgame_reconstructed_first_nine_NOT_prebook_timestamped'})
 # Close current day for use by later dates ONLY.
 for g in daygames:
  q=g['pa']
  for side,code,opp in [('away',g['away'],g['home']),('home',g['home'],g['away'])]:
   other='home' if side=='away' else 'away'
   team_hist[code].append({'date':date,'f5':{'for':g['score_f5'][side],'against':g['score_f5'][other]},
     'full':{'for':g['score_full'][side],'against':g['score_full'][other]}})
  day_by_p=q.groupby(q.batter.astype(str))[['hit','tb','hr','walk','k']].agg(['sum','count'])
  for batter,bs in q.groupby(q.batter.astype(str)):
   a=bs[['hit','tb','hr','walk','k']].sum().to_numpy(float)
   counts=player_records[batter];counts+=np.r_[len(bs),a]
  league+=np.r_[len(q),q[['hit','tb','hr','walk','k']].sum().to_numpy(float)]

frame=pd.DataFrame(record)
assert frame.groupby(['pk','scope']).size().max()==2
frame['period']=np.select([frame.date.between('2024-05-01','2024-09-30'),frame.date.between('2025-05-01','2025-09-30')],['train','test'],default='earlier')
base=['f_off_long','f_off30','f_opp_d_long','f_opp_d30','is_home']
line=['lineup_hit_delta','lineup_tb_delta','lineup_hr_delta','lineup_walk_delta','lineup_k_delta']
# A separate simple additive model, not a retrofit of original v1.2 coefficients.
results={'data_sha256':COMBINED_SHA,'source_rows_after_date_scope':len(x),'baseball_games':len(games),
 'lineup_proxy_reconstruction':summary,'period':'train 2024-05-01..09-30 / test 2025-05-01..09-30',
 'status':'RESEARCH_2024_TRAIN_2025_TEST_RETROSPECTIVE_LINEUP_PROXY_NOT_CHAMPION',
 'hypotheses':{'baseline':base,'challenger':base+line,'alpha':0.10},'scopes':{},
 'limitations':['Starter lineup inferred postgame from first nine distinct batter IDs, not official timestamped pregame lineup','Missing lineups dropped in BOTH candidate and baseline for fair matching','Train 2024 and test 2025 with prior-date player history; no true pitcher-throws platoon matchup, fielding/park or hard-hit velocity model','Team/prior batter features use strictly prior completed dates, same-day feature updates deferred','Fixed alpha 0.10 and 120PA shrinkage are research choices, not validated champion hyperparameters','No forward-date independently captured official lineup timestamps or betting prices; should not promote to Champion']}
predictions=[]
for scope in ('f5','full'):
 d=frame[frame.scope==scope].copy()
 tr=d[d.period=='train'].copy();te=d[d.period=='test'].copy()
 assert len(tr)>400 and len(te)>450,(scope,len(tr),len(te),summary)
 for model_name,feats in [('baseline',base),('lineup_add',base+line)]:
  sx=StandardScaler();Xtr=sx.fit_transform(tr[feats].to_numpy(float));Xte=sx.transform(te[feats].to_numpy(float))
  fit=PoissonRegressor(alpha=.10,max_iter=1000,tol=1e-8)
  fit.fit(Xtr,tr.target.to_numpy(float));mu=np.clip(fit.predict(Xte),.02,25)
  loss=-poisson.logpmf(te.target.to_numpy(int),mu)
  is3=(te.target.to_numpy(int)>=3).astype(float)
  prob3=1-poisson.cdf(2,mu)
  results['scopes'].setdefault(scope,{})[model_name]={
   'train_team_games':len(tr),'test_team_games':len(te),'test_games':int(te.pk.nunique()),
   'score_poisson_nll':round(float(np.mean(loss)),7),
   'score_mae':round(float(np.mean(np.abs(mu-te.target))),7),
   '3plus_Brier':round(float(np.mean((prob3-is3)**2)),7),
   'test_actual_avg':round(float(te.target.mean()),7),'test_pred_avg':round(float(mu.mean()),7),
   'coefs_by_feature':dict(zip(feats,np.round(fit.coef_,6).tolist()))}
  o=te[['pk','date','team','is_home','target']].copy();o['mu']=mu;o['loss']=loss;o['model']=model_name;o['scope']=scope
  predictions.append(o)
 p=pd.concat(predictions[-2:]);wide=p.pivot(index=['pk','is_home'],columns='model',values='mu').reset_index()
 # 3-way F5 WDL & Full side on same pairs, independent Poisson for diagnostic only
 score=te.pivot(index='pk',columns='is_home',values='target');assert len(score)==len(wide.pk.unique())
 for model_name in ['baseline','lineup_add']:
  q=wide.pivot(index='pk',columns='is_home',values=model_name).reindex(score.index)
  home=q[1].to_numpy();away=q[0].to_numpy();real_diff=(score[1]-score[0]).to_numpy()
  probs=np.c_[skellam.cdf(-1,home,away),skellam.pmf(0,home,away),1-skellam.cdf(0,home,away)]
  # columns [away wins,draw,home wins]
  idx=np.where(real_diff>0,2,np.where(real_diff<0,0,1))
  ptrue=probs[np.arange(len(idx)),idx]
  brier=np.mean(np.sum((probs-np.eye(3)[idx])**2,axis=1)/3)
  nll=-np.log(np.maximum(ptrue,1e-9))
  results['scopes'][scope][model_name]['game_side_wdl_logloss']=round(float(nll.mean()),7)
  results['scopes'][scope][model_name]['game_side_wdl_Brier']=round(float(brier),7)
# Paired-game bootstrap on SCORE NLL differences, per game 2 team score entries.
rng=np.random.default_rng(20261104)
for scope in ('f5','full'):
 p=pd.concat([o for o in predictions if o['scope'].iloc[0]==scope])
 q=p.pivot(index=['pk','team'],columns='model',values='loss').reset_index()
 q['delta']=q.lineup_add-q.baseline
 gd=q.groupby('pk').delta.mean().to_numpy(float)
 actual=gd.mean();bs=np.mean(rng.choice(gd,size=(5000,len(gd)),replace=True),axis=1)
 results['scopes'][scope]['delta_lineup_minus_baseline']={
  'n_games':len(gd),'delta_score_Poisson_NLL':round(float(actual),7),
  '95pct_bootstrap_CI':[round(float(x),7) for x in np.quantile(bs,[.025,.975])],
  'is_improved_if_negative':bool(actual<0),'ci_excludes_zero':bool(np.quantile(bs,.025)>0 or np.quantile(bs,.975)<0)}
print('LINEUP_RESEARCH_RESULT='+json.dumps({k:results[k] for k in ['data_sha256','lineup_proxy_reconstruction','period','scopes']},ensure_ascii=False))
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')