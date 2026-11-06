"""Frozen one-feature pitcher platoon residual OOS pilot; RESEARCH ONLY.
Input: SHA-audited 2024/25 Naver PBP training archive and 2026 validated latest release.
Lineups are retrospectively reconstructed (first nine PA batters), NOT pregame snapshots.
All predictor stats use strictly earlier dates, NEVER current/future outcomes.
"""
from __future__ import annotations
import json, math, os, hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import poisson, skellam
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

ROOT=Path(os.environ.get('PLATOON_ROOT','/tmp/kbo-platoon'))
SRC24=ROOT/'kbo_rs_2024_2025.parquet'
SRC26=ROOT/'kbo_pbp_2026_latest.parquet'
assert SRC24.is_file() and SRC26.is_file(), 'Missing PBP sources'
sha26=hashlib.sha256(SRC26.read_bytes()).hexdigest()
raw=pd.concat([pd.read_parquet(SRC24),pd.read_parquet(SRC26)],ignore_index=True)
raw['game_date']=pd.to_datetime(raw.game_date).dt.normalize()
raw=raw[((raw.game_date.dt.year.isin([2024,2025])) & (raw.game_date.dt.month<=9)) |
        ((raw.game_date.dt.year==2026) & (raw.game_date<=pd.Timestamp('2026-10-12')))].copy()
raw=raw.sort_values(['game_date','game_pk','at_bat_number','pitch_number'],kind='stable')
assert not raw.duplicated(['game_pk','at_bat_number','pitch_number']).any(), 'pitch duplicate'
raw['batting_team']=np.where(raw.inning_topbot.astype(str).str.lower().str.startswith('top'),raw.away_team,raw.home_team)
raw['pitching_team']=np.where(raw.inning_topbot.astype(str).str.lower().str.startswith('top'),raw.home_team,raw.away_team)
start=raw.drop_duplicates(['game_pk','pitching_team'],keep='first').set_index(['game_pk','pitching_team'])['pitcher'].astype(str).to_dict()
pa=raw.drop_duplicates(['game_pk','at_bat_number'],keep='last').copy()
pa=pa[pa.events.notna()].copy()
pa['pitcher']=pa.pitcher.astype(str)
pa['batter']=pa.batter.astype(str)
pa['side']=pa.stand.astype(str).str.upper()
pa['e']=pa.events.astype(str)
pa['tb']=pa.e.map({'single':1,'double':2,'triple':3,'home_run':4}).fillna(0).astype(float)
pa['bbhbp']=pa.e.isin(['walk','intent_walk','hit_by_pitch']).astype(float)
pa['k']=pa.e.isin(['strikeout','strikeout_double_play']).astype(float)
pa['proxy']=pa.tb+0.60*pa.bbhbp
pa['is_start_pitcher']=[str(pid)==start.get((pk,t)) for pk,t,pid in zip(pa.game_pk,pa.pitching_team,pa.pitcher)]

games=[];bad=[]
for pk,q in pa.groupby('game_pk',sort=False):
 q=q.sort_values('at_bat_number',kind='stable')
 day=pd.Timestamp(q.game_date.iloc[0]);yr=day.year
 a=str(q.away_team.iloc[0]);h=str(q.home_team.iloc[0])
 f=q[q.inning.astype(int)<=5]
 if len(f)==0 or int(f.inning.max())<5:bad.append([str(pk),'no_f5']);continue
 fl=f.iloc[-1];last=q.iloc[-1]
 out={'away_f5':int(fl.post_away_score),'home_f5':int(fl.post_home_score),
      'away_full':int(last.post_away_score),'home_full':int(last.post_home_score)}
 lineups={}
 for t in [a,h]:
  first9=q[q.batting_team.eq(t)].drop_duplicates('batter',keep='first').head(9)
  if len(first9)!=9 or (first9.inning.astype(int)>4).any() or not first9.side.isin(['L','R']).all():break
  weights=np.array([1.12,1.09,1.06,1.04,1.02,1.00,.96,.92,.89]);weights/=weights.sum()
  lineups[t]=float(np.dot(weights,first9.side.eq('L').astype(float)))
 if len(lineups)!=2:bad.append([str(pk),'lineup']);continue
 games.append(dict(pk=str(pk),date=day,year=yr,away=a,home=h,stats=out,lineups=lineups,
                   starters={t:start[(pk,t)] for t in [a,h]}, pa=q))
assert len(games)>=1800,('insufficient RS games',len(games),bad[:5])
games.sort(key=lambda g:(g['date'],g['pk']))

team_hist=defaultdict(list)
pitch_side=defaultdict(lambda:np.zeros(2,dtype=float))
league_side=defaultdict(lambda:np.zeros(2,dtype=float))
records=[]
for day in sorted(set(g['date'] for g in games)):
 today=[g for g in games if g['date']==day]
 year=int(day.year)
 past_vals=[v for k,hist in team_hist.items() if k[0]==year for v in hist]
 lg5=float(np.mean([v['f5_for'] for v in past_vals])) if past_vals else 2.4
 lgF=float(np.mean([v['full_for'] for v in past_vals])) if past_vals else 4.8
 nl=sum(league_side[x][0] for x in ('L','R'))
 la=(sum(league_side[x][1] for x in ('L','R'))/nl) if nl else 0.75
 lmean={x:(league_side[x][1]/league_side[x][0]) if league_side[x][0]>=100 else la for x in ('L','R')}
 for g in today:
  for t,opp,is_home,sc1,sc2 in [(g['away'],g['home'],0,'away_f5','away_full'),(g['home'],g['away'],1,'home_f5','home_full')]:
   team=team_hist[(year,t)];de=team_hist[(year,opp)]
   def prior(h,col,lg,pseudo=12):
    return (sum(v[col] for v in h)+pseudo*lg)/(len(h)+pseudo)
   def recent(h,col,lg,pseudo=12):
    z=[v[col] for v in h if v['date']>=day-pd.Timedelta(days=30)]
    return (sum(z)+pseudo*lg)/(len(z)+pseudo)
   pid=g['starters'][opp]
   ln=g['lineups'][t]
   L=pitch_side[(pid,'L')];R=pitch_side[(pid,'R')]
   nL,nR=float(L[0]),float(R[0]);pN=nL+nR;pQ=float(L[1]+R[1])
   qL=(L[1]+120*lmean['L'])/(nL+120)
   qR=(R[1]+120*lmean['R'])/(nR+120)
   residual=(qL-qR)-(lmean['L']-lmean['R'])
   historic_mix=nL/pN if pN>0 else float(league_side['L'][0]/nl) if nl else .5
   interact=(ln-historic_mix)*residual
   overall=(pQ+200*la)/(pN+200)-la
   for typ,off0,off1,def0,def1,lg in [
    ('f5','f5_for','f5_against','f5_against','f5_for',lg5),
    ('full','full_for','full_against','full_against','full_for',lgF)]:
    records.append({'pk':g['pk'],'date':str(day.date()),'year':year,'type':typ,'is_home':is_home,
      'team':t,'opp':opp,'starter':pid,'y':int(g['stats'][sc1 if typ=='f5' else sc2]),
      'base_off':prior(team,off0,lg)/lg,'base_off30':recent(team,off0,lg)/lg,
      'base_def':prior(de,def0,lg)/lg,'base_def30':recent(de,def0,lg)/lg,
      'pitch_quality':overall,'left_frac':ln,'platoon_shift':interact,
      'prior_pitcher_BF':pN,'prior_left_BF':nL,'prior_right_BF':nR,
      'lg':lg,'lineup_reconstruction':'FIRST9_BATTERS_BY_4TH_INNING_NOT_PREANNOUNCED'})
 for g in today:
  for team,opp,side,other in [(g['away'],g['home'],'away','home'),(g['home'],g['away'],'home','away')]:
   team_hist[(year,team)].append({'date':day,
    'f5_for':g['stats'][side+'_f5'],'f5_against':g['stats'][other+'_f5'],
    'full_for':g['stats'][side+'_full'],'full_against':g['stats'][other+'_full']})
  q=g['pa'];p=q[q.is_start_pitcher & q.side.isin(['L','R'])]
  if not p.empty:
   sums=p.groupby(['pitcher','side']).agg(n=('proxy','size'),q=('proxy','sum'))
   for (pid,side),row in sums.iterrows():
    pitch_side[(str(pid),str(side))] += [float(row['n']),float(row['q'])]
    league_side[str(side)] += [float(row['n']),float(row['q'])]

frame=pd.DataFrame(records)
assert frame.groupby(['pk','type']).size().eq(2).all(), 'pair integrity'
assert frame.platoon_shift.notna().all()
features=['base_off','base_off30','base_def','base_def30','is_home','pitch_quality','left_frac']
extra=['platoon_shift']
sets=[{'name':'2024train_2025test','train':[2024],'test':2025,'test_end':'2025-09-30'},
      {'name':'2024_2025train_2026test','train':[2024,2025],'test':2026,'test_end':'2026-10-12'}]
res={'status':'RESEARCH_OOS_ONLY_NOT_CHAMPION','source_sha26':sha26,
     'source_24_25_sha':hashlib.sha256(SRC24.read_bytes()).hexdigest(),
     'valid_games':len(games),'bad_game_count':len(bad),
     'source_starter_cutoff':'all historic pitcher appearances from STRICTLY earlier dates; only starting-pitcher PA',
     'lineups':'retrospective first 9 distinct batters by 4th inning; not timestamped pregame',
     'feature_definition':'(current_lineup_left_fraction - historic_starter_left_fraction) * ((shrunk_starter_left_proxy - shrunk_starter_right_proxy) - league_left_right_proxy_difference)',
     'proxy':'total_bases + 0.6*(BB+HBP) per starter BF; L,R each 120BF league pseudo; overall 200BF',
     'models':'frozen baseline + ONE predeclared pitcher-specific interaction feature; identical dates & teams, PoissonRegressor alpha=.1',
     'sets':{},'warnings':['No full official Champion reconstruction or official betting EV.','Postgame lineup-reconstruction proxy can select substitutions and is NOT a pre-timestamped published starting nine.','2026 test not independent prospective as 2026 previously investigated.','Original PS outcomes and target same-game statistics not predictors.']}
rng=np.random.default_rng(20261106)
for setup in sets:
 out={}
 for typ in ['f5','full']:
  z=frame[frame.type==typ]
  tr=z[z.year.isin(setup['train']) & (z.date.str[5:]>='05-01') & (z.date.str[5:]<='09-30')].copy()
  te=z[(z.year==setup['test'])&(z.date.str[5:]>='05-01')&(z.date<=setup['test_end'])].copy()
  assert len(tr)>700 and len(te)>600,(setup,typ,len(tr),len(te))
  evals={};errs={}
  for model,cols in [('baseline',features),('platoon',features+extra)]:
   sx=StandardScaler();Xtr=sx.fit_transform(tr[cols].to_numpy(float));Xte=sx.transform(te[cols].to_numpy(float))
   m=PoissonRegressor(alpha=.1,max_iter=1000,tol=1e-8)
   m.fit(Xtr,tr.y.to_numpy(float));mu=np.clip(m.predict(Xte),.02,25.)
   evals[model]={'pred':mu,'loss':-poisson.logpmf(te.y.to_numpy(int),mu),
     'coefs':dict(zip(cols,np.round(m.coef_,7).tolist()))}
  gl=pd.DataFrame({'pk':te.pk.to_numpy(),'home':te.is_home.to_numpy(), 'y':te.y.to_numpy(),
     'baseline':evals['baseline']['pred'],'platoon':evals['platoon']['pred'],
     'baseline_loss':evals['baseline']['loss'],'platoon_loss':evals['platoon']['loss']})
  gd=gl.groupby('pk').apply(lambda x:float((x.platoon_loss-x.baseline_loss).mean()),include_groups=False).to_numpy()
  bs=np.mean(rng.choice(gd,size=(2000,len(gd)),replace=True),axis=1)
  o={'train_team_games':len(tr),'test_games':gl.pk.nunique(),'test_team_games':len(te),
     'avg_platoon_shift':round(float(te.platoon_shift.mean()),7),
     'test_abs_avg_platoon_shift':round(float(te.platoon_shift.abs().mean()),7),
     'platoon_nonzero_share':round(float((te.platoon_shift.abs()>1e-9).mean()),5),
     'delta_score_nll_platoon_minus_baseline':round(float(gd.mean()),7),
     'delta_score_nll_bootstrap95':[round(float(x),7) for x in np.quantile(bs,[.025,.975])],
     'models':{}}
  for model,e in evals.items():
   wide=gl.pivot(index='pk',columns='home',values=model)
   actual=gl.pivot(index='pk',columns='home',values='y')
   h=wide[1].to_numpy();a=wide[0].to_numpy();actualdiff=(actual[1]-actual[0]).to_numpy()
   pred_wdl=np.c_[skellam.cdf(-1,h,a),skellam.pmf(0,h,a),1-skellam.cdf(0,h,a)]
   idx=np.where(actualdiff>0,2,np.where(actualdiff<0,0,1))
   wdl=-np.log(np.clip(pred_wdl[np.arange(len(idx)),idx],1e-10,1))
   brier=float(np.mean(np.sum((pred_wdl-np.eye(3)[idx])**2,axis=1)/3))
   o['models'][model]={'score_nll':round(float(e['loss'].mean()),7),
       'WDL_logloss':round(float(wdl.mean()),7), 'WDL_brier':round(brier,7),
       '3plus_brier':round(float(np.mean(((1-poisson.cdf(2,e['pred']))-(te.y.to_numpy()>=3))**2)),7),
       'pred_runs_mean':round(float(np.mean(e['pred'])),5),
       'actual_runs_mean':round(float(np.mean(te.y)),5),
       'platoon_coefficient_standardized':e['coefs'].get('platoon_shift')}
   errs[model]=wdl
  out[typ]=o
 res['sets'][setup['name']]=out
 print('PLATOON_OOS_PASS='+json.dumps({setup['name']:out},ensure_ascii=False),flush=True)

pre=pa[(pa.game_date<pd.Timestamp('2026-11-06'))&(pa.game_date.dt.year==2026)&(pa.pitcher_name.astype(str)=='소형준')&pa.side.isin(['L','R'])].copy()
def exact_ops(x):
 e=x.e
 h=int(e.isin(['single','double','triple','home_run']).sum())
 bb=int(e.isin(['walk','intent_walk']).sum());hbp=int(e.eq('hit_by_pitch').sum());sf=int(e.eq('sac_fly').sum())
 ab=int(e.isin(['single','double','triple','home_run','field_out','strikeout','strikeout_double_play','fielders_choice','double_play','triple_play','field_error']).sum())
 tb=float(x.tb.sum())
 obp=(h+bb+hbp)/(ab+bb+hbp+sf) if ab+bb+hbp+sf else None
 slg=tb/ab if ab else None
 return {'PA':len(x),'AB_classified':ab,'H':h,'BB':bb,'HBP':hbp,'SF':sf,'TB':tb,'OBP':obp,'SLG':slg,'OPS':obp+slg if obp is not None and slg is not None else None}
res['so_hyung_jun']={s:exact_ops(pre[pre.side==s]) for s in ['L','R']}
res['so_hyung_jun']['source_PAs']=len(pre)
ROOT.mkdir(parents=True,exist_ok=True)
(ROOT/'platoon_oos_result.json').write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n')
print('PLATOON_RESULT_SUMMARY='+json.dumps({'sets':res['sets'],'so_hyung_jun':res['so_hyung_jun'],'count':res['valid_games']},ensure_ascii=False),flush=True)
