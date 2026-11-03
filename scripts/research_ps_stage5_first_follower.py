"""Research-only: can one anticipate the FIRST post-starter pitcher using prior PS games?
A separate, immutable-time evaluation; 2016-2025 PS-screened pitch-level data.
No future-game (or same-calendar-day) games may enter candidate histories.
Do not confuse observed first-follower identity with pregame-available availability.
"""
from __future__ import annotations
from collections import defaultdict, Counter
from pathlib import Path
import json
import numpy as np
import pandas as pd

SRC=Path('/tmp/ps-work/postseason_screen_pass.parquet')
START=Path('historical_ps/ps_starter_appearances.csv')
OUT=Path('/tmp/ps-work/ps_stage5_first_follower.json')
raw=pd.read_parquet(SRC).reset_index(drop=True)
starts=pd.read_csv(START,dtype={'game_pk':str,'starter_id':str})
good=set(starts.game_pk.astype(str))
assert len(good)==143
raw['game_pk']=raw.game_pk.astype(str)
raw=raw[raw.game_pk.isin(good)].copy()
raw['_ix']=np.arange(len(raw))
raw['half']=raw.inning_topbot.astype(str).str.lower().replace({'bottom':'bot','b':'bot','t':'top'})
assert set(raw.half.unique())=={'top','bot'}
raw['team']=np.where(raw.half=='top',raw.home_team,raw.away_team)
raw['pitcher_id']=raw.pitcher.astype('string').str.strip()
assert raw.pitcher_id.notna().all()
raw['game_date']=pd.to_datetime(raw.game_date)
raw['season']=raw.season.astype(int)
raw['inning']=raw.inning.astype(int)
raw['at_bat_number']=raw.at_bat_number.astype(int)
raw['post_home_score']=raw.post_home_score.astype(int)
raw['post_away_score']=raw.post_away_score.astype(int)
# Unique pitcher appearance order for each team/game; faithful to pitch sequence.
first=raw.drop_duplicates(['game_pk','team','pitcher_id'],keep='first').copy()
first=first.sort_values('_ix')
first['order']=first.groupby(['game_pk','team']).cumcount()
first['role']=np.select([first.order==0,first.order==1],['SP','FIRST'],default='LATER')
first['entry_inning']=first.inning
first['pitcher_name']=first.pitcher_name.astype('string')
pitch_counts=raw.groupby(['game_pk','team','pitcher_id']).size().rename('pitches').reset_index()
first=first.merge(pitch_counts,on=['game_pk','team','pitcher_id'],validate='one_to_one')
# Actual score event / plate appearance map; no official ER assignment.
pa=raw.drop_duplicates(['game_pk','at_bat_number'],keep='last').sort_values('_ix').copy()
for side in ('home','away'):
 col='post_'+side+'_score'
 pa['scored_'+side]=pa[col]-pa.groupby('game_pk')[col].shift().fillna(0)
assert (pa[['scored_home','scored_away']]>=0).all().all()
pa['scored']=np.where(pa.half=='top',pa.scored_away,pa.scored_home)
pa=pa.merge(first[['game_pk','team','pitcher_id','order','role']],on=['game_pk','team','pitcher_id'],validate='many_to_one')
pa_count=pa.groupby(['game_pk','team','pitcher_id']).agg(pa=('scored','size'),runs=('scored','sum')).reset_index()
first=first.merge(pa_count,on=['game_pk','team','pitcher_id'],how='left',validate='one_to_one')
first[['pa','runs']]=first[['pa','runs']].fillna(0).astype(int)
first['year']=first.game_date.dt.year
# All key dates correspond to the group. No same-day ordering inference.
assert first.groupby(['game_pk','team']).apply(lambda x:(x.order==0).sum(),include_groups=False).eq(1).all()
assert first.groupby(['game_pk','team']).order.max().ge(0).all()
assert len(first[first.order==0])==286

# Team-game summary and per-team pitcher history available strictly before game date.
team_games=first.groupby(['season','team','game_date','game_pk'],sort=True)
observ=[]
for (year,team,date,gid),g in team_games:
 gg=g.sort_values('order');f=gg[gg.order==1]
 observ.append({'year':int(year),'team':str(team),'date':date,'game':gid,
               'first_id':str(f.pitcher_id.iloc[0]) if len(f) else None,
               'first_name':str(f.pitcher_name.iloc[0]) if len(f) else None,
               'first_entry':int(f.entry_inning.iloc[0]) if len(f) else None,
               'starter_id':str(gg.pitcher_id.iloc[0]),
               'all_roles':[{ 'id':str(r.pitcher_id),'role':str(r.role),'pitches':int(r.pitches),
                              'pa':int(r.pa),'runs':int(r.runs)} for r in gg.itertuples()]})
obs=sorted(observ,key=lambda r:(r['year'],r['date'],r['game'],r['team']))
assert len(obs)==286

methods=['last_first','most_first','any_recency','role_weighted','role_weighted_rest']
results=[]
for game in obs:
 prev=[z for z in obs if z['year']==game['year'] and z['team']==game['team'] and z['date']<game['date']]
 if not prev:continue
 used={}
 for j,past in enumerate(prev):
  for appearance in past['all_roles']:
   if appearance['role']=='SP':continue
   pid=appearance['id']
   x=used.setdefault(pid,dict(first=0,other=0,last=None,pitches_2d=0,pitches_3d=0,last_first=None,last_any=None))
   if appearance['role']=='FIRST':
    x['first']+=1
    x['last_first']=past['date']
   else:x['other']+=1
   x['last_any']=past['date']
   if (game['date']-past['date']).days<=2:x['pitches_2d']+=appearance['pitches']
   if (game['date']-past['date']).days<=3:x['pitches_3d']+=appearance['pitches']
 assert len(used)>0
 choices={}
 for method in methods:
  def score(pid,item):
   lf=item['last_first'];la=item['last_any'];
   if method=='last_first':return (int(lf.value) if lf is not None else -1, item['first'],pid)
   if method=='most_first':return (item['first'],int(lf.value) if lf is not None else -1,pid)
   if method=='any_recency':return (int(la.value) if la is not None else -1,item['first'],pid)
   w=2*item['first']+.5*item['other']
   if method=='role_weighted_rest':w=w/(1+0.02*item['pitches_2d'])
   return (w, int(la.value) if la is not None else -1,pid)
  choices[method]=sorted(used,key=lambda pid:score(pid,used[pid]),reverse=True)
 days_since_prev=(game['date']-max(z['date'] for z in prev)).days
 rec={'year':game['year'],'game':game['game'],'team':game['team'],'n_prior_team_games':len(prev),
      'days_since_last_team_game':days_since_prev,'true_id':game['first_id'],
      'true_first_entry':game['first_entry'],'n_candidate_prior_relief':len(used),
      'in_prior_roster':bool(game['first_id'] in used)}
 for meth in methods:
  rank=(choices[meth].index(game['first_id'])+1) if game['first_id'] in choices[meth] else None
  rec[meth+'_rank']=rank
 results.append(rec)

# Comparable paired tests (same eligible observations), evaluate all and 2021-25.
def summary(rows):
 ret={'team_games':len(rows),'first_actual':sum(r['true_id'] is not None for r in rows),
      'first_in_prior_relief_roster':sum(r['in_prior_roster'] for r in rows),
      'first_new_or_previous_starter':sum(r['true_id'] is not None and not r['in_prior_roster'] for r in rows),
      'prior_games_mean':round(float(np.mean([r['n_prior_team_games'] for r in rows])),3) if rows else None,
      'first_entry_by_inning':dict(sorted(Counter(str(r['true_first_entry']) for r in rows if r['true_first_entry'] is not None).items()))}
 for m in methods:
  ret[m]={'top1':sum(r[m+'_rank']==1 for r in rows),
          'top3':sum(r[m+'_rank'] is not None and r[m+'_rank']<=3 for r in rows),
          'top1_rate':round(sum(r[m+'_rank']==1 for r in rows)/max(1,len(rows)),5),
          'top3_rate':round(sum(r[m+'_rank'] is not None and r[m+'_rank']<=3 for r in rows)/max(1,len(rows)),5)}
 return ret

def boot_pair(rows,a,b,seed=20261103):
 # 12k game-day-cluster replicates (team games paired when same game).
 unique=sorted(set(r['game'] for r in rows));by={k:[r for r in rows if r['game']==k] for k in unique}
 d=np.array([sum(int(r[a+'_rank']==1)-int(r[b+'_rank']==1) for r in by[k]) for k in unique]);
 nn=np.array([len(by[k]) for k in unique]); rng=np.random.default_rng(seed)
 ix=rng.integers(len(unique),size=(12000,len(unique)))
 est=d.sum()/nn.sum(); dist=d[ix].sum(axis=1)/nn[ix].sum(axis=1)
 return {'top1_difference':round(float(est),6),'cluster_boot_ci95':[round(float(x),6) for x in np.quantile(dist,[.025,.975])]}

hold=[r for r in results if r['year']>=2021]
held_with2=[r for r in hold if r['n_prior_team_games']>=2]
# Start-of-game vs end-of-game label: we never utilize starter's actual IP or follower identity to rank candidates.
result={
 'status':'RESEARCH_PREGAME_FOLLOWER_IDENTITY_REPLAY','source':'historical-ps-master/postseason_screen_pass.parquet',
 'full_team_games':len(obs),'team_games_with_prior_same_year_team_game':len(results),
 'pregame_info_policy':'only earlier-dated games of SAME team and SAME postseason season; no same-calendar-day or future outcomes',
 'all_years':summary(results),'holdout_2021_2025':summary(hold),'holdout_min2_team_games':summary(held_with2),
 'holdout_by_year':{str(y):summary([r for r in hold if r['year']==y]) for y in range(2021,2026)},
 'holdout_paired_differences':{m:boot_pair(hold,m,'last_first') for m in methods if m!='last_first'},
 'first_entry_all':dict(sorted(Counter(str(r['first_entry']) for r in obs if r['first_entry'] is not None).items())),
 'first_appearance_missing':sum(r['first_id'] is None for r in obs),
 'limitations':[
 'The candidate roster consists solely of earlier PS relief appearances, not the actual active roster, so new activation/long reliever first appearances are uncaptured.',
 'Predictions condition on team having prior same-year PS games; no candidate exists in first game of each postseason team.',
 'Previous appearances are pitch-level observations, not verified availability/health or pitch-count caps.',
 'Top1 and top3 accuracy are descriptive, and ranking rules were designed with knowledge of broad prior results; holdout years are retrospective, not live.',
 'No claim of improved run expectancy/market Brier from identity prediction alone.']}
OUT.parent.mkdir(exist_ok=True,parents=True)
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=float)+'\n')
print('STAGE5_RESULT_JSON='+json.dumps(result,ensure_ascii=False,default=float))