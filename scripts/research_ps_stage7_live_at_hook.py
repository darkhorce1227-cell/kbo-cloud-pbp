"""PS first-follower live-at-hook selection using innings already known at the change.
Historical pregame model uses prior club postseason history only; the LIVE variants
add CURRENT follower-entry inning, which is unavailable before first pitch.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import json
from collections import Counter

raw=pd.read_parquet('/tmp/ps-work/postseason_screen_pass.parquet').reset_index(drop=True)
st=pd.read_csv('historical_ps/ps_starter_appearances.csv',dtype={'game_pk':str})
raw['game_pk']=raw.game_pk.astype(str)
raw=raw[raw.game_pk.isin(set(st.game_pk.astype(str)))].copy()
raw['_ix']=np.arange(len(raw));raw['half']=raw.inning_topbot.astype(str).str.lower().replace({'bottom':'bot','b':'bot','t':'top'})
raw['team']=np.where(raw.half=='top',raw.home_team,raw.away_team)
raw['game_date']=pd.to_datetime(raw.game_date);raw['year']=raw.game_date.dt.year
raw['pitcher_id']=raw.pitcher.astype('string').str.strip()
first=raw.drop_duplicates(['game_pk','team','pitcher_id'],keep='first').sort_values('_ix').copy()
first['order']=first.groupby(['game_pk','team']).cumcount()
first['role']=np.select([first.order==0,first.order==1],['SP','FIRST'],default='LATER')
first['entry']=pd.to_numeric(first.inning).astype(int)
assert len(first[first.order==0])==286
obs=[]
for (year,team,date,gid),g in first.groupby(['year','team','game_date','game_pk'],sort=True):
 g=g.sort_values('order');ff=g[g.order==1]
 obs.append({'year':int(year),'team':str(team),'date':date,'game':gid,
             'real_ff':str(ff.pitcher_id.iloc[0]) if len(ff) else None,
             'entry':int(ff.entry.iloc[0]) if len(ff) else None,
             'relievers':[{ 'id':str(z.pitcher_id), 'role':str(z.role), 'inning':int(z.entry)} for z in g.itertuples() if z.order>=1]})
assert len(obs)==286
methods=('baseline','entry_kernel','entry_bucket','entry_first_only')
rows=[]
for target in obs:
 if target['entry'] is None:continue
 prev=[q for q in obs if q['year']==target['year'] and q['team']==target['team'] and q['date']<target['date']]
 if not prev:continue
 candidates={}
 for h in prev:
  for a in h['relievers']:
   c=candidates.setdefault(a['id'],dict(first=0,later=0,starts=[],lastdate=pd.Timestamp('1900-01-01')))
   if a['role']=='FIRST':c['first']+=1
   else:c['later']+=1
   c['starts'].append((a['role'],a['inning']))
   c['lastdate']=max(c['lastdate'],h['date'])
 if not candidates:continue
 at=target['entry'];rankings={}
 for name in methods:
  def value(pid,c):
   basic=2*c['first']+.5*c['later'];start=c['starts']
   if name=='baseline':return (basic,c['lastdate'].value,pid)
   if name=='entry_kernel':
    weights=sum((2 if role=='FIRST' else .5)*np.exp(-abs(at-inn)/1.5) for role,inn in start)
    return (weights,c['lastdate'].value,pid)
   if name=='entry_bucket':
    bucket=lambda v:int(v<=5)
    matched=sum((2 if role=='FIRST' else .5) for role,inn in start if bucket(inn)==bucket(at))
    return (matched+.15*basic,c['lastdate'].value,pid)
   if name=='entry_first_only':
    matched=sum(2*np.exp(-abs(at-inn)/1.5) for role,inn in start if role=='FIRST')
    return (matched+.15*basic,c['lastdate'].value,pid)
  rankings[name]=sorted(candidates,key=lambda pid:value(pid,candidates[pid]),reverse=True)
 rec={'year':target['year'],'game':target['game'],'team':target['team'],'entry':at,'actual':target['real_ff'],
      'entered_before6':at<=5,'was_in_candidate_pool':target['real_ff'] in candidates}
 for name in methods:rec[name+'_rank']=(rankings[name].index(target['real_ff'])+1 if target['real_ff'] in rankings[name] else None)
 rows.append(rec)

def summarize(v):
 res={'n':len(v),'in_pool':sum(x['was_in_candidate_pool'] for x in v)}
 for m in methods:
  res[m]={'top1':sum(q[m+'_rank']==1 for q in v),'top3':sum(q[m+'_rank'] is not None and q[m+'_rank']<=3 for q in v),
          'top1_rate':round(sum(q[m+'_rank']==1 for q in v)/len(v),5) if v else None}
 return res
held=[x for x in rows if x['year']>=2021]
rng=np.random.default_rng(20261103)
gids=sorted(set(x['game'] for x in held));groups={g:[x for x in held if x['game']==g] for g in gids}
ci={}
for m in methods[1:]:
 d=np.array([sum(int(x[m+'_rank']==1)-int(x['baseline_rank']==1) for x in groups[k]) for k in gids],float)
 n=np.array([len(groups[k]) for k in gids]); inds=rng.integers(0,len(gids),size=(10000,len(gids)))
 draw=d[inds].sum(axis=1)/n[inds].sum(axis=1)
 ci[m]={'top1_delta':round(d.sum()/n.sum(),6),'ci95':[round(float(z),6) for z in np.quantile(draw,[.025,.975])]}
result={'status':'RESEARCH_LIVE_AT_HOOK_PREDICTION','n_total_eligible':len(rows),'holdout_2021_2025':summarize(held),
        'holdout_entry_before6':summarize([x for x in held if x['entered_before6']]),
        'holdout_entry_6plus':summarize([x for x in held if not x['entered_before6']]),
        'holdout_by_year':{str(y):summarize([x for x in held if x['year']==y]) for y in range(2021,2026)},
        'paired_boot_10000':ci,'critical_usage_rule':'Live-at-hook entry inning known when replacing the starter; NOT a pregame input and must NEVER be used to boost CLEAN PRE.',
        'limitations':['No current active-roster data or current lead/runners/stamina features.','Retrospective exploratory heuristics, not a prospectively frozen player-selection model.',
                       'No score distribution or EV test in this experiment.']}
path=Path('/tmp/ps-work/ps_stage7_live_follower.json');path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('STAGE7_RESULT_JSON='+json.dumps(result,ensure_ascii=False))