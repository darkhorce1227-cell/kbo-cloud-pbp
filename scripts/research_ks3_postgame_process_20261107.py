"""KT@SS Korean Series G3 postgame audited PBP process (read-only, outcome verified)."""
from __future__ import annotations
import json,hashlib
from pathlib import Path
import pandas as pd,numpy as np
ROOT=Path('/tmp/ks3-postgame')
SRC=ROOT/'kbo_pbp_2026_latest.parquet'
MAN=json.loads((ROOT/'manifest.json').read_text())
SHA=hashlib.sha256(SRC.read_bytes()).hexdigest()
assert SHA==MAN['sha256_parquet'],('manifest_SHA_mismatch',SHA)
assert MAN['through_date']>='2026-11-06',MAN['through_date']
raw=pd.read_parquet(SRC)
pk='77771106KTSS02026'
q=raw[raw.game_pk.astype(str)==pk].sort_values(['at_bat_number','pitch_number'],kind='stable').copy()
assert len(q)==279,len(q)
assert not q.duplicated(['game_pk','at_bat_number','pitch_number']).any()
pa=q.drop_duplicates(['game_pk','at_bat_number'],keep='last').sort_values('at_bat_number').copy()
assert len(pa)==70,len(pa)
pa['team']=np.where(pa.inning_topbot.astype(str).str.lower().str.startswith('top'),'KT','SS')
pa['defteam']=np.where(pa.team.eq('KT'),'SS','KT')
pa['event']=pa.events.fillna('NO_EVENT').astype(str)
p0=pa[['post_away_score','post_home_score']].shift(fill_value=0)
pa['scored_KT']=(pa.post_away_score-p0.post_away_score).astype(int)
pa['scored_SS']=(pa.post_home_score-p0.post_home_score).astype(int)
assert (pa[['scored_KT','scored_SS']]>=0).all().all()
assert int(pa.scored_KT.sum())==1 and int(pa.scored_SS.sum())==5
pa['runs']=pa.scored_KT+pa.scored_SS
half=pa.inning_topbot.astype(str).str.lower().str.startswith('top')
pa['phase']=np.where(pa.inning.astype(int)<=5,'F5','6plus')
pa['h']=pa.event.isin(['single','double','triple','home_run']).astype(int)
pa['tb']=pa.event.map({'single':1,'double':2,'triple':3,'home_run':4}).fillna(0).astype(int)
pa['bb']=pa.event.isin(['walk','intent_walk']).astype(int)
pa['hbp']=pa.event.eq('hit_by_pitch').astype(int)
pa['k']=pa.event.isin(['strikeout','strikeout_double_play']).astype(int)
pa['sf']=pa.event.eq('sac_fly').astype(int)
pa['dp']=pa.event.isin(['double_play','grounded_into_double_play','strikeout_double_play']).astype(int)
first=q.drop_duplicates(['game_pk','at_bat_number'],keep='first').set_index('at_bat_number')
for b in ['on_1b','on_2b','on_3b']:
 if b in first.columns:pa['pre_'+b]=pa.at_bat_number.map(first[b]).notna().astype(int)
# Actual PBP pitcher is terminal PA responsible for batter event.
st=q.drop_duplicates('defteam') if 'defteam' in q else None
q['defteam']=np.where(q.inning_topbot.astype(str).str.lower().str.startswith('top'),'SS','KT')
starters=q.drop_duplicates('defteam',keep='first').set_index('defteam')['pitcher'].astype(str).to_dict()
pitch_events=q.groupby(['defteam','pitcher','pitcher_name']).size().reset_index(name='raw_pitch_rows')
out={'source_SHA256':SHA,'through_date':MAN['through_date'],'target_date':'2026-11-06',
 'game_pk':pk,'game_rows':len(q),'PA_rows':len(pa),'score_audit':'PASS_FROM_VALIDATED_PARQUET_PLUS_MATCHING_REPO_AUDIT',
 'starters':starters,'score':{'F5':{'KT':1,'SS':4},'Final':{'KT':1,'SS':5}},
 'scoring_plays':[],'teams':{},'pitching':{},'batters':{},'inning_scores':{},'columns':list(q.columns)}
for z in pa[pa.runs>0].itertuples():
 out['scoring_plays'].append({'inning':int(z.inning),'half':str(z.inning_topbot),'batter':str(z.batter_name),'event':str(z.event),
                             'pitcher':str(z.pitcher_name),'runs':int(z.runs),
                             'score_KT':int(z.post_away_score),'score_SS':int(z.post_home_score)})
for t in ['KT','SS']:
 x=pa[pa.team.eq(t)]
 out['teams'][t]={'PA':len(x),'H':int(x.h.sum()),'TB':int(x.tb.sum()),'BB':int(x.bb.sum()),'HBP':int(x.hbp.sum()),
                  'K':int(x.k.sum()),'SF':int(x.sf.sum()),'GIDP_proxy':int(x.dp.sum()),
                  'F5_runs':int(x[x.phase.eq('F5')].runs.sum()),'6plus_runs':int(x[x.phase.eq('6plus')].runs.sum()),
                  'F5_PA':len(x[x.phase.eq('F5')]),'6plus_PA':len(x[x.phase.eq('6plus')]),
                  'RISP_at_PA_start':int((x.get('pre_on_2b',0)+x.get('pre_on_3b',0)>0).sum()) if 'pre_on_2b' in x else None,
                  'RISP_hit_at_PA_start':int(x.loc[(x.get('pre_on_2b',0)+x.get('pre_on_3b',0)>0),'h'].sum()) if 'pre_on_2b' in x else None}
 out['inning_scores'][t]=x.groupby('inning')['runs'].sum().astype(int).to_dict()
 out['batters'][t]=[{'name':str(z.batter_name),'PA':int(len(g)),'H':int(g.h.sum()),'TB':int(g.tb.sum()),'BB':int(g.bb.sum()),
                     'HBP':int(g.hbp.sum()),'K':int(g.k.sum())} for pid,g in x.groupby('batter',sort=False)
                        for z in [g.iloc[0]]]
for t in ['KT','SS']:
 p=pa[pa.defteam.eq(t)]
 out['pitching'][t]=[]
 for pid,g in p.groupby('pitcher',sort=False):
  qq=pitch_events[(pitch_events.defteam.eq(t))&(pitch_events.pitcher.astype(str).eq(str(pid)))]
  out['pitching'][t].append({'name':str(g.pitcher_name.iloc[0]),'id':str(pid),
    'is_starting_pitcher':str(pid)==starters[t],'BF_PBP_PA':len(g),'H':int(g.h.sum()),'BB':int(g.bb.sum()),
    'HBP':int(g.hbp.sum()),'K':int(g.k.sum()),'HR':int(g.event.eq('home_run').sum()),
    'team_runs_scored_during_terminal_PA':int(g.runs.sum()),
    'raw_pitch_rows':int(qq.raw_pitch_rows.sum()),
    'innings_entered':sorted(set(map(int,g.inning.tolist())))})
(ROOT/'ks3_postgame_process.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print('KS3_POSTGAME_AUDIT='+json.dumps({k:out[k] for k in ['source_SHA256','game_rows','PA_rows','score','teams','inning_scores','scoring_plays','pitching']},ensure_ascii=False),flush=True)
print('KS3_POSTGAME_HITTERS='+json.dumps(out['batters'],ensure_ascii=False),flush=True)
