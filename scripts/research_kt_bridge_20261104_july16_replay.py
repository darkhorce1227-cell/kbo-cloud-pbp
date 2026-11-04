from pathlib import Path
import pandas as pd
import numpy as np
import hashlib, json
p=Path('/tmp/kt-bridge-verify/kbo_pbp_2026_latest.parquet')
expected='32c70d31c89c83a9e8606254ff6ec084f599371b6debc6dde36066bf122c74e1'
assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
x=pd.read_parquet(p)
assert len(x)==224611
x=x[x.game_pk.str.startswith('2026')].sort_values(['game_pk','at_bat_number','pitch_number'],kind='stable').copy()
assert x.game_pk.nunique()==720
keys=['game_pk','at_bat_number']
first=x.drop_duplicates(keys,keep='first').set_index(keys)
pa=x.drop_duplicates(keys,keep='last').copy().reset_index(drop=True)
idx=pd.MultiIndex.from_frame(pa[keys])
for c in ['home_score','away_score','outs_when_up','on_1b','on_2b','on_3b']:
 pa['first_'+c]=idx.map(first[c])
pa['date']=pd.to_datetime(pa.game_date)
pa['pitching_team']=np.where(pa.inning_topbot.eq('top'),pa.home_team,pa.away_team)
pa['pre_home']=pa.groupby('game_pk').post_home_score.shift().fillna(0).astype(int)
pa['pre_away']=pa.groupby('game_pk').post_away_score.shift().fillna(0).astype(int)
pa['runs']=pa.post_home_score+pa.post_away_score-pa.pre_home-pa.pre_away
same_half=pa.game_pk.eq(pa.game_pk.shift(-1))&pa.inning.eq(pa.inning.shift(-1))&pa.inning_topbot.eq(pa.inning_topbot.shift(-1))
same_game=pa.game_pk.eq(pa.game_pk.shift(-1))
walkoff_last=~same_game&pa.inning_topbot.eq('bot')&pa.post_home_score.gt(pa.post_away_score)
endouts=np.where(same_half,pa.first_outs_when_up.shift(-1),np.where(walkoff_last,pa.post_outs,3))
pa['outs']=endouts-pa.first_outs_when_up
assert pa.outs.between(0,3).all() and pa.runs.ge(0).all()
sx=x.assign(pitching_team=np.where(x.inning_topbot.eq('top'),x.home_team,x.away_team))
sm=sx.groupby(['game_pk','pitching_team'],sort=False).first()['pitcher'].to_dict()
pa['starter']=[sm[(a,b)] for a,b in zip(pa.game_pk,pa.pitching_team)]
pa['is_sp']=pa.pitcher.eq(pa.starter)
park='52060'
pk_ab=pa[pa.pitcher.eq(park)].groupby('game_pk').at_bat_number.min().to_dict()
pa['park_first_ab']=pa.game_pk.map(pk_ab).fillna(np.inf)
pa['role']=np.select([pa.is_sp,pa.pitcher.eq(park),pa.at_bat_number.lt(pa.park_first_ab)],['SP','PARK','BRIDGE'],default='AFTER_PARK')
pa['inherited']=pa[['first_on_1b','first_on_2b','first_on_3b']].notna().sum(axis=1)
stint_first=pa.groupby(['game_pk','pitching_team','pitcher'],sort=False)[['first_outs_when_up','inherited']].transform('first')
pa['clean_stint']=stint_first.first_outs_when_up.eq(0)&stint_first.inherited.eq(0)
kt=pa[(pa.pitching_team=='KT')&(pa.game_pk!='20260604LGKT02026')].copy()
assert kt.game_pk.nunique()==143
cut=pd.Timestamp('2026-07-16')
def ratio(z):
 outs=int(z.outs.sum());runs=int(z.runs.sum());return {'outs':outs,'ip':outs/3,'runs':runs,'runs_per_9IP':round(27*runs/outs,5) if outs else None,'pa':len(z)}
result={}
for part,d in [('first_half',kt[kt.date<cut]),('second_half',kt[kt.date>=cut])]:
 result[part]={'games':int(d.game_pk.nunique()),'all':{r:ratio(q) for r,q in d.groupby('role')},'clean':{r:ratio(q) for r,q in d[d.clean_stint].groupby('role')}}
result['all']={'games':int(kt.game_pk.nunique()),'all':{r:ratio(q) for r,q in kt.groupby('role')},'clean':{r:ratio(q) for r,q in kt[kt.clean_stint].groupby('role')}}
result['validation']={'source_sha256':expected,'regular_season_games':720,'KT_valid_games':143,'user_july16_split':result['first_half']['games']==82 and result['second_half']['games']==61,'half_total_games':result['first_half']['games']+result['second_half']['games']==143}
assert result['validation']['user_july16_split']
print('KT_JULY16_CLEAN_SPLIT_JSON='+json.dumps(result,ensure_ascii=False))
Path('/tmp/kt-bridge-verify/july16_clean_split.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
