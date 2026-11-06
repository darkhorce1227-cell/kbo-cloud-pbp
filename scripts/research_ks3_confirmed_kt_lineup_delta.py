"""READ-ONLY KS3 lineup comparison, strictly 2026 RS PBP. No official mu or coefficients."""
import hashlib,json
from pathlib import Path
import numpy as np, pandas as pd
p=Path('/tmp/kbo-lineup-ks3')
f=p/'kbo_pbp_2026_latest.parquet'
m=json.loads((p/'manifest.json').read_text())
h=hashlib.sha256(f.read_bytes()).hexdigest()
assert h==m['sha256_parquet'],(h,m['sha256_parquet'])
x=pd.read_parquet(f)
x=x[pd.to_datetime(x.game_date).le('2026-10-12')].copy()
assert x.game_pk.nunique()==720
x=x.sort_values(['game_pk','at_bat_number','pitch_number'],kind='stable').drop_duplicates(['game_pk','at_bat_number'],keep='last')
x=x[x.events.notna()].copy()
x['batting_team']=np.where(x.inning_topbot.astype(str).str.lower().str.startswith('top'),x.away_team,x.home_team)
x=x[x.batting_team.eq('KT')].copy()
e=x.events.astype(str)
x['h']=e.isin(['single','double','triple','home_run']).astype(int)
x['walk']=e.isin(['walk','intent_walk']).astype(int)
x['hbp']=e.eq('hit_by_pitch').astype(int)
x['sf']=e.eq('sac_fly').astype(int)
x['ab']=e.isin(['single','double','triple','home_run','field_out','strikeout','strikeout_double_play','fielders_choice','double_play','triple_play','field_error']).astype(int)
x['tb']=e.map({'single':1,'double':2,'triple':3,'home_run':4}).fillna(0).astype(int)
x['k']=e.isin(['strikeout','strikeout_double_play']).astype(int)
old=['최원준','김민혁','안현민','힐리어드','김현수','류현인','허경민','한승택','장준원']
new=['김민혁','안현민','김현수','힐리어드','장성우','김상수','류현인','조대현','장준원']
w=np.array([1.12,1.09,1.06,1.04,1.02,1.00,.96,.92,.89]);w=w/w.sum()
alln=set(old+new)
def summarize(q):
 ab=int(q.ab.sum());h=int(q.h.sum());bb=int(q.walk.sum());hbp=int(q.hbp.sum());sf=int(q.sf.sum());tb=int(q.tb.sum());n=len(q)
 obp=(h+bb+hbp)/(ab+bb+hbp+sf) if ab+bb+hbp+sf>0 else None
 slg=tb/ab if ab else None
 return {'PA':n,'AB':ab,'H':h,'BB':bb,'HBP':hbp,'SF':sf,'TB':tb,
   'AVG':round(h/ab,4) if ab else None,'OBP':round(obp,4) if obp is not None else None,
   'SLG':round(slg,4) if slg is not None else None,
   'OPS':round(obp+slg,4) if obp is not None and slg is not None else None,
   'K':int(q.k.sum())}
byname={}
for n in alln:
 q=x[x.batter_name.astype(str).eq(n)]
 # strict name exact; fallback only if missing to explicitly flagged
 if q.empty:
  opts=x.loc[x.batter_name.astype(str).str.contains(n,regex=False),'batter_name'].value_counts().to_dict()
  byname[n]={'status':'MISSING_EXACT','possible_matching_names':opts}
 else:
  byname[n]={'status':'MATCHED','season':summarize(q),
    'last30':summarize(q[pd.to_datetime(q.game_date).ge('2026-09-13')]),
    'hitter_ids':q.batter.astype(str).unique().tolist(),
    'bat_side_count':q.stand.value_counts().to_dict()}
def weighted(line):
 v=[byname[n].get('season',{}).get('OPS') for n in line]
 return round(float(np.dot(w,np.array(v))),5) if all(q is not None for q in v) else None
out={'source_sha256':h,'source_cutoff':'2026-10-12 RS only','status':'READ_ONLY_LINEUP_SCOUT_NOT_CALIBRATED_MU','lineup_ks2':old,'lineup_ks3':new,'players':byname,
     'weighted_season_OPS_ks2':weighted(old),'weighted_season_OPS_ks3':weighted(new),
     'change_ks3_minus_ks2':round(weighted(new)-weighted(old),5) if weighted(old) is not None and weighted(new) is not None else None,
     'warnings':['Weighted OPS is an uncalibrated scouting comparator, not team run scoring or win probability',
       '2026 RS contains plate appearances against all types of pitchers and ballparks; not specific to Hurado',
       'Catching/fielding defense, pitcher framing/calling and injuries are omitted',
       'Do NOT apply numerical score or win correction from this unvalidated descriptive comparator']}
p.mkdir(parents=True,exist_ok=True);(p/'lineup_ks3_vs_ks2.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print('KS3_LINEUP_AUDIT='+json.dumps({'weighted_OPS_ks2':out['weighted_season_OPS_ks2'],'weighted_OPS_ks3':out['weighted_season_OPS_ks3'],
'delta':out['change_ks3_minus_ks2'],'players':{n:v.get('season') for n,v in byname.items()}},ensure_ascii=False))
