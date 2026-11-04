from pathlib import Path
import pandas as pd,numpy as np,hashlib,json
src=Path("/tmp/ks2split/kbo_pbp_2026_latest.parquet")
sha="32c70d31c89c83a9e8606254ff6ec084f599371b6debc6dde36066bf122c74e1"
assert hashlib.sha256(src.read_bytes()).hexdigest()==sha
raw=pd.read_parquet(src)
print("KS2_SPLIT_COLUMNS="+json.dumps(raw.columns.tolist(),ensure_ascii=False))
assert raw.game_date.max()<'2026-11-04'
r=raw.drop_duplicates(['game_pk','at_bat_number'],keep='last').copy()
keys=["pitcher","pitcher_name","batter","batter_name","stand","batter_hand","batter_side","events","event","game_date","game_pk","home_team","away_team"]
print("KS2_PA_COLUMNS="+json.dumps([k for k in keys if k in r.columns],ensure_ascii=False))
for pid in ["64001","56459"]:
 q=r[r.pitcher.astype(str).eq(pid)].copy()
 cside=next((c for c in ["stand","batter_hand","batter_side","bats"] if c in q.columns),None)
 cname=next((c for c in ["pitcher_name","pitcherName"] if c in q.columns),None)
 col=next((c for c in ["events","event"] if c in q.columns),None)
 out={"pitcher_id":pid,"pa":len(q),"name":q[cname].iloc[-1] if cname and len(q) else None,"side_column":cside,"event_column":col}
 if cside and col and len(q):
  g=q.groupby(cside)
  abevents=['single','double','triple','home_run','field_out','strikeout','grounded_into_double_play','force_out','field_error','fielders_choice','fielders_choice_out','sac_fly','sac_bunt','double_play','triple_play','strikeout_double_play']
  hit=['single','double','triple','home_run']
  h=[]
  for side,x in g:
   ev=x[col].astype(str)
   ab=int(ev.isin(abevents).sum())
   hh=int(ev.isin(hit).sum())
   so=int(ev.eq('strikeout').sum())
   bb=int(ev.isin(['walk','intent_walk']).sum())
   hr=int(ev.eq('home_run').sum())
   h.append({"bat_side":str(side),"pa":len(x),"ab_reconstructed":ab,"hits":hh,"avg":round(hh/ab,4) if ab else None,"strikeouts":so,"walks":bb,"hr":hr,"hit_rate_pa":round(hh/len(x),4)})
  out["splits"]=h
 print("KS2_PITCHER_SIDE_SPLIT="+json.dumps(out,ensure_ascii=False))
