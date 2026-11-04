from __future__ import annotations
import hashlib,json
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np

SOURCE=Path("/tmp/ks1post/kbo_pbp_2026_latest.parquet")
STATUS=Path("state/status.json")
AUDIT=Path("state/postgame_score_audit_2026-11-03.json")
OUT=Path("/tmp/ks1post/ks1_postgame_20261103.json")
GAME="77771103SSKT02026"
stat=json.loads(STATUS.read_text(encoding="utf8"))
audit=json.loads(AUDIT.read_text(encoding="utf8"))
sha=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
assert sha==stat["sha256_parquet"]==audit["sha256_parquet"]
assert audit["target_date"]=="2026-11-03" and audit["validation"]=="PASS"
raw=pd.read_parquet(SOURCE)
x=raw[raw.game_pk.astype(str)==GAME].copy()
assert len(x)==audit["games"][0]["raw_rows"]==321
x["_r"]=np.arange(len(x))
x=x.sort_values(["at_bat_number","pitch_number","_r"],kind="stable")
need=["inning","inning_topbot","at_bat_number","pitcher","post_home_score","post_away_score","outs_when_up","post_outs"]
assert all(c in x for c in need),set(need)-set(x.columns)
x["half"]=np.where(x.inning_topbot.astype(str).str.lower().str.startswith("top"),"top","bot")
x["def_team"]=np.where(x["half"]=="top","KT","SS")
x["batting_team"]=np.where(x["half"]=="top","SS","KT")
x["pitcher_id"]=x.pitcher.astype(str)
roles=x.drop_duplicates(["def_team","pitcher_id"]).copy()
roles["order"]=roles.groupby("def_team").cumcount()
roles=roles[["def_team","pitcher_id","order","pitcher_name"]].rename(columns={"pitcher_name":"role_pitcher_name"})
pa=x.drop_duplicates("at_bat_number",keep="last").copy().sort_values(["at_bat_number","pitch_number","_r"])
assert len(pa)==audit["games"][0]["deduplicated_pa_rows"]==80
for col in ["post_home_score","post_away_score","inning","outs_when_up","post_outs"]:
 pa[col]=pd.to_numeric(pa[col],errors="raise").astype(int)
pa["home_run_delta"]=pa.post_home_score.diff().fillna(pa.post_home_score).astype(int)
pa["away_run_delta"]=pa.post_away_score.diff().fillna(pa.post_away_score).astype(int)
assert pa[["home_run_delta","away_run_delta"]].ge(0).all().all()
pa["runs"]=np.where(pa.half=="top",pa.away_run_delta,pa.home_run_delta)
pa["other_runs"]=np.where(pa.half=="top",pa.home_run_delta,pa.away_run_delta)
assert int(pa.other_runs.sum())==0
pa["outs_recorded"]=pa.post_outs-pa.outs_when_up
assert pa.outs_recorded.between(0,3).all()
assert int(pa.home_run_delta.sum())==8 and int(pa.away_run_delta.sum())==7
pa=pa.merge(roles,on=["def_team","pitcher_id"],how="left",validate="many_to_one")
assert pa["order"].notna().all()
pa["role"]=np.select([pa["order"]==0,pa["order"]==1],["STARTER","FIRST_FOLLOWER"],default="LATER_RELIEF")
pitch_counts=x.groupby(["def_team","pitcher_id"],as_index=False).size().rename(columns={"size":"raw_pitch_rows"})
gg=pa.groupby(["def_team","pitcher_id","pitcher_name","role","order"],as_index=False).agg(
    outs=("outs_recorded","sum"),pa=("runs","size"),runs_on_watch=("runs","sum"),
    f5_runs_on_watch=("runs",lambda v:int(v[pa.loc[v.index,"inning"].le(5)].sum())),
    first_inning=("inning","min"),last_inning=("inning","max")
)
gg=gg.merge(pitch_counts,on=["def_team","pitcher_id"],how="left",validate="one_to_one")
gg["actual_ip"]=(gg["outs"]/3).round(6)
gg=gg.sort_values(["def_team","order"])
def clean_df(df):
  return json.loads(df.replace({np.nan:None}).to_json(orient="records",force_ascii=False))
innings=pa.groupby(["inning","batting_team"],as_index=False)["runs"].sum()
inning_result=[{"inning":i,"SS":int(innings.loc[(innings.inning==i)&(innings.batting_team=="SS"),"runs"].sum()),"KT":int(innings.loc[(innings.inning==i)&(innings.batting_team=="KT"),"runs"].sum())} for i in range(1,10)]
assert sum(v["SS"] for v in inning_result)==7 and sum(v["KT"] for v in inning_result)==8
assert sum(v["SS"] for v in inning_result if v["inning"]<=5)==0
assert sum(v["KT"] for v in inning_result if v["inning"]<=5)==6
hr_events=[]
if "event" in pa.columns:
  hr=pa.loc[pa.event.astype(str).eq("home_run")]
  hr_events=clean_df(hr[[c for c in ["inning","half","batting_team","batter_name","pitcher_name","event","runs"] if c in hr.columns]])
event_cols=[c for c in ["event","events","description","pitch_result","result","type"] if c in x.columns]
scoring=pa.loc[pa.runs.gt(0)]
scoring_records=clean_df(scoring[[c for c in ["inning","half","batting_team","runs","batter_name","pitcher_name","event","events","post_home_score","post_away_score"] if c in pa.columns]])
sp=gg.loc[gg.role=="STARTER"]
assert len(sp)==2
# IMPORTANT: actual outs can be shifted on inherited and mid-PA events; verify values against official box.
assert set(sp.def_team)=={"SS","KT"}
assert int(sp.loc[sp.def_team=="SS","outs"].iloc[0])==6
assert int(sp.loc[sp.def_team=="KT","outs"].iloc[0])==18
reconstructed={
 "date":"2026-11-03","game_pk":GAME,"status":"PASS_SCORE_AUDIT_PITCHER_ON_WATCH",
 "source_rows_all":int(len(raw)),"source_rows_game":int(len(x)),"pa_rows_game":len(pa),"sha256":sha,
 "inning_runs":inning_result,"pitcher_path":clean_df(gg),
 "hr_pa_events":hr_events,"score_events":scoring_records,
 "audit":{"game":"PASS","f5_away":0,"f5_home":6,"full_away":7,"full_home":8,
 "score_partitions":int(gg.loc[gg.def_team=="SS","runs_on_watch"].sum())==8 and int(gg.loc[gg.def_team=="KT","runs_on_watch"].sum())==7,
 "starter_outs":clean_df(sp[["def_team","outs"]])},
 "disclaimers":["Runs-on-watch attributed to last pitcher in PA; not official R or ER or inherited-runner responsibility.","Pitch count is number of retained raw pitch rows; strict pitch classification pending validation.","Historical frozen pregame models must not be replaced by this outcome."],
 "columns":list(x.columns),
}
assert reconstructed["audit"]["score_partitions"]
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps(reconstructed,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
print("KS1_POSTGAME_SUMMARY="+json.dumps({k:reconstructed[k] for k in ["status","sha256","source_rows_game","pa_rows_game","inning_runs","pitcher_path","hr_pa_events","score_events","audit"]},ensure_ascii=False))
