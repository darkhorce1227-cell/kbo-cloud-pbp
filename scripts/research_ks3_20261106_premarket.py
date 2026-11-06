"""KS3 2026-11-06 market-blind research PRE; NOT Operational Champion."""
import sys,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,"scripts")
import build_mu_v1_2_features as b,run_mu_v1_2_pipeline as runner,mu_v1_3 as m
from mu_phase_invariants import assert_phase_consistent,phase_issues
P=Path("/tmp/kbo-ks3/kbo_pbp_2026_latest.parquet");OUT=Path("/tmp/kbo-ks3/ks3_premarket_20261106.json")
SHA="0ca95818ac8ea00030f79d8f5aa9c98987acb4edd334a8228ef9fb3503e12d4e"
assert hashlib.sha256(P.read_bytes()).hexdigest()==SHA
raw=pd.read_parquet(P);assert len(raw)==224975
raw.game_date=pd.to_datetime(raw.game_date);D=pd.Timestamp("2026-11-06")
assert raw.game_date.max()==pd.Timestamp("2026-11-04") and (raw.game_date<D).all()
pa=runner.prepare_pa_fixed(raw);tg,st,bp=b.game_tables(pa)
a12=b.build_feature_row(tg,st,bp,d=D,team="KT",opponent="SS",opposing_starter="53375",is_home=0)
h12=b.build_feature_row(tg,st,bp,d=D,team="SS",opponent="KT",opposing_starter="50030",is_home=1)
sp12=json.loads(Path("research/mu-v1.2/model_spec.json").read_text())
mu12=b.score_mu(a12,h12,sp12)
b.load_seasons=lambda year:raw.copy() if year==2026 else (_ for _ in ()).throw(ValueError("only 2026"))
t,s,b13=m.load_all(2026);cfg=m.config()
a13=m.build_features(t,s,b13,d=D,team="KT",opponent="SS",opposing_starter="53375",is_home=0,cfg=cfg)
h13=m.build_features(t,s,b13,d=D,team="SS",opponent="KT",opposing_starter="50030",is_home=1,cfg=cfg)
sp13=json.loads(Path("research/mu-v1.3/model_spec.json").read_text())
old13=m.score(sp13,a13,h13)
sa=m.raw_component(sp13["components"]["simple_late"],a13)
sh=m.raw_component(sp13["components"]["simple_late"],h13)
ca=m.raw_component(sp13["components"]["core_late"],a13)
ch=m.raw_component(sp13["components"]["core_late"],h13)
assert min(sa,sh,ca,ch)>0
share=ca/(ca+ch);late=sa+sh
mu13={**old13,"full_away":old13["f5_away"]+late*share,"full_home":old13["f5_home"]+late*(1-share)}
for z in (mu12,mu13):assert_phase_consistent(z)
def draw(mu,seed):
 rng=np.random.default_rng(seed);n=320000
 def nb(v,a):return rng.negative_binomial(1/a,1/(1+a*v),size=n).astype(np.int16)
 a5=nb(mu["f5_away"],.54);h5=nb(mu["f5_home"],.54)
 ar=mu["full_away"]-mu["f5_away"];hr=mu["full_home"]-mu["f5_home"]
 assert ar>=0 and hr>=0
 aa=a5+nb(ar,.322);hh=h5+nb(hr,.322);reg_tie=float(np.mean(aa==hh))
 for _ in range(6):
  ix=np.flatnonzero(aa==hh)
  if not len(ix):break
  aa[ix]+=rng.poisson(ar/4,len(ix)).astype(np.int16)
  hh[ix]+=rng.poisson(hr/4,len(ix)).astype(np.int16)
 d=hh-aa;d5=h5-a5;T=hh+aa;T5=h5+a5;p=lambda v:round(float(np.mean(v)),6)
 f5={"KT":p(d5<0),"Draw":p(d5==0),"SS":p(d5>0)}
 full={"KT":p(d<0),"SS":p(d>0),"draw_15":p(d==0),"reg_9_draw":round(reg_tie,6)}
 totals={segment:{str(k+.5):{"U":p(v<=k),"O":p(v>k)} for k in range(2,19)} for segment,v in (("F5",T5),("Full",T))}
 hcap={segment:{str(k+.5):{"SS_minus":p(diff>k),"KT_plus":p(diff<=k),"KT_minus":p(diff<(-k)),"SS_plus":p(diff>=(-k))} for k in range(0,5)} for segment,diff in (("F5",d5),("Full",d))}
 wl1={"SS_by_2plus":p(d>=2),"one_run_or_tie":p(abs(d)<=1),"KT_by_2plus":p(d<=-2)}
 margin={**{f"SS_{k}":p(d==k) for k in range(1,7)}, "SS_7plus":p(d>=7),**{f"KT_{k}":p(d==-k) for k in range(1,7)},"KT_7plus":p(d<=-7),"draw_15":p(d==0)}
 doubles={f"{x}_{y}":p((d5<0 if x=="KT" else d5>0 if x=="SS" else d5==0)&(d<0 if y=="KT" else d>0)) for x in ["KT","Draw","SS"] for y in ["KT","SS"]}
 doubles["draw_15"]=p(d==0)
 exact={str(k):p(T==k) for k in range(1,20)};exact["20plus"]=p(T>=20);exact["0unquoted"]=p(T==0)
 for obj in (f5,wl1,margin,doubles,exact):
  assert abs(sum(obj.values())-1)<.00006,(len(obj),sum(obj.values()))
 return {"mu":mu,"F5_WDL":f5,"Full_WL":full,"totals":totals,"handicap":hcap,
    "win1loss":wl1,"margin":margin,"F5_Full":doubles,"exact_total":exact,
    "HR_score_joint":"UNAVAILABLE_NO_CALIBRATED_HR_RUN_JOINT","simulations":n}
stats={}
for pid,name in [("50030","소형준"),("53375","후라도")]:
 q=st[st.starter.astype(str).eq(pid)].sort_values("game_date")
 assert len(q)>18,(name,len(q))
 stats[name]={"starts_including_PS":len(q),"avg_IP":round(float(q.ip.mean()),4),"last5_avg_IP":round(float(q.tail(5).ip.mean()),4),
    "K_pct":round(float(q.k.sum()/q.pa.sum())*100,2),"BB_pct":round(float(q.bb.sum()/q.pa.sum())*100,2),
    "HR_pct_PA":round(float(q.hr.sum()/q.pa.sum())*100,2),"last_game":str(q.game_date.max().date())}
res={"game_pk":"77771106KTSS02026","date":"2026-11-06","away":"KT","home":"SS","dataset_sha256":SHA,"pitch_rows":len(raw),
 "asof_data_through":"2026-11-04","market_prices_inspected":False,"official_champion":"UNAVAILABLE","confirmed_lineup":"NOT_YET_VERIFIED",
 "status":"MARKET_UNSEEN_RESEARCH_PRE_NOT_CHAMPION","starter_history":stats,"models":{
   "v1_2_research":draw(mu12,2026110601),"v1_3_coherent_research":draw(mu13,2026110602)},
 "v1_3_legacy_phase_issues":phase_issues(old13),"v1_3_legacy_mu":old13,
 "cautions":["2026-only features, source coefficients trained elsewhere","Unvalidated independent NB F5+late score covariance",
 "Unvalidated 10-15 inning extras approximation, no walkoff","Lineup and pitcher-path not numerically adjusted","No betting odds/EV, no Champion or official PRE LOCK"]}
OUT.write_text(json.dumps(res,ensure_ascii=False,indent=2)+"\n")
print("KS3_RESEARCH_PRE_SUMMARY="+json.dumps({
 "hash":SHA,"starters":stats,"models":{k:{"f5_mu":{"KT":v["mu"]["f5_away"],"SS":v["mu"]["f5_home"]},
 "full_mu":{"KT":v["mu"]["full_away"],"SS":v["mu"]["full_home"]},"F5":v["F5_WDL"],
 "Full":v["Full_WL"],"F5_O4p5":v["totals"]["F5"]["4.5"]["O"],
 "F5_O5p5":v["totals"]["F5"]["5.5"]["O"],"Full_O9p5":v["totals"]["Full"]["9.5"]["O"],
 "Full_O10p5":v["totals"]["Full"]["10.5"]["O"],"KT_F5_plus1p5":v["handicap"]["F5"]["1.5"]["KT_plus"]} for k,v in res["models"].items()},
 "legacy_v13_issues":res["v1_3_legacy_phase_issues"]},ensure_ascii=False))
