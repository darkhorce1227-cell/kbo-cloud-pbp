"""KS2 2026-11-04 explicit market-blind research PRE side-by-side mu-v1.2 and mu-v1.3.
Price-blind input: frozen validated PBP through 2026-11-03.
Not Champion, not Official CLEAN PRE LOCK.
"""
from __future__ import annotations
from pathlib import Path
import sys,json,hashlib,math
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path.cwd()/"scripts"))
import build_mu_v1_2_features as b
import run_mu_v1_2_pipeline as runner
import mu_v1_3 as m
SRC=Path("/tmp/ks2/kbo_pbp_2026_latest.parquet")
OUT=Path("/tmp/ks2/ks2_premarket_20261104.json")
SHA="32c70d31c89c83a9e8606254ff6ec084f599371b6debc6dde36066bf122c74e1"
assert hashlib.sha256(SRC.read_bytes()).hexdigest()==SHA,"DATA_HASH_MISMATCH"
raw=pd.read_parquet(SRC)
assert len(raw)==224611
raw['game_date']=pd.to_datetime(raw.game_date)
D=pd.Timestamp('2026-11-04')
assert (raw.game_date<D).all(),"FUTURE_LEAKAGE"
gid="77771104SSKT02026"
pa=runner.prepare_pa_fixed(raw)
team,st,bp=b.game_tables(pa)
va=b.build_feature_row(team,st,bp,d=D,team="SS",opponent="KT",opposing_starter="64001",is_home=0)
vh=b.build_feature_row(team,st,bp,d=D,team="KT",opponent="SS",opposing_starter="56459",is_home=1)
spec12=json.loads(Path("research/mu-v1.2/model_spec.json").read_text())
mu12=b.score_mu(va,vh,spec12)
b.load_seasons=lambda year: raw.copy() if year==2026 else (_ for _ in ()).throw(ValueError("ONLY_2026_VALIDATED"))
t13,s13,b13=m.load_all(2026)
cfg=m.config()
a13=m.build_features(t13,s13,b13,d=D,team="SS",opponent="KT",opposing_starter="64001",is_home=0,cfg=cfg)
h13=m.build_features(t13,s13,b13,d=D,team="KT",opponent="SS",opposing_starter="56459",is_home=1,cfg=cfg)
spec13=json.loads(Path("research/mu-v1.3/model_spec.json").read_text())
mu13=m.score(spec13,a13,h13)
for name,mu in [('v12',mu12),('v13',mu13)]:
 for side in ('home','away'):
  assert 0<mu['f5_'+side]<mu['full_'+side]<16,(name,mu)
def model_draw(mu,seed):
 n=700000
 rng=np.random.default_rng(seed)
 nb=lambda mean,alpha:rng.negative_binomial(1./alpha,1./(1+alpha*mean),size=n).astype(np.int16)
 a5=nb(mu['f5_away'],.54); h5=nb(mu['f5_home'],.54)
 al=nb(mu['full_away']-mu['f5_away'],.322)
 hl=nb(mu['full_home']-mu['f5_home'],.322)
 a=a5+al; h=h5+hl
 tie9=float(np.mean(a==h))
 for inning in range(10,16):
  ix=np.flatnonzero(a==h)
  if len(ix)==0:break
  a[ix]+=rng.poisson((mu['full_away']-mu['f5_away'])/4.,len(ix)).astype(np.int16)
  h[ix]+=rng.poisson((mu['full_home']-mu['f5_home'])/4.,len(ix)).astype(np.int16)
 d=h-a; d5=h5-a5; total=h+a;t5=h5+a5
 prob=lambda mask:round(float(np.mean(mask)),6)
 f5={'KT':prob(d5>0),'Draw':prob(d5==0),'Samsung':prob(d5<0)}
 full={'KT':prob(d>0),'Draw_after_15':prob(d==0),'Samsung':prob(d<0),'Tied_after_9':round(tie9,6)}
 tota={k:{str(x+.5):{'U':prob(y<=x),'O':prob(y>x)} for x in v} for k,y,v in [('F5',t5,range(2,15)),('Full',total,range(3,19))]}
 handicap={k:{str(x+.5):{'KT_minus':prob(y>=x+1),'Samsung_plus':prob(y<=x)} for x in (0,1,2,3)} for k,y in [('F5',d5),('Full',d)]}
 win1loss={'KT_by_2plus':prob(d>=2),'One_run_or_tie':prob(abs(d)<=1),'Samsung_by_2plus':prob(d<=-2)}
 margin={**{f'KT_by_{x}':prob(d==x) for x in range(1,7)},'KT_by_7plus':prob(d>=7),**{f'Samsung_by_{x}':prob(d==-x) for x in range(1,7)},'Samsung_by_7plus':prob(d<=-7),'Draw_after_15':prob(d==0)}
 doub={}
 fordummy=0
 for name1,flag1 in [('KT',d5>0),('Draw',d5==0),('Samsung',d5<0)]:
  for name2,flag2 in [('KT',d>0),('Samsung',d<0)]:
   doub[name1+'_'+name2]=prob(flag1&flag2)
 doub['Draw_final_after_15']=prob(d==0)
 exact={str(i):prob(total==i) for i in range(1,20)}
 exact['20+']=prob(total>=20)
 exact['0_unquoted']=prob(total==0)
 assert abs(sum(f5.values())-1)<0.00001,f5
 assert abs(sum(full[k] for k in ('KT','Samsung','Draw_after_15'))-1)<0.00001,full
 assert abs(sum(win1loss.values())-1)<0.00001,win1loss
 assert abs(sum(exact.values())-1)<0.00005,exact
 assert abs(sum(doub.values())-1)<0.00003,doub
 assert abs(sum(margin.values())-1)<0.00004,margin
 return {'mu':mu,'f5':f5,'full':full,'total':tota,'handicap':handicap,'win1loss':win1loss,'margin':margin,'double':doub,'exact_total':exact,'pool_HR_score':{'status':'UNAVAILABLE_NO_CALIBRATED_HR_RUN_JOINT'},'simulation_paths':n,'status':'RESEARCH_ONLY_NOT_CHAMPION','nonproduction_limitations':['2026-only feature history instead of original 2024-25 priors','No confirmed playing lineup or current actual bullpen usage path integrated','No regression-verified PS-specific scoring adjustment','Extra innings 10-15 approximate independent Poisson, no walkoff/team strategy','No actual sportsbook odds used']}
stats={}
for pid in ("56459","64001"):
 q=st[st.starter.astype(str).eq(pid)].sort_values('game_date')
 stats[pid]={'name':"페덱" if pid=="56459" else "고영표",'starts_including_PS':len(q),'avg_IP':round(float(q.ip.mean()),4),'recent5_IP':round(float(q.tail(5).ip.mean()),4),'K_rate':round(float(q.k.sum()/q.pa.sum()),5),'BB_rate':round(float(q.bb.sum()/q.pa.sum()),5),'HR_rate':round(float(q.hr.sum()/q.pa.sum()),5),'last_game':str(q.game_date.max().date())}
out={'game_pk':gid,'date':'2026-11-04','away':'삼성','home':'KT','stadium':'수원','data_sha256':SHA,'pitch_rows':len(raw),'data_cutoff_exclusive':'2026-11-04','status':'RESEARCH_PRE_UNOFFICIAL_MARKET_UNSEEN','market_prices_used':False,'stats':stats,'models':{'v1.2_research':model_draw(mu12,20261104),'v1.3_research':model_draw(mu13,20261105)},'feature_v12':{'away':va,'home':vh},'feature_v13':{'away':a13,'home':h13},'lineup_status':'NOT_VERIFIED_OFFICIAL','source':'GitHub validated PBP through Nov3'}
OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print("KS2_PRE_SUMMARY="+json.dumps({'data_sha256':SHA,'stats':stats,'models':{name:{k:v[k] for k in ['mu','f5','full','win1loss']}|{'totals_key':{'F5_U5.5':v['total']['F5']['5.5']['U'],'F5_U4.5':v['total']['F5']['4.5']['U'],'Full_U9.5':v['total']['Full']['9.5']['U'],'Full_U8.5':v['total']['Full']['8.5']['U']},'f5_hcp':v['handicap']['F5']['1.5'],'full_hcp':v['handicap']['Full']['1.5']} for name,v in out['models'].items()}},ensure_ascii=False))
