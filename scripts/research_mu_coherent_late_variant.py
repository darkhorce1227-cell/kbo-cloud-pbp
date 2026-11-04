"""One predeclared structural v1.3 alternative: F5 + nonnegative late-component allocation.
Exploratory test only, same 2026 heldout previously used to observe incoherence.
NO champion promotion or frozen model alteration.
"""
import sys,json,hashlib,math
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import poisson,nbinom
sys.path.insert(0,'scripts')
import build_mu_v1_2_features as b
import run_mu_v1_2_pipeline as runner
import mu_v1_3 as m
SRC=Path('/tmp/kbo-coherence/kbo_pbp_2026_latest.parquet')
SHA='32c70d31c89c83a9e8606254ff6ec084f599371b6debc6dde36066bf122c74e1'
assert hashlib.sha256(SRC.read_bytes()).hexdigest()==SHA
raw=pd.read_parquet(SRC);raw['game_date']=pd.to_datetime(raw.game_date)
pa=runner.prepare_pa_fixed(raw)
tg,st,bp=b.game_tables(pa)
m.base.load_seasons=lambda year:raw.copy() if year==2026 else (_ for _ in ()).throw(ValueError('only 2026'))
t,s,bp13=m.load_all(2026)
cfg=m.config();spec=json.loads(m.MODEL_PATH.read_text())
sp=st.set_index(['game_pk','pitching_team'])['starter'].astype(str).to_dict()
r=[];missing=[]
def score_3way_nll(a,h,actual_a,actual_h):
 alpha=.322
 ks=np.arange(45)
 def dist(mu):
  p=nbinom.pmf(ks,1/alpha,1/(1+alpha*mu));return p/p.sum()
 away=dist(a);home=dist(h)
 paway=float(np.sum(away[1:]*np.cumsum(home)[:-1]))
 draw=float(np.dot(away,home))
 phome=float(1-paway-draw)
 target='away' if actual_a>actual_h else 'home' if actual_h>actual_a else 'draw'
 p={'away':paway,'home':phome,'draw':draw}[target]
 return float(-np.log(max(p,1e-12)))
for pk,gg in tg[(tg.game_date>=pd.Timestamp('2026-07-16'))&(tg.game_date<=pd.Timestamp('2026-10-12'))].groupby('game_pk'):
 if len(gg)!=2:continue
 g=gg.sort_values('is_home');a=g.iloc[0];h=g.iloc[1];dt=a.game_date
 asid=sp.get((pk,a.team));hsid=sp.get((pk,h.team))
 if not asid or not hsid:missing.append(str(pk));continue
 try:
  fa=m.build_features(t,s,bp13,d=dt,team=a.team,opponent=h.team,opposing_starter=hsid,is_home=0,cfg=cfg)
  fh=m.build_features(t,s,bp13,d=dt,team=h.team,opponent=a.team,opposing_starter=asid,is_home=1,cfg=cfg)
 except Exception as ex:
  missing.append(str(pk)+':'+str(ex));continue
 orig=m.score(spec,fa,fh)
 sa=m.raw_component(spec['components']['simple_late'],fa)
 sh=m.raw_component(spec['components']['simple_late'],fh)
 ca=m.raw_component(spec['components']['core_late'],fa)
 ch=m.raw_component(spec['components']['core_late'],fh)
 assert min(sa,sh,ca,ch)>0
 late_total=sa+sh
 fraction=ca/(ca+ch)
 coa=orig['f5_away']+late_total*fraction
 coh=orig['f5_home']+late_total*(1-fraction)
 # Necessarily nonnegative late and same Full-total as source model.
 assert coa>=orig['f5_away'] and coh>=orig['f5_home']
 assert abs(coa+coh-orig['full_total'])<1e-7
 aa=int(a.runs_full);hh=int(h.runs_full)
 def loss(x,y):
  return [float((-poisson.logpmf(aa,x)-poisson.logpmf(hh,y))/2),score_3way_nll(x,y,aa,hh)]
 l0=loss(orig['full_away'],orig['full_home'])
 l1=loss(coa,coh)
 r.append({'pk':str(pk),'date':str(dt.date()),'away':str(a.team),'home':str(h.team),
           'actual_a':aa,'actual_h':hh,'orig_a':orig['full_away'],'orig_h':orig['full_home'],
           'coherent_a':coa,'coherent_h':coh,'orig_nll':l0[0],'coh_nll':l1[0],
           'orig_wdl_logloss':l0[1],'coh_wdl_logloss':l1[1],
           'orig_negative_late':orig['full_away']<orig['f5_away'] or orig['full_home']<orig['f5_home']})
df=pd.DataFrame(r);assert len(df)>250,(len(df),missing)
rng=np.random.default_rng(20261104)
out={'source_sha256':SHA,'status':'RESEARCH_EXPLORATORY_NO_CHAMPION_NO_FROZEN_PRE_CHANGE',
     'model':'frozen v1.3 versus same-f5-total+natural-core_late-share coherent variant',
     'test_period':'2026-07-16 through 2026-10-12','games':len(df),'missing_first_start_game_count':len(missing),
     'orig_negative_late_count':int(df.orig_negative_late.sum()),
     'method':'Keep v1.3 F5 team means. Late total=raw simple_late away+home. Late share=raw core_late away/(away+home), positive. Full side=F5 side + late side. No fitted new late gamma.',
     'notes':['This candidate was proposed after the same heldout dataset exposed a structural fault; must NOT claim independent prospective confirmation.','Only 2026 feature history (not 2024/25).','Score-level Poisson NLL and full WDL NB2 alpha .322 from fixed 45-run grid are research diagnostics, not Betting EV.']}
for metric,orig,coh in [('score_poisson_nll','orig_nll','coh_nll'),('full_wdl_logloss','orig_wdl_logloss','coh_wdl_logloss')]:
 dif=df[coh].to_numpy()-df[orig].to_numpy()
 bs=np.mean(rng.choice(dif,size=(5000,len(dif)),replace=True),axis=1)
 out[metric]={'orig':round(float(df[orig].mean()),7),'coherent':round(float(df[coh].mean()),7),
   'coh_minus_orig':round(float(dif.mean()),7),
   '95pct_game_bootstrap_CI':[round(float(z),7) for z in np.quantile(bs,[.025,.975])]}
out['worst_orig_incoherent_game']=df[df.orig_negative_late].head(3).to_dict(orient='records')
print('COHERENT_LATE_VARIANT_RESULT='+json.dumps(out,ensure_ascii=False))
Path('/tmp/kbo-coherence/coherent_late_variant_2026.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
