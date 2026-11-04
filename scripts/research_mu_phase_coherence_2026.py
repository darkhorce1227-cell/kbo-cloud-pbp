"""Detect physically impossible F5>Full projected team means in frozen v1.2/v1.3.
No model-fitting/odds and NO same-game leakage. 2026 dated cross-check.
"""
import sys,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,'scripts')
import build_mu_v1_2_features as b
import run_mu_v1_2_pipeline as runner
import mu_v1_3 as m
SRC=Path('/tmp/kbo-coherence/kbo_pbp_2026_latest.parquet')
EXPECTED='32c70d31c89c83a9e8606254ff6ec084f599371b6debc6dde36066bf122c74e1'
assert hashlib.sha256(SRC.read_bytes()).hexdigest()==EXPECTED
raw=pd.read_parquet(SRC)
assert len(raw)==224611
raw['game_date']=pd.to_datetime(raw.game_date)
raw=raw[raw.game_date<=pd.Timestamp('2026-11-03')].copy()
pa=runner.prepare_pa_fixed(raw)
tg,st,bp=b.game_tables(pa)
m.base.load_seasons=lambda year:raw.copy() if year==2026 else (_ for _ in ()).throw(ValueError('2026_ONLY'))
t13,s13,b13=m.load_all(2026)
model12=json.loads(Path('research/mu-v1.2/model_spec.json').read_text())
model13=json.loads(Path('research/mu-v1.3/model_spec.json').read_text())
cfg=m.config()
mapst=st.set_index(['game_pk','pitching_team'])['starter'].astype(str).to_dict()
all_dates=tg[(tg.game_date>=pd.Timestamp('2026-07-16')) & (tg.game_date<=pd.Timestamp('2026-10-12'))]
logs=[]
missing=[]
for pk,gg in all_dates.groupby('game_pk'):
 if len(gg)!=2:continue
 g=gg.sort_values('is_home')
 ar=g.iloc[0];ho=g.iloc[1]
 da=ar.game_date
 aid=mapst.get((pk,ar.team));hid=mapst.get((pk,ho.team))
 if not aid or not hid:
  missing.append(pk);continue
 try:
  x12a=b.build_feature_row(tg,st,bp,d=da,team=ar.team,opponent=ho.team,opposing_starter=hid,is_home=0)
  x12h=b.build_feature_row(tg,st,bp,d=da,team=ho.team,opponent=ar.team,opposing_starter=aid,is_home=1)
  out12=b.score_mu(x12a,x12h,model12)
  x13a=m.build_features(t13,s13,b13,d=da,team=ar.team,opponent=ho.team,opposing_starter=hid,is_home=0,cfg=cfg)
  x13h=m.build_features(t13,s13,b13,d=da,team=ho.team,opponent=ar.team,opposing_starter=aid,is_home=1,cfg=cfg)
  out13=m.score(model13,x13a,x13h)
 except Exception as e:
  missing.append(str(pk)+':'+str(e));continue
 for name,pred in [('v1.2',out12),('v1.3',out13)]:
  lf=float(pred['full_away']-pred['f5_away']);lh=float(pred['full_home']-pred['f5_home'])
  logs.append({'pk':pk,'date':str(da.date()),'model':name,'away':ar.team,'home':ho.team,
     'f5away':float(pred['f5_away']),'f5home':float(pred['f5_home']),
     'fullaway':float(pred['full_away']),'fullhome':float(pred['full_home']),
     'lateaway':lf,'latehome':lh,'sumlate':lf+lh,
     'f5scoremargin':float(pred['f5_home']-pred['f5_away']),
     'fullscoremargin':float(pred['full_home']-pred['full_away'])})
df=pd.DataFrame(logs)
assert len(df)>400,(len(df),missing[:20])
res={'source_hash':EXPECTED,'status':'RESEARCH_STRUCTURAL_VALIDITY_NO_MODEL_PROMOTION',
     'data_range':'2026-07-16 through 2026-10-12','data':'2026 historical only; strictly prior games used for every dated feature',
     'games_per_model':{},'missing_games':missing[:12]}
for name,g in df.groupby('model'):
 n=len(g)
 negatives=g[(g.lateaway < -1e-9)|(g.latehome < -1e-9)]
 m=(g.f5scoremargin*g.fullscoremargin<0)
 flips=(m&g.f5scoremargin.abs().gt(.1))
 severe=g[(g.lateaway<.25)|(g.latehome<.25)]
 res['games_per_model'][name]={
    'n_games':n,'negative_late_any_game':len(negatives),'negative_late_pct':len(negatives)/n,
    'negative_late_team_game':int(g.lateaway.lt(0).sum()+g.latehome.lt(0).sum()),
    'late_away_min':round(float(g.lateaway.min()),6),'late_home_min':round(float(g.latehome.min()),6),
    'late_team_game_under_0p25':int(g.lateaway.lt(.25).sum()+g.latehome.lt(.25).sum()),
    'expected_lead_side_flip_games':int(m.sum()),'expected_flip_pct':float(m.mean()),
    'expected_flip_abs_f5margin_over_0p1':int(flips.sum()),
    'negative_examples':negatives.sort_values(['lateaway','latehome']).head(10).to_dict(orient='records'),
    'small_late_examples':severe.sort_values(['lateaway','latehome']).head(6).to_dict(orient='records'),
    'late_team_rate_quantiles':[round(float(v),5) for v in np.quantile(np.r_[g.lateaway,g.latehome],[0,.01,.05,.5,.95,.99,1])],
 }
print('MU_PHASE_COHERENCE_RESULT='+json.dumps(res,ensure_ascii=False))
Path('/tmp/kbo-coherence/coherence_2026.json').write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n')
