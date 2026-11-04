"""PRICE OPEN audit: uses frozen artifact from 41798174203, never recomputes model."""
from __future__ import annotations
import json,math
from pathlib import Path
PRE=Path('/tmp/ks2price/ks2_premarket_20261104.json')
OUT=Path('/tmp/ks2price/ks2_price_audit_20261104.json')
pre=json.loads(PRE.read_text())
assert pre['data_sha256']=='32c70d31c89c83a9e8606254ff6ec084f599371b6debc6dde36066bf122c74e1'
assert pre['status']=='RESEARCH_PRE_UNOFFICIAL_MARKET_UNSEEN' and not pre['market_prices_used']
assert pre['game_pk']=='77771104SSKT02026'
def bd():
 return [
  ('Full W/L','KT 승',1.64,lambda m:m['full']['KT']),
  ('Full W/L','삼성 승',1.85,lambda m:m['full']['Samsung']),
  ('Full 승1패','KT 2점차 이상',2.38,lambda m:m['win1loss']['KT_by_2plus']),
  ('Full 승1패','1점차 이내/무승부',3.20,lambda m:m['win1loss']['One_run_or_tie']),
  ('Full 승1패','삼성 2점차 이상',2.40,lambda m:m['win1loss']['Samsung_by_2plus']),
  ('Full H-2.5','KT -2.5',2.97,lambda m:m['handicap']['Full']['2.5']['KT_minus']),
  ('Full H-2.5','삼성 +2.5',1.23,lambda m:m['handicap']['Full']['2.5']['Samsung_plus']),
  ('Full U/O8.5','Under 8.5',1.77,lambda m:m['total']['Full']['8.5']['U']),
  ('Full U/O8.5','Over 8.5',1.71,lambda m:m['total']['Full']['8.5']['O']),
  ('F5 승무패','KT F5 승',1.96,lambda m:m['f5']['KT']),
  ('F5 승무패','F5 무',5.40,lambda m:m['f5']['Draw']),
  ('F5 승무패','삼성 F5 승',2.20,lambda m:m['f5']['Samsung']),
  ('F5 H-1.5','KT F5 -1.5',2.72,lambda m:m['handicap']['F5']['1.5']['KT_minus']),
  ('F5 H-1.5','삼성 F5 +1.5',1.28,lambda m:m['handicap']['F5']['1.5']['Samsung_plus']),
  ('F5 U/O4.5','F5 Under 4.5',1.70,lambda m:m['total']['F5']['4.5']['U']),
  ('F5 U/O4.5','F5 Over 4.5',1.78,lambda m:m['total']['F5']['4.5']['O']),
 ]
rows=[]
for fam,selection,odds,fx in bd():
 v={}
 for k,m in pre['models'].items():
  p=float(fx(m));v[k]={'p':p,'fair':round(1/p,4) if p else None,'edge_pp':round(100*(p-1/odds),3),'ev_pct':round(100*(p*odds-1),3)}
 rows.append({'market':fam,'selection':selection,'odds':odds,'break_even':round(1/odds,6),**v})
group={}
for r in rows:group.setdefault(r['market'],[]).append(r)
margins={key:round(sum(1/a['odds'] for a in z)-1,5) for key,z in group.items()}
near={'both_positive':[r for r in rows if all(r[k]['ev_pct']>0 for k in pre['models'])],
      'v12_plus':[r for r in rows if r['v1.2_research']['ev_pct']>0],
      'v13_plus':[r for r in rows if r['v1.3_research']['ev_pct']>0]}
out={'status':'SECONDARY_PRICE_MODEL_COMPARISON_NO_CHAMPION','source':'Zentoto 2026 proto 131 search result, NOT Wisetoto direct; quotes may move','source_urls':['https://v1.zentoto.com/proto/game','https://v1.zentoto.com/proto'],'provenance':'Frozen independent 2026-11-04 17:16 KST market-unseen GitHub run 41798174203','source_data_sha256':pre['data_sha256'],'rows':rows,'margins':margins,'top':near,'practical':{'action':'HOLD_0U_UNTIL_OFFICIAL_CHAMPION_INPUT_AND_LINEUP','reason':'Official model cannot be reproduced; v1.3 late split conflicts with validation; price is secondary only'},'no_bet_claim':True}
OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print('KS2_PRICE_ROWS='+json.dumps(rows,ensure_ascii=False))
print('KS2_PRICE_SUMMARY='+json.dumps({'margins':margins,'positive_both':near['both_positive'],'positive_v12':near['v12_plus'],'positive_v13':near['v13_plus']},ensure_ascii=False))
