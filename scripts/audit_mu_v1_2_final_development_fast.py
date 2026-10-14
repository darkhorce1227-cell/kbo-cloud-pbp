from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import pandas as pd
import build_mu_v1_2_features as mu
from run_mu_v1_2_pipeline import prepare_pa_fixed
from diagnose_mu_v1_2_sp_bp_definitions import starter_game_table
from audit_mu_v1_2_final_development import ROOT,SHA24,SHA25,FROZEN,make_features,metric

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--training-dir',default='out_training'); ap.add_argument('--out',default='state/mu_v1_2_final_development_fast.json'); z=ap.parse_args()
    p24=ROOT/z.training_dir/'kbo_pbp_2024_training.csv.gz'; p25=ROOT/z.training_dir/'kbo_pbp_2025_training.csv.gz'
    assert hashlib.sha256(p24.read_bytes()).hexdigest()==SHA24 and hashlib.sha256(p25.read_bytes()).hexdigest()==SHA25
    df=pd.concat([pd.read_csv(p24,low_memory=False),pd.read_csv(p25,low_memory=False)],ignore_index=True); pa=prepare_pa_fixed(df); tg,st,bp=mu.game_tables(pa); sta=starter_game_table(pa)
    cand=[]
    for scope in ['continuous','season_reset']:
      for sm in ['classified_pa_kbb','all_pa_hbp']:
        f=make_features(tg,st,sta,bp,scope,sm,0.0); checks={k:metric(v,k) for k,v in f.items()}; cand.append({'history_scope':scope,'starter_mode':sm,'rows':len(next(iter(f.values()))),'loss':sum(x['loss'] for x in checks.values()),'checks':checks})
    cand.sort(key=lambda x:x['loss']); payload={'version':'mu-v1.2-final-development-fast-v0.1','expected_rows':2628,'official_shadow_allowed':False,'candidates_ranked':cand}
    out=ROOT/z.out; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(payload,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
