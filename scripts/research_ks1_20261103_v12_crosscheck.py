"""KS1 2026-11-03 PRE cross-check frozen mu-v1.2 research specification, 2026-only data."""
import sys,json,hashlib
from pathlib import Path
import pandas as pd,numpy as np
sys.path.insert(0,"scripts")
import build_mu_v1_2_features as b
import run_mu_v1_2_pipeline as runner
import mu_v1_3 as v
src=Path("/tmp/ks1/kbo_pbp_2026_latest.parquet")
stat=json.loads(Path("state/status.json").read_text())
assert hashlib.sha256(src.read_bytes()).hexdigest()==stat["sha256_parquet"]
raw=pd.read_parquet(src)
pa=runner.prepare_pa_fixed(raw)
team,sp,bp=b.game_tables(pa)
dt=pd.Timestamp("2026-11-03")
away=b.build_feature_row(team,sp,bp,d=dt,team="SS",opponent="KT",opposing_starter="56002",is_home=0)
home=b.build_feature_row(team,sp,bp,d=dt,team="KT",opponent="SS",opposing_starter="65320",is_home=1)
spec=json.loads(Path("research/mu-v1.2/model_spec.json").read_text())
pred=b.score_mu(away,home,spec)
print("KS1_V12_PRE_RESULT="+json.dumps({"model":spec["version"],"source_rows":len(raw),"source_hash":stat["sha256_parquet"],"mu":pred,"f5_wdl_9inning":v.wdl_probs(pred["f5_away"],pred["f5_home"],0.54).tolist(),"full_wdl_9inning":v.wdl_probs(pred["full_away"],pred["full_home"],0.322).tolist(),"features":{"away":away,"home":home},"note":"2026-only feature history, market unseen, candidate not Champion"},ensure_ascii=False,default=float))
