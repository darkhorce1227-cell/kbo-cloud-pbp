from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import mu_v1_3 as engine
import audit_mu_v1_2_recovered_oos as recovered

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "research" / "mu-v1.2" / "model_spec.json"
METRICS_PATH = ROOT / "research" / "mu-v1.2" / "metrics.json"
OUT_PATH = ROOT / "state" / "mu_v1_2_late_total_variants.json"
STARTER_COLS = [
    "sp_kbb_long","sp_kbb_recent5","sp_hr_long","sp_hr_recent5",
    "sp_h_long","sp_h_recent5","sp_ip_long","sp_ip_recent5",
]


def swap_starter(rows):
    z = rows.copy(deep=True)
    for _, idx in z.groupby("game_pk").groups.items():
        ids = list(idx)
        if len(ids) != 2:
            continue
        a, b = ids
        av = z.loc[a, STARTER_COLS].copy()
        bv = z.loc[b, STARTER_COLS].copy()
        z.loc[a, STARTER_COLS] = bv.to_numpy()
        z.loc[b, STARTER_COLS] = av.to_numpy()
    return z


def shrink_mean(total, n, prior, pseudo):
    den = float(n) + float(pseudo)
    return float((float(total) + float(pseudo) * float(prior)) / den) if den > 0 else float(prior)


def apply_late_features(base_rows, tg, target_col):
    z = base_rows.copy(deep=True)
    vals_long=[]; vals_recent=[]; vals_env=[]
    for r in z.itertuples(index=False):
        d = pd.Timestamp(r.game_date).normalize()
        team = str(r.team)
        pre = tg[(tg.game_date < d) & (tg.game_date >= recovered.HISTORY_START)]
        pre30 = pre[pre.game_date >= d - pd.Timedelta(days=30)]
        my = pre[pre.team.astype(str).eq(team)]
        my30 = pre30[pre30.team.astype(str).eq(team)]
        if my30.empty:
            my30 = my.tail(min(30, len(my)))
        lg = float(pre[target_col].mean()) if len(pre) else 0.0
        lg30 = float(pre30[target_col].mean()) if len(pre30) else lg
        team_long = shrink_mean(float(my[target_col].sum()), len(my), lg, recovered.OFF_LATE_LONG_P)
        team_recent = shrink_mean(float(my30[target_col].sum()), len(my30), lg30, recovered.OFF_LATE_30D_P)
        vals_long.append(team_long / lg if lg else 1.0)
        vals_recent.append(team_recent / lg30 if lg30 else 1.0)
        vals_env.append(lg30)
    z["off_late_long_ratio"] = vals_long
    z["off_late_30d_ratio"] = vals_recent
    z["league_late_30d"] = vals_env
    return z


def mean_mu(rows, model):
    f5=[]; full=[]
    for _, x in rows.groupby("game_pk"):
        if len(x) != 2:
            continue
        away=x[x.is_home==0].iloc[0].to_dict(); home=x[x.is_home==1].iloc[0].to_dict()
        p=engine.score(model,away,home)
        f5 += [float(p["f5_away"]),float(p["f5_home"])]
        full += [float(p["full_away"]),float(p["full_home"])]
    return float(np.mean(f5)),float(np.mean(full))


def summarize(rows, model):
    e=engine.evaluate(rows,model)
    mf5,mfull=mean_mu(rows,model)
    return {
        "f5":{"mae":e["f5"]["team_score_mae"],"wdl_logloss":e["f5"]["wdl_logloss"],"side":e["f5"]["decided_side_hit"],"mean_mu":mf5},
        "full":{"mae":e["full"]["team_score_mae"],"wdl_logloss":e["full"]["wdl_logloss"],"side":e["full"]["decided_side_hit"],"mean_mu":mfull},
    }


def main():
    model=json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    puball=json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    tg,st,bp6=recovered.load_tables()

    base_rows = recovered.make_oos_rows(tg,st,bp6)
    base_rows = swap_starter(base_rows)
    if len(base_rows)!=1368 or base_rows.game_pk.nunique()!=684:
        raise RuntimeError(f"OOS gate failed: rows={len(base_rows)} games={base_rows.game_pk.nunique()}")

    targets={
        "full_minus_f5":"runs_late",
        "inning6plus_pa":"late_6plus_pa",
        "inning6to9_pa":"late_6to9_pa",
    }
    results={}
    for name,col in targets.items():
        rows = apply_late_features(base_rows, tg, col)
        results[name]=summarize(rows,model)

    pub={
        "f5":{"mae":puball["oos_2026"]["f5"]["hybrid"]["mae"],"wdl_logloss":puball["oos_2026"]["f5"]["hybrid"]["wdl_logloss"],"side":puball["decided_side_accuracy_2026"]["f5"],"mean_mu":puball["oos_2026"]["f5"]["hybrid"]["mean_mu"]},
        "full":{"mae":puball["oos_2026"]["full"]["hybrid"]["mae"],"wdl_logloss":puball["oos_2026"]["full"]["hybrid"]["wdl_logloss"],"side":puball["decided_side_accuracy_2026"]["full"],"mean_mu":puball["oos_2026"]["full"]["hybrid"]["mean_mu"]},
    }
    ranked=[]
    for name,r in results.items():
        dist=(abs(r["full"]["mean_mu"]-pub["full"]["mean_mu"])
              +abs(r["full"]["mae"]-pub["full"]["mae"])
              +abs(r["full"]["wdl_logloss"]-pub["full"]["wdl_logloss"]))
        ranked.append({"variant":name,"distance":float(dist),**r})
    ranked.sort(key=lambda x:x["distance"])
    payload={
        "version":"mu-v1.2-late-total-variants-v0.2-fast",
        "status":"DIAGNOSTIC_ONLY",
        "official_shadow_allowed":False,
        "rows":1368,
        "games":684,
        "starter_mapping":"SELF_TEAM_STARTER",
        "published":pub,
        "ranked":ranked,
    }
    OUT_PATH.parent.mkdir(parents=True,exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
