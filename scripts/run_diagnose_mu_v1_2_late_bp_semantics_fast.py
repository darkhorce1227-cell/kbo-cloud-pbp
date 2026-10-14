from __future__ import annotations

import numpy as np
import pandas as pd
import diagnose_mu_v1_2_late_bp_semantics_fast as core


def off_suff_fixed(tg, target, window):
    rows=[]
    for r in core.dev(tg).itertuples(index=False):
        d=pd.Timestamp(r.game_date)
        pre=tg[tg.game_date<d]
        my=pre[pre.team.astype(str).eq(str(r.team))]
        if my.empty:
            my=pre
        lg=float(pre[target].mean()) if len(pre) else 1.0
        if window=='calendar30':
            rec=pre[pre.game_date>=d-pd.Timedelta(days=30)]
            myr=rec[rec.team.astype(str).eq(str(r.team))]
            if myr.empty:
                myr=my.tail(min(30,len(my)))
            lgr=float(rec[target].mean()) if len(rec) else lg
        else:
            myr=my.tail(30)
            cutoff=myr.game_date.min() if len(myr) else d
            rec=pre[pre.game_date>=cutoff]
            lgr=float(rec[target].mean()) if len(rec) else lg
        rows.append((float(my[target].sum()),float(len(my)),lg,float(myr[target].sum()),float(len(myr)),lgr))
    return np.asarray(rows,float)

core.off_suff=off_suff_fixed

if __name__=='__main__':
    core.main()
