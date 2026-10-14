from __future__ import annotations

import reverse_mu_v1_2_final_development as core

_original = core.build_sufficient


def build_sufficient_fixed(tg, stc, sta, bp):
    stc = stc.copy()
    if "hbp" not in stc.columns:
        stc["hbp"] = 0.0
    return _original(tg, stc, sta, bp)


core.build_sufficient = build_sufficient_fixed

if __name__ == "__main__":
    core.main()
