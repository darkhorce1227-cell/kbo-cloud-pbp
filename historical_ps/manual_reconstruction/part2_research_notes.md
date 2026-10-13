# Manual reconstruction v2 — part 2 research audit

This file records the second-pass source audit for the six missing postseason games other than 2008 QP Game 1.

Principles:
- Preserve only source-explicit PA/event information.
- Do not infer omitted PAs from batting order, score arithmetic, or box-score totals.
- A row marked `COMPLETE_HALF_INNING_PA_ORDER` is complete only when the source description closes all three outs for that half-inning.
- `PARTIAL_SEQUENCE_ANCHOR` rows are anchors only; sequence numbers are not literal PA numbers.

Second-pass upgrades added after the base v2 reconstruction:
- 2008 QP Game 2: contemporary OSEN photo/article index confirms additional non-scoring PA anchors (5th-inning strikeout, 7th-inning pinch-hit strikeout and later strikeout, plus a later 3rd-inning flyout).
- 2011 QP Game 3: Yonhap game story confirms the KIA sixth-inning strikeouts after the pitching change.
- 2011 KS Game 2: a contemporaneous game recap gives the complete SK first-inning PA order, adding a third complete half-inning for this game.
- 2012 QP Game 3 lineup was rechecked against the contemporaneous Newsis lineup article; the existing repository lineup is correct: Lee Jong-wook, Min Byung-heon, Kim Hyun-soo, Yoon Seok-min, Choi Joon-seok, Oh Jae-won, Lee Won-seok, Yang Eui-ji, Kim Jae-ho.

The effective PA view must still be generated with `scripts/build_manual_postseason_reconstruction.py`.
