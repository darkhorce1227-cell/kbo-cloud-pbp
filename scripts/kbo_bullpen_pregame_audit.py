#!/usr/bin/env python3
"""Read-only reproducible KBO bullpen candidate availability audit.
Inputs are solely PBP from dates strictly earlier than --date.
NO calibrated run coefficient, win-probability adjustment, or Champion promotion.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd


def audit(path:Path,team:str,day:str,expect_sha:str|None=None)->dict:
    d=pd.Timestamp(day).normalize()
    if expect_sha:
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        if sha!=expect_sha:raise RuntimeError(f'SHA mismatch: {sha}')
    else:sha=None
    raw=pd.read_parquet(path)
    need={'game_date','game_pk','pitcher','pitcher_name','inning','inning_topbot','home_team','away_team','at_bat_number','pitch_number','events'}
    if need-set(raw.columns):raise ValueError(f'missing columns {need-set(raw.columns)}')
    raw['date']=pd.to_datetime(raw.game_date).dt.normalize()
    x=raw[raw.date.lt(d)].copy()
    if not x.date.lt(d).all():raise RuntimeError('Pregame leakage')
    if x.duplicated(['game_pk','at_bat_number','pitch_number']).any():raise RuntimeError('duplicate pitch keys')
    x['pitching_team']=np.where(x.inning_topbot.astype(str).str.lower().str.startswith('top'),x.home_team,x.away_team)
    x=x[x.pitching_team.astype(str).eq(team)].copy()
    if x.empty:return {'status':'UNAVAILABLE_TEAM_HISTORY','team':team,'as_of_date':str(d.date())}
    x['pitcher']=x.pitcher.astype(str)
    x=x.sort_values(['date','game_pk','at_bat_number','pitch_number'],kind='stable')
    first=x.drop_duplicates('game_pk',keep='first')[['game_pk','pitcher']].rename(columns={'pitcher':'starter_id'})
    x=x.merge(first,on='game_pk',validate='many_to_one')
    x['is_relief']=x.pitcher.ne(x.starter_id)
    relief=x[x.is_relief].copy()
    pa=relief.drop_duplicates(['game_pk','at_bat_number'],keep='last').copy()
    pa['bf']=pa.events.notna().astype(int)
    pa['k']=pa.events.eq('strikeout').astype(int)
    pa['bb']=pa.events.eq('walk').astype(int)
    pa['hr']=pa.events.eq('home_run').astype(int)
    pa['late68']=pa.inning.between(6,8).astype(int)
    role=pa[pa.date.ge(d-pd.Timedelta(days=35))].groupby('pitcher',as_index=False)['late68'].sum()
    ids=role.loc[role.late68>=2,'pitcher'].astype(str).tolist()
    cand=[]
    for pid in ids:
        q=pa[pa.pitcher.eq(pid)]
        r=relief[relief.pitcher.eq(pid)]
        q35=q[q.date.ge(d-pd.Timedelta(days=35))]
        bf=int(q.bf.sum()); k=int(q.k.sum());bb=int(q.bb.sum());hr=int(q.hr.sum())
        last=str(q.date.max().date())
        n1=int(len(r[r.date.ge(d-pd.Timedelta(days=1))]));n2=int(len(r[r.date.ge(d-pd.Timedelta(days=2))]));n3=int(len(r[r.date.ge(d-pd.Timedelta(days=3))]))
        cand.append({'pitcher_id':pid,'pitcher_name':str(q.pitcher_name.iloc[-1]),
          'last_relief_date':last,'prior_relief_games':int(q.game_pk.nunique()),
          'prior_relief_bf':bf,'k_minus_bb_pct':round((k-bb)/bf,4) if bf>0 else None,
          'hr_per_bf':round(hr/bf,4) if bf>0 else None,
          'last35_late68_PA':int(q35.late68.sum()),
          'last1d_raw_pitch_rows':n1,'last2d_raw_pitch_rows':n2,'last3d_raw_pitch_rows':n3,
          'workload_watch_2d_40plus':bool(n2>=40),
          'quality_sample':'THIN' if bf<100 else 'DESCRIPTIVE_ONLY'})
    cand.sort(key=lambda z:(-z['last35_late68_PA'],z['pitcher_id']))
    return {'status':'PREGAME_ROLE_AUDIT_NO_RUN_COEFFICIENT','date_cutoff_exclusive':str(d.date()),
            'team':team,'season':d.year,'source_sha256':sha,
            'prior_team_games':int(x.game_pk.nunique()),'candidate_count':len(cand),
            'candidates':cand,
            'caution':['Candidates are proxies based on prior 35 calendar days in relief during innings 6-8, NOT the actual activated roster.',
                       'No future game (date>=target) is included; same-day other games are also excluded.',
                       'Raw pitch rows need not equal official pitch count; cannot infer confirmed availability from these alone.',
                       'No Park / closer probability or bullpen ability coefficient is produced.',
                       'K-BB and HR rates are raw descriptive; thin samples and deployment quality require shrinkage before model use.']}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--team',required=True)
    ap.add_argument('--date',required=True);ap.add_argument('--expect-sha');ap.add_argument('--output')
    args=ap.parse_args()
    out=audit(Path(args.source),args.team,args.date,args.expect_sha)
    if args.output:Path(args.output).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print('KBO_PRE_BULLPEN_AUDIT='+json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()