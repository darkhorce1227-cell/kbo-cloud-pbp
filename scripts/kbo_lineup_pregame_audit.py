"""Read-only pregame lineup scouting, auditable source cutoff; DOES NOT alter μ or Champion."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

def audit(source:Path,schedule:Path,expected_sha:str,output:Path):
    actual_sha=hashlib.sha256(source.read_bytes()).hexdigest()
    assert actual_sha==expected_sha,('HASH_MISMATCH',actual_sha)
    raw=pd.read_parquet(source)
    cfg=json.loads(schedule.read_text(encoding="utf8"))
    date=pd.Timestamp(cfg['date'])
    assert cfg.get('status')=='USER_SUPPLIED_CONFIRMED_LINEUP'
    assert len(cfg['teams'])==2
    assert set(x['team'] for x in cfg['teams'])=={'KT','SS'}
    for side in cfg['teams']:
        assert len(side['lineup'])==9 and len(set(side['lineup']))==9
    raw['game_date']=pd.to_datetime(raw.game_date)
    assert (raw.game_date<date).all(),'SOURCE_INCLUDES_TARGET_OR_FUTURE_GAME'
    # Per PA, last pitch carries terminal batter outcome.
    x=raw.drop_duplicates(['game_pk','at_bat_number'],keep='last').copy()
    x['batting_team']=np.where(x.inning_topbot.eq('top'),x.away_team,x.home_team)
    ev=x.events.fillna('').astype(str)
    x['hit']=ev.isin(['single','double','triple','home_run']).astype(int)
    x['homer']=ev.eq('home_run').astype(int)
    x['walk']=ev.isin(['walk','intent_walk','hit_by_pitch']).astype(int)
    x['so']=ev.isin(['strikeout','strikeout_double_play']).astype(int)
    x['total_bases']=ev.map({'single':1,'double':2,'triple':3,'home_run':4}).fillna(0).astype(int)
    nm=x[['batter_name','batter','batting_team','game_date','stand']].copy()
    results={'date':str(date.date()),'data_sha256':actual_sha,'status':'AUDIT_ONLY_NO_CALIBRATED_LINEUP_RUN_COEFFICIENT',
       'season_hist_only':'2026','data_asof_exclusive':str(date.date()),
       'source_lineup':cfg.get('source_description','user-confirmed'),
       'teams':[],'limitations':['2026 data only, not full player career','Batter handedness inferred from prior observed plate appearance, not an audited 2026 roster declaration','30-day and whole-season descriptive batting-event rates; no offense/defense/park/opp-hand causal adjustment','Lineup input is user image; official lineup publication clock not independently machine-verified','No coach/active roster condition inference and NO μ adjustment'] }
    for obj in cfg['teams']:
        team=obj['team']; lineup=obj['lineup']
        roster=[]
        for slot,name in enumerate(lineup,1):
            q=x[x.batter_name.eq(name)&x.batting_team.eq(team)]
            if q.empty:
                roster.append({'slot':slot,'name':name,'status':'UNRESOLVED_NO_2026_PA','PA':0,'hand':'UNKNOWN'})
                continue
            unique=q.batter.astype(str).unique().tolist()
            # An ambiguous name must not be silently assigned the wrong batter.
            if len(unique)!=1:
                roster.append({'slot':slot,'name':name,'status':'AMBIGUOUS_PLAYER_NAME','candidate_ids':unique})
                continue
            pid=unique[0]
            prior=x[x.batter.astype(str).eq(pid)]
            last30=prior[prior.game_date>=date-pd.Timedelta(days=30)]
            hand_counts=prior.stand.value_counts(dropna=True)
            hand=hand_counts.index[0] if len(hand_counts)==1 else ('MIXED' if len(hand_counts)>1 else 'UNKNOWN')
            def metrics(q):
                n=len(q)
                return {'PA':n,'hit':int(q.hit.sum()),'HR':int(q.homer.sum()),'BB_HBP':int(q.walk.sum()),
                     'SO':int(q.so.sum()),'TB':int(q.total_bases.sum()),'H_per_PA':round(float(q.hit.mean()),4) if n else None,
                     'HR_per_PA':round(float(q.homer.mean()),4) if n else None,
                     'BB_HBP_per_PA':round(float(q.walk.mean()),4) if n else None,
                     'TB_per_PA':round(float(q.total_bases.mean()),4) if n else None}
            roster.append({'slot':slot,'name':name,'id':pid,'hand':hand,
                'status':'MATCHED_NAME_AND_TEAM','last_game':str(prior.game_date.max().date()),
                'season':metrics(prior),'last_30d':metrics(last30),
                'prior_PAs_by_batting_side':{str(k):int(v) for k,v in hand_counts.items()}})
        assert all(a['status']=='MATCHED_NAME_AND_TEAM' for a in roster),(team,roster)
        count_hand={k:sum(1 for a in roster if a['hand']==k) for k in ['L','R','MIXED','UNKNOWN']}
        results['teams'].append({'team':team,'confirmed_starter':obj.get('starter'),'count_handedness':count_hand,
            'lineup':roster,
            'n_prior_PA_total_9':sum(a['season']['PA'] for a in roster),
            'n_prior_PA_last30_total_9':sum(a['last_30d']['PA'] for a in roster),
            'top_6_left':sum(a['hand']=='L' for a in roster[:6])})
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print('LINEUP_ASOF_AUDIT='+json.dumps({
        'status':results['status'],'source_sha256':actual_sha,
        'teams':[{'team':z['team'],'n_prior_PA_total_9':z['n_prior_PA_total_9'],
                   'count_handedness':z['count_handedness'],'top_6_left':z['top_6_left'],
                   'lineup':[{'slot':p['slot'],'name':p['name'],'hand':p['hand'],'PA':p['season']['PA'],
                             'hit_rate':p['season']['H_per_PA'],'last_30d_pa':p['last_30d']['PA']} for p in z['lineup']]} for z in results['teams']]
        },ensure_ascii=False))
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',required=True)
    ap.add_argument('--lineup',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--sha',required=True)
    a=ap.parse_args()
    audit(Path(a.source),Path(a.lineup),a.sha,Path(a.output))
