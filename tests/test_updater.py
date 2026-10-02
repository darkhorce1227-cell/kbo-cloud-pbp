from datetime import date, time
import pandas as pd

from cloud.update_pbp import (
    daily_schedule, final_mask, first_check_time, has_1400_game,
    validate_candidate,
)

def test_start_gate():
    assert first_check_time(True) == time(17,0)
    assert first_check_time(False) == time(22,0)
    assert first_check_time(None) == time(17,0)

def test_1400_detect():
    df = pd.DataFrame([{"gameId":"20261001AABB02026","gameDateTime":"2026-10-01 14:00:00"}])
    assert has_1400_game(df) is True

def test_statusnum_final():
    df = pd.DataFrame([{"statusNum":3},{"statusNum":2},{"statusNum":1}])
    assert final_mask(df).tolist() == [True, False, False]

def test_daily_schedule_filters_round_and_cancel():
    df = pd.DataFrame([
        {"gameId":"20261001AABB02026","roundCode":"kbo_r","statusInfo":"9회말","played":True},
        {"gameId":"20261001CCDD02026","roundCode":"kbo_r","statusInfo":"경기취소","played":False},
        {"gameId":"20261001EEFF02026","roundCode":"kbo_as","statusInfo":"9회말","played":True},
        {"gameId":"20261002GGHH02026","roundCode":"kbo_r","statusInfo":"9회말","played":True},
    ])
    got = daily_schedule(df, date(2026,10,1))
    assert got["gameId"].tolist() == ["20261001AABB02026"]

def test_validation_happy_path():
    rows=[]
    for gid in ["20261001AABB02026","20261001CCDD02026"]:
        for ab,p in [(1,1),(1,2),(2,1)]:
            rows.append({
                "game_pk":gid,"game_date":"2026-10-01","home_team":"BB","away_team":"AA",
                "inning":1,"inning_topbot":"top","at_bat_number":ab,"pitch_number":p,
                "batter":"1","pitcher":"2","batter_name":"A","pitcher_name":"P",
                "balls":0,"strikes":0,"outs_when_up":0,"home_score":0,"away_score":0,
                "pitch_result":"S","pitch_type":"FF","pitch_name":"4-Seam Fastball",
                "release_speed_kmh":145,"plate_x":0.1,"plate_z":2.5,
                "events":None,"post_home_score":0,"post_away_score":0,
            })
    df=pd.DataFrame(rows)
    ok, diag=validate_candidate(df,date(2026,10,1),
                               ["20261001AABB02026","20261001CCDD02026"],{})
    assert ok
    assert diag["duplicate_stable_keys"] == 0


def test_final_mask_statusinfo_fallback():
    df = pd.DataFrame([
        {"statusInfo":"9회말"},
        {"statusInfo":"경기종료"},
    ])
    got = final_mask(df)
    assert len(got) == 2
