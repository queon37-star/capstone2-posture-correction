"""Synthetic checks for trial sequencing and evaluation bookkeeping, not video performance."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_squat_lopo_augmentation import trim_indices,overlap_iou,match_events,stats
import numpy as np


def test_trim():
    assert trim_indices(100,0)==[(0,100)]
    assert trim_indices(100,.05)==[(0,100),(5,100),(0,95),(5,95)]
    assert trim_indices(100,.10)==[(0,100),(10,100),(0,90),(10,90)]
    for left,right in trim_indices(101,.10):
        assert 0<=left<right<=101
    try: trim_indices(12,.2)
    except ValueError: pass
    else: raise AssertionError('Should reject short crops')


def test_match():
    s=[{'ids':np.arange(10,31)},{'ids':np.arange(40,61)}]
    e=[{'start_frame':11,'end_frame':32},{'start_frame':39,'end_frame':59}]
    matched=match_events(s,e)
    assert len(matched)==2
    assert matched[0]['start_frame']==11 and matched[1]['start_frame']==39
    assert overlap_iou(1,3,5,7)==0
    assert overlap_iou(1,3,1,3)==1


def test_stats():
    m=stats([1,1,0,0],[0,1,1,0])
    assert m['n']==4 and m['accuracy']==.5 and m['false_alarm']==1 and m['missed_bad']==1

if __name__=='__main__':
    test_trim();test_match();test_stats()
    print('LOPO robustness experiment synthetic tests: PASS')
