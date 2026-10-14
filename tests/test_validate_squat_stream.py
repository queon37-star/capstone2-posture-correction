"""Synthetic regression tests for post-hoc evaluation; no REHAB24 data required."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from validate_squat_stream import segment_iou, evaluate


def run():
    assert segment_iou((10,20),(10,20))==1.
    assert segment_iou((10,20),(21,30))==0.
    gt=[{'start':10,'end':30,'rep':1,'label':1},
        {'start':40,'end':60,'rep':2,'label':0},
        {'start':70,'end':90,'rep':3,'label':0}]
    pred=[{'start':12,'end':28,'posture':'good','count':1},
          {'start':43,'end':55,'posture':'good','count':2},
          {'start':95,'end':105,'posture':'bad','count':3}]
    summ,rows=evaluate(gt,pred)
    assert (summ['matched'],summ['missed'],summ['extra']) == (2,1,1)
    assert summ['gt_incorrect']==2 and summ['detected_and_classified_incorrect']==0
    assert summ['matched_classifier_accuracy']==0.5
    assert summ['incorrect_end_to_end_detection_rate']==0
    assert sum(r['status']=='missed' for r in rows)==1
    assert sum(r['status']=='extra' for r in rows)==1
    print('Stream evaluator synthetic regression tests: PASS')


if __name__ == '__main__':
    run()
