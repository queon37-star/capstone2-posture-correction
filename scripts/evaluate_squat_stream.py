"""Post-hoc only: match online detected reps against REHAB24-6 annotations.

GT is NEVER fed to online counter or classifier. Matching is for evaluation.
"""
import argparse
import csv
import json
from pathlib import Path


def overlap_iou(a, b):
    intersection = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - intersection
    return intersection / union if union > 0 else 0.


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--video', default='PM_022')
    ap.add_argument('--events', type=Path, default=Path('results/stream_test/PM_022_events.jsonl'))
    ap.add_argument('--annotations', type=Path, default=Path('data/raw/REHAB24-6/Segmentation.csv'))
    ap.add_argument('--start-frame', type=int, default=0)
    ap.add_argument('--end-frame', type=int, default=1199)
    ap.add_argument('--min-iou', type=float, default=.2)
    args=ap.parse_args()
    if args.end_frame <= args.start_frame:
        raise ValueError('Bad validation frame window')
    with args.annotations.open(newline='') as f:
        gt_all=[r for r in csv.DictReader(f, delimiter=';') if r['video_id']==args.video and r['exercise_id']=='6']
    gt=[{'start':int(r['first_frame']), 'end':int(r['last_frame']),
         'rep':int(r['repetition_number']), 'correct':int(r['correctness'])}
        for r in gt_all if int(r['first_frame'])>=args.start_frame and int(r['last_frame'])<=args.end_frame]
    pred=[]
    with args.events.open() as f:
        for line in f:
            if not line.strip():continue
            r=json.loads(line)
            if r.get('type')!='rep':continue
            st,ed=int(r['start_frame']),int(r['end_frame'])
            if st>=args.start_frame and ed<=args.end_frame:
                pred.append({'start':st,'end':ed,'posture':r.get('posture','unknown'),'count':r.get('count')})
    candidates=sorted(((overlap_iou((g['start'],g['end']),(p['start'],p['end'])), i, j)
                       for i,g in enumerate(gt) for j,p in enumerate(pred)),reverse=True)
    match=[]; done_gt=set();done_pred=set()
    for iou,i,j in candidates:
        if iou < args.min_iou:break
        if i in done_gt or j in done_pred:continue
        done_gt.add(i);done_pred.add(j)
        match.append((gt[i],pred[j],iou))
    print('Video:',args.video,'validation range:',args.start_frame,'..',args.end_frame)
    print('Complete annotated repetitions:',len(gt),'predicted completed repetitions:',len(pred))
    print('Matched:',len(match),'missed:',len(gt)-len(match),'extra:',len(pred)-len(match))
    if match:
        start_err=sum(abs(g['start']-p['start']) for g,p,_ in match)/len(match)
        end_err=sum(abs(g['end']-p['end']) for g,p,_ in match)/len(match)
        correct=sum((p['posture']=='good')==(g['correct']==1) for g,p,_ in match)
        print(f'Start frame MAE: {start_err:.2f}; End frame MAE: {end_err:.2f}')
        print(f'Classifier accuracy on matched detected reps only: {correct}/{len(match)} ({correct/len(match):.3f})')
        print('This accuracy is NOT equivalent to LOPO accuracy and may use training people.')
        for g,p,iou in sorted(match,key=lambda r:r[0]['start']):
            print(f"GT_rep={g['rep']:02d} GT={g['start']}-{g['end']} "
                  f"pred={p['start']}-{p['end']} IoU={iou:.2f} "
                  f"GT_label={g['correct']} pred={p['posture']}")
    else:
        print('No reps matched: diagnose calibration, knee/hip thresholds and person tracking.')


if __name__=='__main__': main()
