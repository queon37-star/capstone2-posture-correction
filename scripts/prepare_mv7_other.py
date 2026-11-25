#!/usr/bin/env python3
"""Select and screen a balanced subset of MV7 Other clips. Run at repository root.
Uses existing pose extraction modules; leaves original files, caches and earlier reports intact.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random

WORK = Path('data/preprocess/extra_datasets')
OTHER = {'abs','back','bicep_curl','shoulder','tricep'}


def read(name):
    return json.loads((WORK/name).read_text())


def save(name, data):
    path = WORK/name
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


def select(rows, persons, excluded, n, seed):
    pool = [v for v in rows if v['dataset']=='mv7' and v['exercise'] in OTHER
            and v['person_id'] in persons and v['sample_id'] not in excluded]
    rng = random.Random(seed)
    pool.sort(key=lambda v:v['sample_id']); rng.shuffle(pool)
    pc, ec, vc, qc, ev = (Counter() for _ in range(5)); picked=[]
    while pool and len(picked)<n:
        j=min(range(len(pool)), key=lambda j: (
            pc[pool[j]['person_id']],
            ev[pool[j]['exercise'],pool[j]['camera']],
            ec[pool[j]['exercise']], vc[pool[j]['camera']],
            qc[pool[j]['clip_correctness']]))
        v=pool.pop(j); picked.append(v)
        pc[v['person_id']]+=1; ec[v['exercise']]+=1; vc[v['camera']]+=1
        ev[v['exercise'],v['camera']]+=1; qc[v['clip_correctness']]+=1
    return picked


def plan(n,seed):
    m=read('manifest.json'); groups=read('mv7_duplicate_groups.json')
    excluded={v['sample_id'] for g in groups for v in g['videos']}
    rows=m['videos']; lookup={v['sample_id']:v for v in rows}
    if not excluded or not excluded <= lookup.keys(): raise ValueError('Missing or inconsistent duplicate inventory')
    if sum(v['dataset']=='mv7' for v in rows)!=1042: raise ValueError('Unexpected MV7 inventory')
    # Keep duplicate-linked subjects together in any later person-wise split.
    parent={v['person_id']:v['person_id'] for v in rows if v['dataset']=='mv7'}
    def find(x):
        while parent[x]!=x: x=parent[x]
        return x
    for g in groups:
        ids=sorted({v['person_id'] for v in g['videos']})
        for x in ids[1:]: parent[find(x)]=find(ids[0])
    save('mv7_exclusion_policy.json',dict(excluded_sample_ids=sorted(excluded),
         person_fold_groups={x:find(x) for x in sorted(parent)},policy='Exclude all exact duplicate group members; group linked IDs conservatively'))
    with (WORK/'mv7_cached_body_quality.csv').open(newline='') as f:
        q=list(csv.DictReader(f))
    persons={v['person_id'] for v in q if v['sample_id'] not in excluded and v['quality_pass'].lower()=='true'}
    if not persons: raise ValueError('No eligible target-motion subjects')
    picked=select(rows,persons,excluded,n,seed)
    if not picked: raise ValueError('No Other candidates')
    save('mv7_other_plan.json',dict(seed=seed,requested=n,total=len(picked),sample_ids=[v['sample_id'] for v in picked]))
    print('Excluded:',len(excluded),'Other planned:',len(picked),'persons:',len({v['person_id'] for v in picked}))
    print('Exercise:',dict(Counter(v['exercise'] for v in picked)))
    print('Camera:',dict(Counter(v['camera'] for v in picked)))


def screen():
    import numpy as np
    import preprocess_rehab24 as pre
    import screen_mv7_anchors as sc
    from extract_video_kp_cache import UltralyticsPoseBackend,sha256_file
    # Generic arm or torso/leg chain for negative samples, not posture grading.
    for ex in OTHER:
        sc.CHAINS[ex]=((5,7,9),(6,8,10),(5,11,13),(6,12,14))
    m=read('manifest.json'); mh=m['weights_sha256']
    if sha256_file(m['weights'])!=mh: raise ValueError('Weights changed')
    lookup={v['sample_id']:v for v in m['videos']}
    ids=read('mv7_other_plan.json')['sample_ids']
    excluded=set(read('mv7_exclusion_policy.json')['excluded_sample_ids'])
    if set(ids)&excluded: raise ValueError('Plan contains excluded IDs')
    anchors=read('anchors.json') if (WORK/'anchors.json').exists() else {}
    backend=UltralyticsPoseBackend(m['weights'],device='0',imgsz=640,conf=.25)
    records=[]; ready=[]; pending=[]; counts=Counter()
    for sid in ids:
        v=lookup[sid]; a=anchors.get(sid)
        rec={k:v[k] for k in ('sample_id','exercise','camera','person_id')}
        try:
            state=pre.check_cache(v,mh,verify_anchor=a)
            if state=='complete': status,reason='existing','complete_cache'
            elif state!='missing': status,reason='error',state
            elif a is not None: status,reason='existing','anchor_preserved'
            else:
                probes=sc.inspect(v,backend)
                status,reason,new,detail=sc.classify(probes)
                rec.update(detail)
                if new is not None:
                    new['selection_method']='auto_mv7_other_generic_chain_v1'
                    anchors[sid]=new; pre.save_json(WORK/'anchors.json',anchors)
        except Exception as e:
            status,reason='error',str(e)
        rec.update(status=status,reason=reason); records.append(rec); counts[status]+=1
        if status in ('auto','existing'): ready.append(sid)
        elif status=='review': pending.append(sid)
        print(f'{len(records)}/{len(ids)} {sid} {status} {reason}',flush=True)
        save('mv7_other_screening.json',dict(total=len(ids),processed=len(records),counts=dict(counts),records=records))
        (WORK/'mv7_other_ready.txt').write_text(''.join(x+'\n' for x in ready))
        (WORK/'mv7_other_review.txt').write_text(''.join(x+'\n' for x in pending))
    print('Summary:',dict(counts))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['plan','screen']); p.add_argument('--count',type=int,default=150); p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if a.count<1: p.error('--count must be positive')
    if a.mode=='plan': plan(a.count,a.seed)
    else: screen()
