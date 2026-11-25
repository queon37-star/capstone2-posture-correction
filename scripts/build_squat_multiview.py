#!/usr/bin/env python3
"""Build full/Crop10 180-feature arrays from original GT NPZ and extra caches.
Run in pose_training_offline with this script in scripts/. No YOLO inference.
Original 195 samples are never filtered; extra views are quality-screened.
"""
import argparse, hashlib, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def digest(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def feature_rows(seq, names):
    from train_squat_crop10_all9 import make_augmented_rows
    from train_squat_extended import extract_feature_groups
    x, _ = make_augmented_rows([seq], [0], names, extractor=extract_feature_groups, fraction=.10)
    if x.shape != (4, 180) or np.isinf(x).any() or not np.isfinite(x).any(axis=1).all():
        raise ValueError('Invalid Crop10 features')
    return x


def assess(k, status, boxes, a, b):
    seq = np.array(k[a:b+1], dtype=np.float32, copy=True)
    sel = status[a:b+1] == 0
    seq[~sel] = np.nan
    valid = np.isfinite(seq[..., :2]).all(-1) & np.isfinite(seq[..., 2]) & (seq[..., 2] >= .3)
    ctr = (boxes[:, :2] + boxes[:, 2:]) / 2
    scale = np.maximum(np.linalg.norm(boxes[:, 2:] - boxes[:, :2], axis=1), 1)
    # Only adjacent selected boxes; flag jumps with either endpoint in the rep.
    jumps = (status[:-1] == 0) & (status[1:] == 0) & (np.linalg.norm(np.diff(ctr, axis=0), axis=1) / scale[:-1] >= .25)
    count = int(jumps[max(0,a-1):min(len(jumps), b+1)].sum())
    q = dict(selected_pct=100*float(sel.mean()), valid_kp_pct=100*float(valid.mean()), large_jump_pairs=count)
    reasons = []
    if len(seq) < 15: reasons.append('too_short')
    if sel.mean() < .99: reasons.append('selection_below_99pct')
    if valid.mean() < .75: reasons.append('valid_kp_below_75pct')
    if count: reasons.append('large_box_jump')
    return seq, q, reasons


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-dir', type=Path, default=Path('data/processed/REHAB24-6/camera17_yolov8m_tracked'))
    p.add_argument('--template', type=Path, default=Path('models/squat_plus_asymmetry180.joblib'))
    p.add_argument('--out-dir', type=Path, default=Path('results/squat_multiview/build'))
    args=p.parse_args()
    import preprocess_rehab24 as pre
    from extract_video_kp_cache import KP_DUMP_DTYPE
    from train_lopo_heldout_models import load_corpus
    original=args.original_dir
    if not (original/'index.csv').is_file() and original==Path('data/processed/REHAB24-6/camera17_yolov8m_tracked'):
        original=Path.home()/'capstone2/pose_training'/original
    rpath=Path('data/preprocess/rehab24/batch_manifest.json')
    epath=Path('data/preprocess/extra_datasets/manifest.json')
    r,e=read(rpath),read(epath)
    if r['weights_sha256']!=e['weights_sha256']: raise ValueError('Pose weight hashes differ')
    source=joblib.load(args.template)  # trusted local model only
    names=list(source['feature_names'])
    if source.get('feature_variant')!='PlusAsymmetry180' or len(names)!=180 or len(set(names))!=180: raise ValueError('Wrong template schema')
    idx,seqs=load_corpus(original)
    r24=[v for v in r['videos'] if int(v['exercise_id'])==6]
    r26=[v for v in e['videos'] if v['dataset']=='rehab26' and int(v['exercise_id'])==6]
    assert len(r24)==18 and len(r26)==15
    old={}
    for v in r24:
        if int(v['camera'])==17:
            for a in v['annotations']:
                key=(v['video_id'],int(a['repetition_number']))
                if key in old: raise ValueError(f'Duplicate original annotation {key}')
                old[key]=(v,a)
    assert len(old)==195
    rows=[]; xs=[]; audit=[]
    for row,seq in zip(idx.itertuples(index=False),seqs):
        v,a=old[(str(row.video_id),int(row.repetition_number))]
        if int(row.person_id)!=int(v['person_id']) or int(row.label)!=int(a['correctness']): raise ValueError('Original manifest/index label/person mismatch')
        # Preserve the exact original NPZ feature input/order, even if its quality differs.
        meta=dict(dataset='rehab24',camera=17,person_id=str(row.person_id),group=f'rehab24:{int(row.person_id)}',video_id=str(row.video_id),sample_id=v['sample_id'],repetition_number=int(row.repetition_number),orientation='',viewing_angle_deg='',label=int(row.label),first_frame=int(a['first_frame']),last_frame=int(a['last_frame']),error_description='',additional_errors='',rep_id=f'rehab24:{row.video_id}:{int(row.repetition_number)}',source='original_npz')
        xs.append(feature_rows(seq,names)); rows.append(meta); audit.append({**meta,'accepted':True,'reason':'original_baseline_preserved'})
    assert len(rows)==195
    for ds,videos in [('rehab24',[v for v in r24 if int(v['camera'])==18]),('rehab26',r26)]:
        for v in videos:
            state=pre.check_cache(v,r['weights_sha256'])
            if state!='complete': raise RuntimeError(f'{v["sample_id"]}: {state}')
            if abs(float(v['fps'])-30)>.01: raise ValueError('Expected 30 FPS for unchanged temporal features')
            paths=pre.cache_paths(v); data=np.memmap(paths[0],dtype=KP_DUMP_DTYPE,mode='r')
            if not np.array_equal(data['frame_id'],np.arange(len(data))): raise ValueError('Noncontiguous cache')
            with np.load(paths[2],allow_pickle=False) as z:
                status=z['status']; boxes=z['selected_box']
            n=len(data); count=0
            for a in v['annotations']:
                lo,hi=int(a['first_frame']),int(a['last_frame']); rep=int(a['repetition_number']); label=int(a['correctness'])
                if not 0<=lo<=hi<n or label not in (0,1): raise ValueError(f'Invalid annotation {a}')
                person=str(int(v['person_id'])) if ds=='rehab24' else str(v['person_id'])
                orient='' if ds=='rehab24' else int(a['person_orientation'])
                rid=f'rehab24:{v["video_id"]}:{rep}' if ds=='rehab24' else f'rehab26:{v["recording_id"]}:{orient}:{rep}'
                meta=dict(dataset=ds,camera=int(v['camera']),person_id=person,group=f'{ds}:{person}',video_id=v['video_id'],sample_id=v['sample_id'],repetition_number=rep,orientation=orient,viewing_angle_deg=a.get('viewing_angle_deg',''),label=label,first_frame=lo,last_frame=hi,error_description=a.get('error_description',''),additional_errors=a.get('additional_errors',''),rep_id=rid,source='full_video_cache')
                if ds=='rehab24':
                    ov,oa=old[(v['video_id'],rep)]
                    if (int(ov['person_id']),int(oa['correctness']),int(oa['first_frame']),int(oa['last_frame']))!=(int(person),label,lo,hi): raise ValueError('Paired camera annotation mismatch')
                seq,q,reasons=assess(data['kp'],status,boxes,lo,hi)
                if not reasons:
                    try: x=feature_rows(seq,names)
                    except (ValueError,KeyError,AssertionError) as exc: reasons.append(f'feature_failure:{type(exc).__name__}:{exc}')
                audit.append({**meta,**q,'accepted':not reasons,'reason':';'.join(reasons)})
                if not reasons:
                    xs.append(x); rows.append(meta); count+=1
            print(v['sample_id'],f'{count}/{len(v["annotations"])} accepted',flush=True)
    df=pd.DataFrame(rows); qa=pd.DataFrame(audit); X=np.stack(xs)
    if df.duplicated(['rep_id','camera']).any() or qa.duplicated(['rep_id','camera']).any(): raise ValueError('Duplicate rep/view')
    if (qa.groupby('rep_id').label.nunique()>1).any() or (qa.groupby('rep_id').group.nunique()>1).any(): raise ValueError('Cross-view label/group conflict')
    if len(qa)!=2052 or X.shape!=(len(df),4,180): raise ValueError(f'Unexpected inventory {len(qa)}, {X.shape}')
    args.out_dir.mkdir(parents=True,exist_ok=False)
    np.save(args.out_dir/'features_crop10.npy',X)
    df.to_csv(args.out_dir/'repetitions.csv',index=False); qa.to_csv(args.out_dir/'quality_audit.csv',index=False)
    manifest=dict(feature_names=names,feature_shape=list(X.shape),augmentation=['full','trim_start','trim_end','trim_both'],template=str(args.template),template_sha256=digest(args.template),original_index_sha256=digest(original/'index.csv'),rehab24_manifest_sha256=digest(rpath),rehab26_manifest_sha256=digest(epath),pose_weights_sha256=r['weights_sha256'],quality_policy='Original 195 preserved; extra selected>=99%, COCO17 valid>=75%, no box jump touching rep, all four feature rows extractable',label_map={'0':'bad','1':'good'},features_sha256=digest(args.out_dir/'features_crop10.npy'),repetitions_sha256=digest(args.out_dir/'repetitions.csv'))
    (args.out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('\nAccepted by dataset/camera/label:\n',pd.crosstab([df.dataset,df.camera],df.label))
    print('\nQuality outcomes:\n',qa.groupby(['dataset','camera','accepted']).size().to_string())
    print('\nRejected extra repetitions:\n',qa.loc[~qa.accepted,'reason'].value_counts().to_string())
    print('Accepted view-reps:',len(df),'Unique annotated repetitions:',df.rep_id.nunique(),'Groups:',df.group.nunique(),'Feature shape:',X.shape)
    print('Saved:',args.out_dir)

if __name__=='__main__': main()
