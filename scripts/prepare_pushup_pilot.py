#!/usr/bin/env python3
"""Inspect MV7 side-view push-up caches before designing a repetition counter.
Run from pose_training_offline with this script in scripts/. No YOLO inference,
no changes to source videos, models, or caches. Clip labels are NOT rep GT.
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORK = Path('data/preprocess/extra_datasets')

def read(name):
    return json.loads((WORK/name).read_text(encoding='utf-8'))

def angle(a,b,c):
    u,v=a-b,c-b
    den=np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)
    out=np.full(len(a),np.nan)
    ok=np.isfinite(den)&(den>1e-6)
    out[ok]=np.degrees(np.arccos(np.clip(np.sum(u[ok]*v[ok],axis=1)/den[ok],-1,1)))
    return out

def mean_sides(a,b):
    x=np.stack([a,b],axis=1)
    good=np.isfinite(x); count=good.sum(1)
    return np.divide(np.where(good,x,0).sum(1),count,out=np.full(len(a),np.nan),where=count>0)

def signals(k,status):
    xy=np.array(k[:,:,:2],dtype=float,copy=True)
    valid=np.isfinite(xy).all(-1)&np.isfinite(k[:,:,2])&(k[:,:,2]>=.3)&(status[:,None]==0)
    xy[~valid]=np.nan
    scales=np.stack([np.linalg.norm(xy[:,s]-xy[:,h],axis=1) for s,h in [(5,11),(6,12)]],axis=1)
    usable=scales[np.isfinite(scales)&(scales>10)]
    if not len(usable): raise ValueError('No usable torso scale')
    scale=float(np.median(usable))
    data={}
    for side,(s,e,w,h,a) in [('left',(5,7,9,11,15)),('right',(6,8,10,12,16))]:
        data[f'elbow_{side}_deg']=angle(xy[:,s],xy[:,e],xy[:,w])
        data[f'hip_{side}_deg']=angle(xy[:,s],xy[:,h],xy[:,a])
        data[f'shoulder_y_{side}_norm']= (xy[:,s,1]-xy[:,a,1])/scale
        # Signed vertical deviation from line shoulder->ankle (image y down).
        d=xy[:,a]-xy[:,s]; norm=np.linalg.norm(d,axis=1)
        # Orient the line left-to-right for a consistent sign in image space.
        d=d*np.where(d[:,0]<0,-1.,1.)[:,None]
        r=xy[:,h]-xy[:,s]
        cross=d[:,0]*r[:,1]-d[:,1]*r[:,0]
        data[f'hip_offset_{side}_norm']=np.divide(cross,norm*scale,out=np.full(len(k),np.nan),where=np.isfinite(norm)&(norm>1e-6))
    for prefix,suffix in [('elbow','deg'),('hip','deg'),('shoulder_y','norm'),('hip_offset','norm')]:
        data[f'{prefix}_mean_{suffix}']=mean_sides(data[f'{prefix}_left_{suffix}'],data[f'{prefix}_right_{suffix}'])
    return pd.DataFrame(data),scale

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs',type=int,default=3)
    p.add_argument('--out-dir',type=Path,default=Path('results/pushup/pilot'))
    a=p.parse_args()
    if a.pairs<1: p.error('--pairs must be positive')
    import preprocess_rehab24 as pre
    from extract_video_kp_cache import KP_DUMP_DTYPE
    m=read('manifest.json'); policy=read('mv7_exclusion_policy.json')
    excluded=set(policy['excluded_sample_ids']); groups=policy['person_fold_groups']
    if len(excluded)!=67: raise ValueError('Unexpected duplicate exclusion policy')
    q=pd.read_csv(WORK/'mv7_cached_body_quality.csv')
    if len(q)!=129 or q.sample_id.duplicated().any(): raise ValueError('Expected updated 129-row MV7 quality inventory')
    passed=set(q.loc[q.quality_pass.astype(str).str.lower()=='true','sample_id'])
    rows=[]; eligible=defaultdict(dict); signals_by_id={}
    for v in sorted(m['videos'],key=lambda v:v['sample_id']):
        if v['dataset']!='mv7' or v['exercise']!='push_up' or v['camera']!='side': continue
        sid=v['sample_id']; person=v['person_id']; label=int(v['clip_correctness'])
        row=dict(sample_id=sid,person_id=person,group='mv7:'+groups[person],clip_label=label,frames=v['frames'],fps=v['fps'],video_path=v['path'])
        reason=''; df=None
        if sid in excluded: reason='exact_duplicate_group'
        elif sid not in passed: reason='not_in_cached_quality_pass'
        else:
            state=pre.check_cache(v,m['weights_sha256'])
            if state!='complete': raise RuntimeError(f'{sid}: {state}')
            paths=pre.cache_paths(v)
            raw=np.memmap(paths[0],dtype=KP_DUMP_DTYPE,mode='r')
            if not np.array_equal(raw['frame_id'],np.arange(len(raw))): raise ValueError(f'{sid}: bad frame indices')
            k=np.asarray(raw['kp'])
            with np.load(paths[2],allow_pickle=False) as z:
                status=z['status']; boxes=z['selected_box']
            sel=status==0
            center=(boxes[:,:2]+boxes[:,2:])/2
            diag=np.maximum(np.linalg.norm(boxes[:,2:]-boxes[:,:2],axis=1),1)
            jumps=sel[:-1]&sel[1:]&(np.linalg.norm(np.diff(center,axis=0),axis=1)/diag[:-1]>=.25)
            valid=np.isfinite(k[:,:,:2]).all(-1)&np.isfinite(k[:,:,2])&(k[:,:,2]>=.3)&sel[:,None]
            chain=valid[:,[5,7,9,11,13,15]].all(1)|valid[:,[6,8,10,12,14,16]].all(1)
            edges=np.flatnonzero(np.diff(np.r_[False,~chain,False])); gaps=edges[1::2]-edges[::2]
            gap=float(max(gaps,default=0)/v['fps'])
            row.update(selected_pct=100*sel.mean(),chain_valid_pct=100*chain.mean(),longest_chain_gap_sec=gap,large_jumps=int(jumps.sum()))
            if jumps.any(): reason='large_box_jump'
            elif sel.mean()<.95 or chain.mean()<.90 or gap>.5: reason='recomputed_quality_fail'
            else:
                try: df,scale=signals(k,status)
                except ValueError as e: reason=str(e)
                else:
                    df.insert(0,'frame_id',np.arange(len(k))); df.insert(1,'t_sec',np.arange(len(k))/v['fps'])
                    df['selected']=sel; df['chain_valid']=chain
                    row['torso_scale_px']=scale
                    row['elbow_min_deg']=float(df.elbow_mean_deg.min()); row['elbow_max_deg']=float(df.elbow_mean_deg.max())
                    if label in eligible[person]: raise ValueError(f'Duplicate side clip label: {person}')
                    eligible[person][label]=row; signals_by_id[sid]=df
        row['eligible']=not reason; row['reason']=reason; rows.append(row)
    # Deterministic paired subjects, one pair per conservative fold group.
    selected=[]; used=set()
    for person,pair in sorted(eligible.items()):
        if set(pair)!={0,1}: continue
        g=pair[0]['group']
        if g in used: continue
        used.add(g); selected.extend([pair[1],pair[0]])
        if len(used)==a.pairs: break
    a.out_dir.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(rows).to_csv(a.out_dir/'side_inventory.csv',index=False)
    if len(used)!=a.pairs: raise RuntimeError(f'Only {len(used)} distinct paired groups; see side_inventory.csv; use a new out-dir and fewer --pairs')
    pilot=pd.DataFrame(selected); pilot.to_csv(a.out_dir/'pilot_inventory.csv',index=False)
    # User can independently watch each original clip and record counts.
    pilot[['sample_id','video_path']].assign(manual_rep_count='',notes='').to_csv(a.out_dir/'manual_counts.csv',index=False)
    for r in selected:
        sid=r['sample_id']; df=signals_by_id[sid]; df.to_csv(a.out_dir/f'{sid}.csv',index=False)
        fig,axes=plt.subplots(4,1,figsize=(12,9),sharex=True)
        f=df.frame_id
        for ax,(prefix,suffix,ylabel) in zip(axes,[('elbow','deg','Elbow angle (deg)'),('shoulder_y','norm','Shoulder Y relative to ankle / torso'),('hip','deg','Shoulder-hip-ankle (deg)'),('hip_offset','norm','Hip offset / torso')]):
            for side in ['left','right','mean']:
                vals=df[f'{prefix}_{side}_{suffix}']; ax.plot(f,vals,label=side,alpha=.85 if side=='mean' else .45,lw=1.6 if side=='mean' else .8)
            ax.set_ylabel(ylabel); ax.grid(alpha=.25); ax.legend(loc='upper right',ncol=3)
        axes[-1].set_xlabel('Frame ID (zero-based)')
        fig.suptitle(f'{sid} | clip label={r["clip_label"]} | {r["fps"]:.2f} FPS | no rep GT')
        fig.tight_layout(); fig.savefig(a.out_dir/f'{sid}.png',dpi=130); plt.close(fig)
    with ZipFile(a.out_dir/'pilot_diagnostics.zip','w',ZIP_DEFLATED) as z:
        for file in sorted(a.out_dir.iterdir()):
            if file.suffix in ('.csv','.png'): z.write(file,file.name)
    print('Eligible side clips:',sum(r['eligible'] for r in rows),'paired subjects:',sum(set(p)=={0,1} for p in eligible.values()))
    print(pilot[['sample_id','frames','fps','chain_valid_pct','elbow_min_deg','elbow_max_deg']].to_string(index=False))
    print('Saved:',a.out_dir/'pilot_diagnostics.zip')

if __name__=='__main__': main()
