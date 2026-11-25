#!/usr/bin/env python3
"""Generate a NumPy-only fixed-FPS runtime from the verified offline source.
Build/check on WSL; verify on the board. No original source or model is overwritten.
Run from the repository root with this script under scripts/.
"""
from __future__ import annotations
import argparse
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / 'scripts'
OUT = S / 'live_components'
DEP = ROOT / 'results/deployment_candidates'
FIXTURE = ROOT / 'results/live_runtime_check/fixture.npz'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def extract(file, names):
    src = (S/file).read_text(encoding='utf-8')
    tree = ast.parse(src)
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            keys = [node.name]
        elif isinstance(node, ast.Assign):
            keys = [n.id for target in node.targets for n in ast.walk(target) if isinstance(n, ast.Name)]
        else:
            continue
        for key in keys:
            if key in names:
                found[key] = node
    if set(found) != set(names):
        raise RuntimeError(f'{file}: missing {set(names)-set(found)}')
    nodes = sorted({id(n):n for n in found.values()}.values(), key=lambda n:n.lineno)
    return '\n\n'.join(ast.get_source_segment(src,n) for n in nodes)+'\n'


def build():
    if OUT.exists():
        raise FileExistsError(f'{OUT} exists; review/remove only generated directory before rebuilding')
    manifest = json.loads((DEP/'manifest.json').read_text())
    if sha(S/'train_exercise_baseline.py') != manifest['exercise_training']['feature_source_sha256']:
        raise RuntimeError('Exercise feature source differs from training manifest')
    modules = {}
    def add(name, imports, parts, tail=''):
        modules[name] = '# Generated from frozen offline source by prepare_board_runtime.py.\nfrom __future__ import annotations\n'+imports+'\n\n'+'\n'.join(extract(f,n) for f,n in parts)+tail
    add('exercise_features.py','import numpy as np', [('train_exercise_baseline.py',['longest_bad','features'])])
    add('detector.py','import numpy as np\nfrom collections import deque\nfrom . import exercise_features as baseline', [('eval_ai_exercise_stream.py',['AIExerciseDetector']),('eval_session_policy.py',['ExitHysteresisDetector'])])
    add('squat_base.py','import numpy as np', [('train_squat_baseline.py',['LS','RS','LH','RH','LK','RK','LA','RA','distance','mid','angle','stats','extract_features','mid_scalar'])])
    # Preserve the complete feature extractor, including unused phase groups.
    # The original pandas centered rolling median is replaced by equivalent
    # NumPy for the finite interpolated series. Only base/asymmetry reach LR180.
    median = '''\ndef running_median(x, width=7):
    x = np.asarray(x, dtype=float)
    if width != 7 or not np.isfinite(x).all():
        raise ValueError('Expected finite hip interpolation and width=7')
    return np.array([np.median(x[max(0,i-3):min(len(x),i+4)]) for i in range(len(x))])
'''
    add('squat_features.py','import numpy as np\nfrom . import squat_base as base', [('train_squat_extended.py',['LS','RS','LH','RH','LK','RK','LA','RA','PHASES','EVENTS','safe_mean','safe_stat','stats5','finite_interp','get_phase_indices','near_value','extract_feature_groups'])], median)
    add('squat_classifier.py','import numpy as np\nfrom .squat_features import extract_feature_groups',[('squat_classification.py',['classify'])])
    add('pushup_counter.py','import numpy as np\nfrom collections import deque',[('eval_pushup_counter_pilot.py',['DepthCounter','PushupCounter'])])
    add('pushup_features.py','import numpy as np',[('train_pushup_posture.py',['SIGNALS','STATS','FEATURE_NAMES','angle','mean_sides','stats','extract_features','quality'])])
    add('pushup_branch.py','import numpy as np\nfrom collections import deque\nfrom .pushup_counter import PushupCounter\nfrom .pushup_features import extract_features, quality, FEATURE_NAMES',[('probe_pushup_feedback.py',['THRESHOLDS']),('eval_pushup_gated_stream.py',['feedback','Branch'])])
    add('unified.py','import numpy as np\nfrom collections import deque\nfrom auto_detect_hailo_bridge import AutoDetectPipeline\nfrom .pushup_branch import Branch',[('eval_unified_exercise_stream_v2.py',['UnifiedExercisePipeline'])])
    modules['__init__.py']='"""Generated inference-only components."""\n'
    for name,src in modules.items(): compile(src,name,'exec')
    OUT.mkdir()
    for name,src in modules.items(): (OUT/name).write_text(src,encoding='utf-8')
    sources=['train_exercise_baseline.py','eval_ai_exercise_stream.py','eval_session_policy.py','train_squat_baseline.py','train_squat_extended.py','squat_classification.py','eval_pushup_counter_pilot.py','train_pushup_posture.py','probe_pushup_feedback.py','eval_pushup_gated_stream.py','eval_unified_exercise_stream_v2.py','auto_detect_hailo_bridge.py','exercise_detector.py','squat_stream_pipeline.py','squat_rep_buffer.py','stream_squat_core.py','deploy_lr_numpy.py']
    (OUT/'source_manifest.json').write_text(json.dumps({'sources':{p:sha(S/p) for p in sources},'generated':{p:sha(OUT/p) for p in modules}},indent=2)+'\n')
    print('Generated',OUT)


def models():
    from deploy_lr_numpy import NumpyLR
    m=json.loads((DEP/'manifest.json').read_text()); b=json.loads((DEP/'lr_numpy_bundle.json').read_text())
    if b['schema_version']!=1 or b['configuration']!=m['configuration']:
        raise ValueError('Model configuration mismatch')
    for n in ('exercise','squat'):
        if b['models'][n]['source_sha256'] != m['models'][n]['sha256']:
            raise ValueError('Model source mismatch '+n)
    return {'pipeline':NumpyLR(b['models']['exercise'])}, {'pipeline':NumpyLR(b['models']['squat']),'feature_names':m['squat_training']['feature_names']}


def verify_sources():
    m=json.loads((OUT/'source_manifest.json').read_text())
    for base,kind in [(S,'sources'),(OUT,'generated')]:
        for p,h in m[kind].items():
            if sha(base/p)!=h: raise ValueError('Runtime source hash mismatch '+p)


def run(k,sel,fps,logpath,reference=False):
    from squat_stream_pipeline import SquatStreamPipeline
    if reference:
        from eval_unified_exercise_stream_v2 import UnifiedExercisePipeline
        from eval_session_policy import ExitHysteresisDetector
        from squat_classification import classify
    else:
        from live_components.unified import UnifiedExercisePipeline
        from live_components.detector import ExitHysteresisDetector
        from live_components.squat_classifier import classify
    exercise,squat=models()
    def factory():
        return SquatStreamPipeline(artifact=squat,fps=fps,bottom_depth=.2,bottom_knee=120,pre_frames=16,post_frames=22,classifier=classify)
    # Never inherit stale experiment overrides from the shell.
    os.environ['SQUAT_AUTO_TIMEBASE']='file'
    events=[]; active=[]; tic=time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        p=UnifiedExercisePipeline(factory(),fps,events_path=logpath,kp_dump_path='',make_base=factory,gating=1,detector=ExitHysteresisDetector(exercise,fps,2.5),replay_sec=6.,lying_frames=5,gap_frames=max(10,int(fps*.5)))
        p.lost_sec=3.
        for i,(x,s) in enumerate(zip(k,sel)):
            events.extend(p.push_selected(x,bool(s),frame_id=i)); active.append(p.active_exercise)
        events.extend(p.flush())
    elapsed=time.perf_counter()-tic
    transitions=[json.loads(l) for l in logpath.read_text().splitlines()]
    return {'squat':events,'pushup':p.push_events,'active':active,'transitions':transitions},elapsed


def equal(a,b,path='root'):
    if isinstance(a,dict) and isinstance(b,dict):
        if a.keys()!=b.keys(): raise AssertionError(f'{path}: keys differ')
        for k in a: equal(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,list) and isinstance(b,list):
        if len(a)!=len(b): raise AssertionError(f'{path}: length {len(a)} vs {len(b)}')
        for i,(x,y) in enumerate(zip(a,b)): equal(x,y,f'{path}[{i}]')
    elif isinstance(a,(float,np.floating)) or isinstance(b,(float,np.floating)):
        if not np.isclose(a,b,atol=1e-9,rtol=1e-9,equal_nan=True): raise AssertionError(f'{path}: {a} vs {b}')
    elif a!=b: raise AssertionError(f'{path}: {a} vs {b}')


def cached_examples():
    from auto_detect_hailo_bridge import read_kp_dump
    from preprocess_rehab24 import cache_paths,check_cache
    def load(v,m,mask):
        if check_cache(v,m['weights_sha256'])!='complete': raise ValueError('Cache incomplete '+v['sample_id'])
        paths=cache_paths(v); d=read_kp_dump(paths[0])
        with np.load(paths[2],allow_pickle=False) as z: sel=(z['status']==0)&(d['pose_valid']!=0)
        k=np.array(d['kp'],copy=True)
        if len(k)!=int(v['frames']) or len(sel)!=len(k) or not np.array_equal(d['frame_id'],np.arange(len(k))): raise ValueError('Frame mismatch')
        if mask: k[k[:,:,2]<.3,:2]=np.nan
        k[~sel,:,:2]=np.nan; k[~sel,:,2]=0
        return k,sel,float(v['fps'])
    r=json.loads((ROOT/'data/preprocess/rehab24/batch_manifest.json').read_text())
    m=json.loads((ROOT/'data/preprocess/extra_datasets/manifest.json').read_text())
    for sid,manifest,mask in [('Ex6_Camera17_PM_008',r,True),('Ex3_Camera17_PM_010',r,True),('MV7_subject_001_push_up_good_side',m,False)]:
        v=next(v for v in manifest['videos'] if v['sample_id']==sid)
        yield sid,*load(v,manifest,mask)


def check():
    verify_sources()
    from threadpoolctl import threadpool_limits
    fixture={}; expected={}; output=FIXTURE.parent
    output.mkdir(parents=True,exist_ok=True)
    if FIXTURE.exists(): raise FileExistsError(FIXTURE)
    with threadpool_limits(limits=1):
        for i,(sid,k,sel,fps) in enumerate(cached_examples()):
            ref,_=run(k,sel,fps,output/f'{i}_reference.jsonl',True)
            got,elapsed=run(k,sel,fps,output/f'{i}_runtime.jsonl')
            equal(ref,got,sid)
            if any('classification_error' in e for e in got['squat']): raise RuntimeError('Squat classification failure '+sid)
            if i==0 and not got['squat'] or i==2 and not got['pushup']: raise RuntimeError('Missing target events '+sid)
            fixture.update({f'k{i}':k,f'sel{i}':sel,f'fps{i}':np.array(fps)})
            expected[str(i)]={'sample':sid,'events':ref}
            print('PASS',sid,'frames',len(k),'squat',len(got['squat']),'pushup',len(got['pushup']),'seconds',round(elapsed,3),flush=True)
    fixture['expected']=np.array(json.dumps(expected))
    fixture['bundle_sha256']=np.array(sha(DEP/'lr_numpy_bundle.json'))
    np.savez_compressed(FIXTURE,**fixture)
    print('Saved private replay fixture:',FIXTURE,'sha256:',sha(FIXTURE))


def verify(path):
    verify_sources()
    # Reject accidental training library imports in the deployment process.
    with np.load(path,allow_pickle=False) as z:
        if str(z['bundle_sha256'])!=sha(DEP/'lr_numpy_bundle.json'): raise ValueError('Fixture model hash mismatch')
        expected=json.loads(str(z['expected']))
        output=ROOT/'results/live_runtime_board'; output.mkdir(parents=True,exist_ok=True)
        for i,ref in expected.items():
            got,elapsed=run(z[f'k{i}'],z[f'sel{i}'],float(z[f'fps{i}']),output/f'{i}_runtime.jsonl')
            equal(ref['events'],got,ref['sample'])
            print('PASS',ref['sample'],'frames',len(z[f'k{i}']),'squat',len(got['squat']),'pushup',len(got['pushup']),'replay_fps',round(len(z[f'k{i}'])/elapsed,2),flush=True)
    imported=[n for n in ('pandas','sklearn','joblib','matplotlib','torch','ultralytics') if n in sys.modules]
    if imported: raise RuntimeError('Training libraries imported: '+str(imported))
    print('PASS: fixed-FPS keypoint-to-event parity; NumPy',np.__version__)


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='mode',required=True)
    sub.add_parser('build'); sub.add_parser('check'); v=sub.add_parser('verify'); v.add_argument('fixture',type=Path)
    a=p.parse_args(); os.chdir(ROOT)
    if a.mode=='build': build()
    elif a.mode=='check': check()
    else: verify(a.fixture)

if __name__=='__main__': main()
