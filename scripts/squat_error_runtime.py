"""NumPy-only squat error candidate runtime. Manual view policy; no pose/view estimator."""
import json
from pathlib import Path
import numpy as np
from deploy_lr_numpy import NumpyLR, compare, sha

def stat(a, q=.5):
    a=np.asarray(a); a=a[np.isfinite(a)]
    return float(np.quantile(a,q)) if len(a) else np.nan

def midpoint(a,b):
    return (a+b)/2

def angle(a,b,c):
    u=a-b; v=c-b
    den=np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)
    with np.errstate(invalid='ignore',divide='ignore'):
        return np.where(den>1e-6,np.degrees(np.arccos(np.clip(np.sum(u*v,axis=1)/den,-1,1))),np.nan)

def geometry(seq):
    k=np.asarray(seq,float); xy=k[:,:,:2].copy()
    xy[~(np.isfinite(k).all(2)&(k[:,:,2]>=.3))]=np.nan
    S,H,K,A=[midpoint(xy[:,l],xy[:,r]) for l,r in [(5,6),(11,12),(13,14),(15,16)]]
    n=len(k); m=max(3,int(np.ceil(.15*n))); start=slice(0,m)
    leg=np.concatenate([np.linalg.norm(xy[:,h]-xy[:,j],axis=1)+np.linalg.norm(xy[:,j]-xy[:,a],axis=1) for h,j,a in [(11,13,15),(12,14,16)]])
    scale=stat(leg)
    if not np.isfinite(scale) or scale<=10: raise ValueError('No usable leg scale')
    hip=H[:,1].copy(); ok=np.isfinite(hip)
    if ok.sum()<3: raise ValueError('No usable hip trajectory')
    v=np.interp(np.arange(n),np.flatnonzero(ok),hip[ok]); sm=np.array([np.median(v[max(0,i-3):i+4]) for i in range(n)])
    b=int(np.argmax(sm)); radius=max(2,int(round(.05*n))); bottom=slice(max(0,b-radius),min(n,b+radius+1))
    def sep(l,r): return np.abs(xy[:,l,0]-xy[:,r,0])/scale
    signals={'hip_height_rel_ankle':(A[:,1]-H[:,1])/scale,
             'torso_x_abs':np.abs(S[:,0]-H[:,0])/scale,
             'torso_x_signed':(S[:,0]-H[:,0])/scale,
             'shoulder_level_abs':np.abs(xy[:,5,1]-xy[:,6,1])/scale,
             'hip_level_abs':np.abs(xy[:,11,1]-xy[:,12,1])/scale,
             'knee_width':sep(13,14),'ankle_width':sep(15,16),
             'knee_minus_ankle_width':sep(13,14)-sep(15,16),
             'knee_center_offset':np.abs(K[:,0]-A[:,0])/scale}
    width=sep(15,16)
    signals['knee_ankle_ratio']=np.divide(sep(13,14),width,out=np.full(n,np.nan),where=width>=.15)
    # Absolute projected offsets are proxies, not foot/toe or heel measurements.
    for side,(h,j,a) in [('l',(11,13,15)),('r',(12,14,16))]:
        signals['knee_angle_'+side]=angle(xy[:,h],xy[:,j],xy[:,a])/180
        signals['knee_ankle_x_'+side]=np.abs(xy[:,j,0]-xy[:,a,0])/scale
        signals['ankle_y_'+side]=xy[:,a,1]/scale
        signals['hip_knee_y_'+side]=(xy[:,j,1]-xy[:,h,1])/scale
        # inward movement relative to ankle midpoint, masked for overlapping ankles.
        inward=(np.abs(xy[:,a,0]-A[:,0])-np.abs(xy[:,j,0]-A[:,0]))/scale
        inward[~(width>=.15)]=np.nan
        signals['inward_'+side]=inward
    # Symmetric aggregates avoid reliance on left vs right anatomy; use both
    # signed change and absolute change from the start of the repetition.
    features={}
    for key,x in signals.items():
        ref=stat(x[start]); delta=x-ref
        for suffix,val in [('delta_p10',stat(delta,.1)),('delta_p90',stat(delta,.9)),('abs_delta_p90',stat(np.abs(delta),.9)),('bottom_delta',stat(delta[bottom])),('range',stat(x,.9)-stat(x,.1))]:
            features[key+'_'+suffix]=val
        # No absolute pixel y location: ankle_y only relative temporal features.
        if not key.startswith('ankle_y_'):
            features[key+'_bottom']=stat(x[bottom])
    return features

THRESHOLD = .9
VIEWS = {'ERR_KNEES_INSIDE':'frontal','ERR_TORSO_TO_SIDE':'frontal','ERR_SHALLOW_SQUAT':'oblique_profile'}

class SquatErrorRuntime:
    def __init__(self, path):
        self.bundle=json.loads(Path(path).read_text())
        if self.bundle['schema_version']!=1 or set(self.bundle['models'])!=set(VIEWS):
            raise ValueError('Unexpected error bundle schema')
        self.models={}
        for code, v in VIEWS.items():
            spec=self.bundle['models'][code]
            if spec['view_policy']!=v or spec['threshold']!=THRESHOLD or spec['classes']!=[False, True]:
                raise ValueError('Unexpected model policy: '+code)
            model=NumpyLR(spec)
            names=spec['feature_names']
            if len(names)!=model.n_features_in_ or len(set(names))!=len(names):
                raise ValueError('Invalid feature order')
            self.models[code]=(model,names)

    def classify(self, seq, posture, view):
        # Do not compute features or call the error models for Good/unknown.
        result=dict(issues=[],error_scores={},error_status='not_applicable')
        if posture!='bad': return result
        if view not in ('frontal','oblique_profile'):
            result['error_status']='view_unknown'; return result
        try:
            f=geometry(seq)
        except (ValueError, IndexError, TypeError):
            result['error_status']='unassessable'; return result
        for code,(model,names) in self.models.items():
            if VIEWS[code]!=view: continue
            x=np.array([[f[n] for n in names]],dtype=np.float64)
            if not np.isfinite(x).any() or np.isinf(x).any(): continue
            score=float(model.predict_proba(x)[0,1])
            result['error_scores'][code]=score
            if score>=THRESHOLD: result['issues'].append(code)
        result['error_status']=('identified' if result['issues'] else 'undetermined' if result['error_scores'] else 'unassessable')
        return result

def verify(path):
    runtime=SquatErrorRuntime(path)
    for code,(model,names) in runtime.models.items():
        g=runtime.bundle['golden'][code]
        x=np.asarray(g['x'],dtype=np.float64)
        p=np.asarray(g['probabilities'],dtype=np.float64)
        y=np.asarray(g['labels'])
        stats=compare(model,x,p,y,code)
        assert np.array_equal(model.predict_proba(x)[:,1]>=THRESHOLD,np.asarray(g['alerts'],bool))
        print(code,stats,'alert parity PASS',flush=True)
    assert runtime.classify(None,'good','frontal')['error_status']=='not_applicable'
    assert runtime.classify(None,'bad','unknown')['error_status']=='view_unknown'
    print('PASS: squat error NumPy bundle; SHA256',sha(path))

if __name__=='__main__':
    import sys
    if len(sys.argv)!=2: raise SystemExit('Usage: python squat_error_runtime.py BUNDLE.json')
    verify(sys.argv[1])
