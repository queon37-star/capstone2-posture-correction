#!/usr/bin/env python3
"""Fit the selected three error classifiers and export NumPy-only deployment bundle.
Run on WSL with prior selective experiment scripts and cache. Does not change existing models.
"""
import argparse, ast, inspect, json
from pathlib import Path
import numpy as np
import joblib
from train_squat_error_selective import SPECS, load_geometry, make_model
import train_squat_error_geometry as source
import squat_error_runtime as runtime
from deploy_lr_numpy import export_spec, NumpyLR, compare, null_rows, finite_list, sha


def check_geometry():
    for name in ('stat','midpoint','angle','geometry'):
        a=ast.dump(ast.parse(inspect.getsource(getattr(source,name))))
        b=ast.dump(ast.parse(inspect.getsource(getattr(runtime,name))))
        if a!=b: raise RuntimeError('Geometry source mismatch: '+name)


def fit_export(d,g,out):
    check_geometry()
    if len(d)!=len(g) or not d.index.equals(g.index): raise ValueError('Row alignment mismatch')
    if set(SPECS)!=set(runtime.VIEWS): raise ValueError('Unexpected specs')
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    bundle=dict(schema_version=1,scope='full-training candidate; not independent accuracy; annotated view filters; fixed 0.9 score threshold',models={},golden={})
    summary={}
    for code,(prefixes,view) in SPECS.items():
        if view!=runtime.VIEWS[code]: raise ValueError('View policy mismatch')
        mask=d.viewing_angle_deg.abs().le(30) if view=='frontal' else d.viewing_angle_deg.abs().gt(30)
        z=d.loc[mask]; names=[c for c in g.columns if any(c.startswith(p+'_') for p in prefixes)]
        expected={'ERR_KNEES_INSIDE':36,'ERR_TORSO_TO_SIDE':24,'ERR_SHALLOW_SQUAT':30}[code]
        if len(names)!=expected: raise ValueError('Feature count mismatch')
        x=g.loc[mask,names].to_numpy(dtype=np.float64); y=z.truth.eq(code).to_numpy()
        if not y.any() or y.all(): raise ValueError('Missing class')
        pipe=make_model().fit(x,y)
        path=out/(code+'.joblib')
        joblib.dump(dict(pipeline=pipe,feature_names=names,view_policy=view,threshold=runtime.THRESHOLD),path)
        saved=joblib.load(path)['pipeline']
        spec=export_spec(saved,len(names),[False,True]); spec.update(feature_names=names,view_policy=view,threshold=runtime.THRESHOLD,source_sha256=sha(path))
        model=NumpyLR(spec); prob=pipe.predict_proba(x); labels=pipe.predict(x)
        stats=compare(model,x,prob,labels,code+' all fit rows')
        assert np.array_equal(model.predict_proba(x)[:,1]>=runtime.THRESHOLD,prob[:,1]>=runtime.THRESHOLD)
        assert np.array_equal(saved.predict_proba(x),prob)
        ids=[]
        for cls in (False,True):
            idx=np.flatnonzero(y==cls); ids.extend(idx[np.linspace(0,len(idx)-1,min(32,len(idx)),dtype=int)])
        test=x[sorted(set(ids))].copy(); missing=test[:8].copy(); missing[:,::7]=np.nan
        test=np.concatenate([test,missing]); pp=pipe.predict_proba(test); yy=pipe.predict(test)
        compare(model,test,pp,yy,code+' golden')
        bundle['models'][code]=spec
        bundle['golden'][code]=dict(x=null_rows(test),probabilities=finite_list(pp),labels=yy.tolist(),alerts=(pp[:,1]>=runtime.THRESHOLD).tolist())
        summary[code]=dict(view_policy=view,feature_dim=len(names),training_views=len(z),positive_views=int(y.sum()),negative_views=int((~y).sum()),unique_reps=int(z.rep_id.nunique()),persons=int(z.person_id.nunique()),numpy_parity=stats,joblib_sha256=sha(path))
        print(code,json.dumps(summary[code]),flush=True)
    dest=out/'squat_error_numpy.json'
    with dest.open('x') as f: json.dump(bundle,f,allow_nan=False,separators=(',',':'))
    runtime.verify(dest)
    return dict(models=summary,bundle_sha256=sha(dest),runtime_sha256=sha(Path(runtime.__file__)),threshold=runtime.THRESHOLD,scope=bundle['scope'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-dir',type=Path,default=Path('results/squat_multiview/build'))
    p.add_argument('--annotation',type=Path,default=Path.home()/'capstone2/datasets/REHAB26-ViewAngles/Segmentation.csv')
    p.add_argument('--out-dir',type=Path,default=Path('results/squat_error_deployment'))
    args=p.parse_args()
    if args.out_dir.exists(): raise FileExistsError(args.out_dir)
    check_geometry(); d,g=load_geometry(args)
    result=fit_export(d,g,args.out_dir)
    result['source_hashes']={str(p):sha(p) for p in [args.annotation,args.build_dir/'manifest.json',Path(source.__file__),Path(inspect.getfile(load_geometry)),Path(inspect.getfile(make_model)),Path(__file__)]}
    (args.out_dir/'manifest.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('Saved:',args.out_dir,flush=True)

if __name__=='__main__': main()
