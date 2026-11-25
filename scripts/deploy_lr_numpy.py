#!/usr/bin/env python3
"""Export trusted deployment joblib pipelines to JSON; verify NumPy inference.
Export on training PC only. Verify needs only Python 3.11 and NumPy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def finite_list(x):
    a = np.asarray(x, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite model parameter')
    return a.tolist()


class NumpyLR:
    def __init__(self, spec):
        self.classes_ = np.asarray(spec['classes'])
        self.n_features_in_ = int(spec['feature_dim'])
        self.mean = np.asarray(spec['mean'], dtype=np.float64)
        self.scale = np.asarray(spec['scale'], dtype=np.float64)
        self.coef = np.asarray(spec['coef'], dtype=np.float64)
        self.intercept = np.asarray(spec['intercept'], dtype=np.float64)
        self.impute = None if spec['impute'] is None else np.asarray(spec['impute'], dtype=np.float64)
        d = self.n_features_in_
        n = 1 if len(self.classes_) == 2 else len(self.classes_)
        if self.mean.shape != (d,) or self.scale.shape != (d,) or self.coef.shape != (n,d) or self.intercept.shape != (n,):
            raise ValueError('Bad parameter dimensions')
        if self.impute is not None and self.impute.shape != (d,):
            raise ValueError('Bad imputer dimension')
        for a in (self.mean,self.scale,self.coef,self.intercept,self.impute):
            if a is not None and not np.isfinite(a).all():
                raise ValueError('Nonfinite parameters')
        if np.any(self.scale <= 0):
            raise ValueError('Nonpositive scaling')

    def predict_proba(self, X):
        x = np.asarray(X, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_:
            raise ValueError('Expected [N, feature_dim]')
        if np.isinf(x).any():
            raise ValueError('Infinite input')
        if self.impute is not None:
            x = np.where(np.isnan(x), self.impute, x)
        if not np.isfinite(x).all():
            raise ValueError('Missing input without imputer')
        z = ((x-self.mean)/self.scale) @ self.coef.T + self.intercept
        if len(self.classes_) == 2:
            v = z[:,0]
            # Stable sigmoid, including large negative scores.
            e = np.exp(-np.abs(v))
            p = np.where(v >= 0, 1/(1+e), e/(1+e))
            return np.column_stack((1-p,p))
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def export_spec(pipe, expected_dim, expected_classes):
    steps = [s for _,s in pipe.steps]
    names = [type(s).__name__ for s in steps]
    if names not in (['StandardScaler','LogisticRegression'], ['SimpleImputer','StandardScaler','LogisticRegression']):
        raise ValueError(f'Unsupported pipeline: {names}')
    scaler, lr = steps[-2:]
    if not scaler.with_mean or not scaler.with_std:
        raise ValueError('Unsupported scaler configuration')
    if lr.classes_.tolist() != expected_classes:
        raise ValueError(f'Unexpected class order: {lr.classes_}')
    if getattr(lr,'multi_class',None) == 'ovr' and len(expected_classes)>2:
        raise ValueError('OVR unsupported')
    impute = None
    if len(steps)==3:
        imp=steps[0]
        if imp.strategy!='median' or imp.add_indicator or not np.isnan(imp.missing_values):
            raise ValueError('Unsupported imputation configuration')
        impute=finite_list(imp.statistics_)
    spec=dict(feature_dim=expected_dim, classes=expected_classes, impute=impute,
              mean=finite_list(scaler.mean_),scale=finite_list(scaler.scale_),
              coef=finite_list(lr.coef_),intercept=finite_list(lr.intercept_))
    NumpyLR(spec)
    return spec


def compare(model, x, p, labels, tag):
    got=model.predict_proba(x)
    err=float(np.max(np.abs(got-p)))
    if not np.isfinite(err) or err>1e-9:
        raise AssertionError(f'{tag}: probability error {err}')
    mismatch=int(np.count_nonzero(model.predict(x)!=labels))
    if mismatch:
        raise AssertionError(f'{tag}: {mismatch} label mismatches')
    return dict(rows=len(x), max_prob_error=err, label_mismatches=mismatch)


def null_rows(a):
    return [[None if np.isnan(v) else float(v) for v in row] for row in a]


def export(root, output):
    import joblib
    root=Path(root)
    dep=root/'results/deployment_candidates'
    manifest=json.loads((dep/'manifest.json').read_text())
    exercise_x=np.load(root/'results/exercise_baseline/2s/features.npy',allow_pickle=False)
    # Verify all original multiview features against squat pipeline. This includes
    # held-out/external feature rows for parity only, not a performance evaluation.
    squat_x=np.load(root/'results/squat_multiview/build/features_crop10.npy',allow_pickle=False).reshape(-1,180)
    if exercise_x.shape!=(26018,156) or squat_x.shape!=(8192,180):
        raise ValueError(f'Unexpected feature shapes: {exercise_x.shape}, {squat_x.shape}')
    for file,expected in [(root/'results/exercise_baseline/2s/features.npy',manifest['exercise_training']['inputs']['features.npy']['sha256']),
                          (root/'results/squat_multiview/build/features_crop10.npy',manifest['squat_training']['features_sha256'])]:
        if sha(file)!=expected: raise ValueError(f'Feature hash mismatch: {file}')
    out=dict(schema_version=1,models={},golden={},source_versions=manifest['versions'],configuration=manifest['configuration'])
    for name,d,classes,x in [('exercise',156,['Other','Pushup','Squat'],exercise_x),('squat',180,[0,1],squat_x)]:
        info=manifest['models'][name]
        path=dep/info['file']
        if sha(path)!=info['sha256']: raise ValueError(f'Model hash mismatch: {path}')
        pipe=joblib.load(path)['pipeline']
        spec=export_spec(pipe,d,classes)
        spec['source_sha256']=info['sha256']
        model=NumpyLR(spec)
        xx=np.asarray(x,dtype=np.float64)
        p=pipe.predict_proba(xx); y=pipe.predict(xx)
        stats=compare(model,xx,p,y,name+' full features float64')
        # Actual source dtype comparison is recorded separately (scaler rounding).
        native=pipe.predict_proba(x)
        stats['source_dtype']=str(x.dtype)
        stats['native_dtype_max_prob_difference']=float(np.max(np.abs(native-p)))
        stats['native_dtype_label_mismatches']=int(np.count_nonzero(pipe.predict(x)!=y))
        # Balanced deterministic sample across predicted classes, plus missing-data tests.
        ids=[]
        for c in classes:
            idx=np.flatnonzero(y==c)
            if len(idx): ids.extend(idx[np.linspace(0,len(idx)-1,min(32,len(idx)),dtype=int)].tolist())
        test=xx[sorted(set(ids))].copy()
        if spec['impute'] is not None:
            extra=test[:min(8,len(test))].copy(); extra[:,::7]=np.nan
            test=np.concatenate([test,extra,np.full((1,d),np.nan)])
        pp=pipe.predict_proba(test); yy=pipe.predict(test)
        compare(model,test,pp,yy,name+' golden')
        out['models'][name]=spec
        out['golden'][name]=dict(x=null_rows(test),probabilities=finite_list(pp),labels=yy.tolist(),export_check=stats)
        print(name,json.dumps(stats),flush=True)
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    # JSON null encodes NaN features; no pickle is used on the board.
    with output.open('x',encoding='utf-8') as f:
        json.dump(out,f,allow_nan=False,separators=(',',':'))
    print('Saved:',output,'sha256:',sha(output))


def verify(path):
    b=json.loads(Path(path).read_text())
    if b['schema_version']!=1: raise ValueError('Unsupported schema')
    for name in ('exercise','squat'):
        model=NumpyLR(b['models'][name]); g=b['golden'][name]
        result=compare(model,np.asarray(g['x'],dtype=np.float64),np.asarray(g['probabilities']),np.asarray(g['labels']),name+' golden')
        print(name,json.dumps(result),flush=True)
    print('PASS: both models; NumPy',np.__version__,'bundle sha256',sha(path))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    s=p.add_subparsers(dest='mode',required=True)
    e=s.add_parser('export'); e.add_argument('--root',type=Path,default=Path('.')); e.add_argument('--output',type=Path,default=Path('results/deployment_candidates/lr_numpy_bundle.json'))
    v=s.add_parser('verify'); v.add_argument('bundle',type=Path)
    a=p.parse_args()
    if a.mode=='export': export(a.root,a.output)
    else: verify(a.bundle)

if __name__=='__main__': main()
