"""Synthetic verification of signed feature decomposition; no REHAB data needed."""
import sys
from pathlib import Path
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from inspect_squat_lopo_features import explain_vector

X = np.array([[0.1,1,0],[0.2,2,1],[1.1,0,0],[1.5,1,1],[1.8,2,0],
              [2.2,3,1],[0.3,0,0],[1.9,4,1]],dtype=float)
y = np.array([0,0,0,1,1,1,0,1])
p = Pipeline([('imputer',SimpleImputer()),('scaler',StandardScaler()),
              ('model',LogisticRegression(C=0.1))]).fit(X,y)
for row in X:
    e,b,d,prob,pred = explain_vector(p,row)
    assert np.isclose(b+np.sum(e), d)
    assert pred == p.predict(row.reshape(1,-1))[0]
    assert np.isclose(prob,p.predict_proba(row.reshape(1,-1))[0,1])
print('Feature evidence decomposition test: PASS (synthetic only)')
