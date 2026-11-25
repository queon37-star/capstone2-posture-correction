# Generated from frozen offline source by prepare_board_runtime.py.
from __future__ import annotations
import numpy as np

LS, RS, LH, RH, LK, RK, LA, RA = 5, 6, 11, 12, 13, 14, 15, 16

def distance(a, b):
    return np.linalg.norm(a-b, axis=-1)

def mid(a, b):
    # Treat missing points conservatively: one known side can form a center.
    both = np.isfinite(a).all(axis=-1) & np.isfinite(b).all(axis=-1)
    only_a = np.isfinite(a).all(axis=-1) & ~np.isfinite(b).all(axis=-1)
    only_b = np.isfinite(b).all(axis=-1) & ~np.isfinite(a).all(axis=-1)
    out = np.full_like(a, np.nan)
    out[both] = (a[both] + b[both]) * 0.5
    out[only_a] = a[only_a]
    out[only_b] = b[only_b]
    return out

def angle(a, b, c):
    """Angle ABC in degrees, invalid geometry -> NaN."""
    u, v = a-b, c-b
    nu, nv = np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1)
    den = nu*nv
    cos = np.full(len(den), np.nan)
    valid = np.isfinite(den) & (den > 1e-6)
    cos[valid] = np.sum(u[valid]*v[valid],axis=1) / den[valid]
    return np.degrees(np.arccos(np.clip(cos,-1,1)))

def stats(series, prefix):
    series = np.asarray(series, dtype=float)
    valid = series[np.isfinite(series)]
    keys = ['mean','std','min','max','q10','median','q90','range','first','last']
    if len(valid) == 0:
        return {f'{prefix}_{k}': np.nan for k in keys}
    # Percentages, not absolute frame counts, to avoid rep duration leakage.
    n = max(1, int(np.ceil(len(series) * 0.15)))
    def finite_mean(values):
        good = values[np.isfinite(values)]
        return float(good.mean()) if len(good) else np.nan
    vals = [float(valid.mean()),float(valid.std()),float(valid.min()),
            float(valid.max()),float(np.quantile(valid,0.10)),
            float(np.median(valid)),float(np.quantile(valid,0.90)),
            float(valid.max()-valid.min()),
            finite_mean(series[:n]),finite_mean(series[-n:])]
    return {f'{prefix}_{key}':value for key,value in zip(keys,vals)}

def extract_features(seq):
    """Repetition-level geometry summary; avoids absolute camera location/size."""
    if seq.ndim != 3 or seq.shape[1:] != (17,3) or len(seq) < 10:
        raise ValueError(f'Unexpected keypoint shape {seq.shape}')
    xy = np.array(seq[:,:,:2],dtype=float,copy=True)
    cf = np.asarray(seq[:,:,2], dtype=float)
    xy[(cf < 0.30) | ~np.isfinite(cf)] = np.nan

    shoulder = mid(xy[:,LS],xy[:,RS])
    hip = mid(xy[:,LH],xy[:,RH])
    ankle = mid(xy[:,LA],xy[:,RA])

    torso = distance(shoulder,hip)
    good_torso = torso[np.isfinite(torso) & (torso > 10)]
    scale = float(np.median(good_torso)) if len(good_torso) else np.nan
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Missing usable torso scale')
    norm = lambda d: d/scale

    left_knee = angle(xy[:,LH],xy[:,LK],xy[:,LA])
    right_knee = angle(xy[:,RH],xy[:,RK],xy[:,RA])
    left_hip = angle(xy[:,LS],xy[:,LH],xy[:,LK])
    right_hip = angle(xy[:,RS],xy[:,RH],xy[:,RK])

    # torso tilt relative to vertical; front and side camera interpretation differs.
    dx = np.abs(shoulder[:,0]-hip[:,0])
    dy = np.abs(shoulder[:,1]-hip[:,1])
    torso_tilt = np.degrees(np.arctan2(dx,dy))

    ankle_sep = np.abs(xy[:,LA,0]-xy[:,RA,0])
    knee_sep = np.abs(xy[:,LK,0]-xy[:,RK,0])
    # More stable than a ratio when the ankles overlap in half-profile view.
    data = {
        'left_knee_angle': left_knee,
        'right_knee_angle': right_knee,
        'knee_angle_mean': mid_scalar(left_knee,right_knee),
        'left_hip_angle': left_hip,
        'right_hip_angle': right_hip,
        'torso_tilt': torso_tilt,
        'hip_ankle_vertical': norm(ankle[:,1]-hip[:,1]),
        'hip_knee_vertical_left': norm(xy[:,LK,1]-xy[:,LH,1]),
        'hip_knee_vertical_right': norm(xy[:,RK,1]-xy[:,RH,1]),
        'knee_ankle_horizontal_left': norm(xy[:,LK,0]-xy[:,LA,0]),
        'knee_ankle_horizontal_right': norm(xy[:,RK,0]-xy[:,RA,0]),
        'knee_separation': norm(knee_sep),
        'ankle_separation': norm(ankle_sep),
        'knee_minus_ankle_separation': norm(knee_sep-ankle_sep),
        'hip_relative_to_ankles_x': norm(hip[:,0]-ankle[:,0]),
    }
    # Squat depth (vertical hip travel) relative to mean of first 15% frames.
    valid_hip = hip[:,1][np.isfinite(hip[:,1])]
    if len(valid_hip):
        n = max(1,int(np.ceil(len(hip)*0.15)))
        ref = hip[:n,1]
        ref = ref[np.isfinite(ref)]
        if len(ref):
            data['hip_y_from_start'] = norm(hip[:,1]-float(ref.mean()))

    result = {}
    for name,values in data.items():
        result.update(stats(values,name))
    return result

def mid_scalar(a,b):
    a,b = np.asarray(a),np.asarray(b)
    return np.where(np.isfinite(a)&np.isfinite(b), (a+b)/2,
                    np.where(np.isfinite(a),a,b))
