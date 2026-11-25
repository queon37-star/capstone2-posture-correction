# Generated from frozen offline source by prepare_board_runtime.py.
from __future__ import annotations
import numpy as np
from . import squat_base as base

LS, RS, LH, RH, LK, RK, LA, RA = base.LS, base.RS, base.LH, base.RH, base.LK, base.RK, base.LA, base.RA

PHASES = ('descent', 'bottom', 'ascent')

EVENTS = ('descent25', 'descent75', 'bottom', 'ascent25', 'ascent75')

def safe_mean(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else np.nan

def safe_stat(a, what):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return np.nan
    if what == 'mean': return float(a.mean())
    if what == 'std': return float(a.std())
    if what == 'max': return float(a.max())
    if what == 'q90': return float(np.quantile(a, 0.9))
    if what == 'range': return float(a.max() - a.min())
    raise ValueError(what)

def stats5(name, values):
    return {f'{name}_{s}': safe_stat(values, s)
            for s in ('mean', 'std', 'max', 'q90', 'range')}

def finite_interp(x):
    """Fill missing hip observations for *event localization only*.

    Raw per-frame pose features remain NaN at missing joints. We do not make up
    observations from interpolated poses.
    """
    x = np.asarray(x, dtype=float)
    good = np.isfinite(x)
    if good.sum() < 3:
        return None
    idx = np.arange(len(x))
    return np.interp(idx, idx[good], x[good])

def get_phase_indices(t, bottom_idx):
    """Approximate descent, deepest moment, ascent by hip trajectory.

    Phase features are only extracted if there are sufficient samples on both
    sides of the bottom. This avoids a false interpretation for incomplete reps.
    """
    phases = {k: np.array([], dtype=int) for k in PHASES}
    if bottom_idx is None or bottom_idx < max(4, int(.12*t)) or bottom_idx > min(t-5, int(.88*t)):
        return phases
    # Descent/ascent: central 60% of each segment; bottom: +/- 5% of repetition.
    d0 = int(np.floor(.2 * bottom_idx))
    d1 = int(np.ceil(.8 * bottom_idx)) + 1
    rest = (t - 1) - bottom_idx
    a0 = bottom_idx + int(np.floor(.2 * rest))
    a1 = bottom_idx + int(np.ceil(.8 * rest)) + 1
    radius = max(2, int(round(t * .05)))
    phases['descent'] = np.arange(d0, min(d1, t))
    phases['bottom'] = np.arange(max(0, bottom_idx - radius), min(t, bottom_idx + radius + 1))
    phases['ascent'] = np.arange(a0, min(a1, t))
    return phases

def near_value(a, idx, radius=2):
    i = int(round(idx))
    a = np.asarray(a, dtype=float)
    return safe_mean(a[max(0, i-radius): min(len(a), i+radius+1)])

def extract_feature_groups(seq):
    """Return exact original 160 features and new interpretable feature groups.

    Keypoint layout: [T, 17, 3] (x/y pixels + detection confidence).
    All 2D geometry is scale normalized by median torso length where needed.
    Knee-in features are masked when ankle separation is too small for reliable
    front-view geometry, reducing spurious side-view ratio features.
    """
    original = base.extract_features(seq)  # Validate shape and scale, exactly as before.
    xy = np.asarray(seq[:, :, :2], dtype=float).copy()
    cf = np.asarray(seq[:, :, 2], dtype=float)
    xy[(cf < 0.30) | ~np.isfinite(cf)] = np.nan
    t = len(seq)

    shoulders = base.mid(xy[:, LS], xy[:, RS])
    hips = base.mid(xy[:, LH], xy[:, RH])
    ankles = base.mid(xy[:, LA], xy[:, RA])
    torso = base.distance(shoulders, hips)
    good_scale = torso[np.isfinite(torso) & (torso > 10)]
    scale = float(np.median(good_scale))
    assert np.isfinite(scale) and scale > 0

    l_knee = base.angle(xy[:, LH], xy[:, LK], xy[:, LA])
    r_knee = base.angle(xy[:, RH], xy[:, RK], xy[:, RA])
    l_hip = base.angle(xy[:, LS], xy[:, LH], xy[:, LK])
    r_hip = base.angle(xy[:, RS], xy[:, RH], xy[:, RK])
    knee_mean = base.mid_scalar(l_knee, r_knee)
    hip_mean = base.mid_scalar(l_hip, r_hip)
    tilt = np.degrees(np.arctan2(np.abs(shoulders[:, 0]-hips[:, 0]),
                                 np.abs(shoulders[:, 1]-hips[:, 1])))

    knee_diff = np.abs(l_knee-r_knee)
    hip_diff = np.abs(l_hip-r_hip)
    # Per-frame left/right difference in change of knee angle.
    # Unlike raw speed, this is insensitive to moving both legs in synchrony.
    angle_step_mismatch = np.full(t, np.nan)
    angle_step_mismatch[1:] = np.abs(np.diff(l_knee) - np.diff(r_knee))
    left_depth = (xy[:, LK, 1] - xy[:, LH, 1]) / scale
    right_depth = (xy[:, RK, 1] - xy[:, RH, 1]) / scale
    leg_vertical_asym = np.abs(left_depth - right_depth)

    asymmetry = {}
    for name, vals in {
        'asym_knee_angle': knee_diff,
        'asym_hip_angle': hip_diff,
        'asym_knee_motion': angle_step_mismatch,
        'asym_leg_vertical': leg_vertical_asym,
    }.items():
        asymmetry.update(stats5(name, vals))

    # Projected frontal alignment. Never force ratios when the ankles overlap
    # in half-profile view. A missing ratio is imputed within the training fold.
    ankle_sep = np.abs(xy[:, LA, 0]-xy[:, RA, 0])
    knee_sep = np.abs(xy[:, LK, 0]-xy[:, RK, 0])
    width_ok = np.isfinite(ankle_sep) & (ankle_sep >= .35 * scale)
    width_ok &= np.isfinite(knee_sep)
    with np.errstate(divide='ignore', invalid='ignore'):
        width_ratio = np.where(width_ok, np.clip(knee_sep / ankle_sep, 0, 4), np.nan)
    # Positive means knees project closer to center than the ankles.
    ankle_mid_x = (xy[:, LA, 0] + xy[:, RA, 0]) / 2.
    l_inward = (np.abs(xy[:, LA, 0] - ankle_mid_x) - np.abs(xy[:, LK, 0] - ankle_mid_x))/scale
    r_inward = (np.abs(xy[:, RA, 0] - ankle_mid_x) - np.abs(xy[:, RK, 0] - ankle_mid_x))/scale
    l_inward[~width_ok] = np.nan
    r_inward[~width_ok] = np.nan
    knee_width_loss = np.where(width_ok, (ankle_sep - knee_sep)/scale, np.nan)
    alignment = {}
    for name, vals in {
        'align_knee_to_ankle_width_ratio': width_ratio,
        'align_knee_width_loss': knee_width_loss,
        'align_left_inward': l_inward,
        'align_right_inward': r_inward,
    }.items():
        alignment.update(stats5(name, vals))
    alignment['align_width_usable_fraction'] = float(width_ok.mean())

    # Larger hip y means deeper squat in normal downward-positive image coords.
    ref_start = safe_mean(hips[:max(1, int(np.ceil(.15 * t))), 1])
    depth = (hips[:, 1] - ref_start)/scale if np.isfinite(ref_start) else np.full(t, np.nan)
    hip_y_interp = finite_interp(hips[:, 1])
    bottom = int(np.argmax(running_median(hip_y_interp))) if hip_y_interp is not None else None
    ph_idx = get_phase_indices(t, bottom)

    signals = {
        'knee_mean': knee_mean,
        'hip_mean': hip_mean,
        'knee_asym': knee_diff,
        'torso_tilt': tilt,
        'squat_depth': depth,
        'knee_width_loss': knee_width_loss,
        'knee_width_ratio': width_ratio,
    }
    phase_features = {}
    for name, vals in signals.items():
        for part in PHASES:
            phase_features[f'phase_{name}_{part}_mean'] = safe_mean(vals[ph_idx[part]])
    for name in ('knee_mean', 'torso_tilt', 'knee_asym'):
        if len(ph_idx['bottom']):
            phase_features[f'phase_{name}_bottom_minus_start'] = (
                safe_mean(signals[name][ph_idx['bottom']]) - safe_mean(signals[name][:max(1, int(.15*t))]))
        else:
            phase_features[f'phase_{name}_bottom_minus_start'] = np.nan

    # Stage-aligned landmarks, not raw time percentage, so descent/ascent
    # rates can differ across participants. Local averages suppress jitter.
    trajectory = {}
    for name, vals in signals.items():
        for phase_name in EVENTS:
            if bottom is None or len(ph_idx['descent']) == 0 or len(ph_idx['ascent']) == 0:
                value = np.nan
            else:
                rest = t - 1 - bottom
                at = {
                    'descent25': .25 * bottom,
                    'descent75': .75 * bottom,
                    'bottom': float(bottom),
                    'ascent25': bottom + .25 * rest,
                    'ascent75': bottom + .75 * rest,
                }[phase_name]
                value = near_value(vals, at, radius=2)
            trajectory[f'traj_{name}_{phase_name}'] = value

    # Quality cues are *not* anatomy. Keep separately identifiable so an
    # ablation can test if confidence/occlusion is causing shortcuts.
    bilateral_legs = np.isfinite(xy[:, [LH, RH, LK, RK, LA, RA], :]).all(axis=(1, 2))
    quality = {
        'quality_bilateral_leg_visibility': float(np.mean(bilateral_legs)),
        'quality_bilateral_knee_angle_visibility': float(np.mean(np.isfinite(l_knee) & np.isfinite(r_knee))),
        'quality_mean_leg_confidence': safe_mean(cf[:, [LH, RH, LK, RK, LA, RA]]),
    }
    return {
        'base': original,
        'asymmetry': asymmetry,
        'alignment': alignment,
        'phase': phase_features,
        'trajectory': trajectory,
        'quality': quality,
    }

def running_median(x, width=7):
    x = np.asarray(x, dtype=float)
    if width != 7 or not np.isfinite(x).all():
        raise ValueError('Expected finite hip interpolation and width=7')
    return np.array([np.median(x[max(0,i-3):min(len(x),i+4)]) for i in range(len(x))])
