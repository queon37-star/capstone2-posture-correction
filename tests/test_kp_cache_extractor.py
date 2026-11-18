"""extract_video_kp_cache.py with a scripted pose backend (no YOLO / GPU / real data)."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import extract_video_kp_cache as ex
from auto_detect_hailo_bridge import read_kp_dump

ANCHOR = {'bbox': [100.0, 50.0, 200.0, 250.0], 'anchor_frame': 0}


def write_video(path, n=30, fps=30.0):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (320, 240))
    for i in range(n):
        w.write(np.full((240, 320, 3), i * 5 % 255, np.uint8))
    w.release()


def person_kp(conf=0.9):
    kp = np.zeros((17, 3), np.float32)
    kp[:, 0], kp[:, 1], kp[:, 2] = 150, np.linspace(60, 240, 17), conf
    kp[3, 2] = 0.05          # a low-confidence joint must be stored RAW, not masked
    return kp


class ScriptedBackend:
    """frames 0-4: nothing; 5-9: a box far outside the ROI; 10-24: target;
    25: target box without keypoints; 26-: two people (target + bystander)."""

    def __init__(self):
        self.n = 0

    def predict(self, frame):
        i, self.n = self.n, self.n + 1
        target = np.array([[105, 55, 195, 245]], np.float32)
        if i < 5:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), None
        if i < 10:
            return np.array([[280, 0, 320, 40]], np.float32), np.array([.9], np.float32), person_kp()[None]
        if i == 25:
            return target, np.array([.8], np.float32), None
        if i >= 26:
            boxes = np.vstack([np.array([[280, 0, 320, 40]], np.float32), target])
            return boxes, np.array([.95, .7], np.float32), np.stack([person_kp(.5), person_kp()])
        return target, np.array([.8], np.float32), person_kp()[None]

    def describe(self):
        return {'backend': 'scripted', 'imgsz': 640, 'conf': 0.25}


def test_cache_keeps_every_frame_and_failures(tmp_path):
    video = tmp_path / 'PM_X-Camera17-30fps.mp4'
    write_video(video, n=30)
    meta = ex.extract_video('PM_X', video, ANCHOR, ScriptedBackend(), tmp_path / 'cache')
    data = read_kp_dump(tmp_path / 'cache' / 'PM_X.kp.bin')
    frames = np.load(tmp_path / 'cache' / 'PM_X.frames.npz')
    assert meta['complete'] and meta['frames_decoded'] == 30 == len(data)
    assert np.array_equal(data['frame_id'], np.arange(30))
    assert np.allclose(data['t'], np.arange(30) / 30.0)
    assert list(frames['status'][:5]) == [ex.STATUS_NO_DETECTION] * 5
    assert list(frames['status'][5:10]) == [ex.STATUS_NO_TARGET] * 5
    assert frames['status'][25] == ex.STATUS_NO_KEYPOINTS
    assert (frames['status'][26:] == ex.STATUS_SELECTED).all()          # bystander ignored
    assert np.array_equal(data['pose_valid'], (frames['status'] == 0).astype(np.uint8))
    assert np.isnan(data['kp'][frames['status'] != 0][:, :, :2]).all()   # failures preserved as NaN
    sel = data['kp'][10]
    assert sel[3, 2] == pytest.approx(0.05) and np.isfinite(sel[3, :2]).all()   # raw, unmasked
    assert frames['person_count'][26] == 2
    saved = json.loads((tmp_path / 'cache' / 'PM_X.kp.bin.json').read_text())
    assert saved['video']['sha256'] == ex.sha256_file(video)
    assert saved['video']['fps'] == 30.0 and saved['timebase'] == 'file'
    assert saved['pose_model']['backend'] == 'scripted'
    assert saved['status_counts'] == {'selected': 19, 'no_detection': 5, 'no_target': 5, 'no_keypoints': 1}


def test_refuses_overwrite_and_marks_partial_runs(tmp_path):
    video = tmp_path / 'v.mp4'
    write_video(video, n=12)
    ex.extract_video('PM_Y', video, ANCHOR, ScriptedBackend(), tmp_path / 'c', max_frames=8)
    meta = json.loads((tmp_path / 'c' / 'PM_Y.kp.bin.json').read_text())
    assert meta['complete'] is False and meta['frames_decoded'] == 8
    with pytest.raises(FileExistsError):
        ex.extract_video('PM_Y', video, ANCHOR, ScriptedBackend(), tmp_path / 'c')
    ex.extract_video('PM_Y', video, ANCHOR, ScriptedBackend(), tmp_path / 'c', overwrite=True)
    assert json.loads((tmp_path / 'c' / 'PM_Y.kp.bin.json').read_text())['complete'] is True
    assert not list((tmp_path / 'c').glob('*.partial*'))


def test_compare_with_processed_flags_person_switch():
    import compare_kp_cache_with_processed as cmp
    base = np.repeat(person_kp()[None], 10, axis=0)
    npz = base.copy()
    npz[npz[:, :, 2] < 0.3, :2] = np.nan
    same = cmp.compare(base.copy(), npz)
    assert same['presence_mismatch'] == 0 and same['different_person_frames'] == 0
    assert same['xy_abs_diff_mean'] == 0.0
    other = base.copy()
    other[3:5, :, 0] += 200                    # another person selected in two frames
    other[7] = np.nan                          # cache lost the target in one frame
    res = cmp.compare(other, npz)
    assert res['different_person_frames'] == 2 and res['presence_mismatch'] == 1
