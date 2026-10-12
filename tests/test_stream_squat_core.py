"""Small unit checks for online state transitions; not a real video benchmark."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from stream_squat_core import SquatCounter


def pose(y_hip=400, angle=170):
    # Simple synthetic side pose, chosen to produce a plausible knee angle.
    # The explicit knee angle is encoded geometrically from hip to knee to ankle.
    a = np.zeros((17,3),np.float32)
    a[:, :2] = np.nan
    a[:, 2] = 0
    for i,(x,y) in {5:(475,y_hip-160),6:(525,y_hip-160),
                    11:(475,y_hip),12:(525,y_hip),
                    13:(475,y_hip+110),14:(525,y_hip+110),
                    15:(477,y_hip+220),16:(527,y_hip+220)}.items():
        a[i] = [x,y,0.98]
    return a


def test_invalid_pose():
    a = np.zeros((17, 3),np.float32);a[:,:2]=np.nan
    s=SquatCounter()
    for i in range(20):
        assert s.push(a,frame_id=i) is None
    assert s.count==0


def test_calibrate():
    s=SquatCounter()
    for i in range(20):s.push(pose(),frame_id=i)
    assert s.state=='READY',s.state


if __name__ == '__main__':
    test_invalid_pose();test_calibrate()
    print('Basic online counter tests: PASS (synthetic only)')
