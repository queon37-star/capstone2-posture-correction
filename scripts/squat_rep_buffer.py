"""Causal, bounded-memory frame padding for SquatCounter repetitions.

No GT labels, segmentation CSV, look-ahead beyond the chosen post_frames,
or model outputs are used when selecting frames.
"""
from collections import deque
from dataclasses import dataclass
import numpy as np


@dataclass
class BufferedRep:
    keypoints: np.ndarray
    start_frame: int
    end_frame: int
    detected_start_frame: int
    detected_end_frame: int
    peak_depth: float
    min_knee_angle: float
    post_context_complete: bool


class RepBoundaryBuffer:
    def __init__(self, fps=30.0, pre_frames=16, post_frames=22, max_active_seconds=8.0):
        self.pre_frames = int(pre_frames)
        self.post_frames = int(post_frames)
        if self.pre_frames < 0 or self.post_frames < 0:
            raise ValueError('pre_frames and post_frames must be nonnegative')
        self.history = deque(maxlen=int(round(max_active_seconds * fps)) + self.pre_frames + self.post_frames + 48)
        self.pending = deque()
        self.previous_emitted_end = -1
        self.last_frame = -1

    def push(self, frame_id, keypoints, detected_rep=None):
        frame_id = int(frame_id)
        if frame_id <= self.last_frame:
            raise ValueError('frame IDs must increase')
        self.last_frame = frame_id
        kp = np.asarray(keypoints, np.float32)
        if kp.shape != (17, 3):
            raise ValueError('Expected (17,3) keypoints')
        self.history.append((frame_id, kp.copy()))
        if detected_rep is not None:
            if detected_rep.end_frame > frame_id or detected_rep.start_frame > detected_rep.end_frame:
                raise ValueError('Invalid event frame bounds')
            self.pending.append(detected_rep)
        completed = []
        while self.pending and frame_id >= self.pending[0].end_frame + self.post_frames:
            event = self.pending.popleft()
            built = self._finalize(event, frame_id, post_context_complete=True)
            if built is not None:
                completed.append(built)
        return completed

    def flush(self):
        """Call at stream EOF; may have incomplete post-context."""
        output = []
        while self.pending:
            event = self.pending.popleft()
            built = self._finalize(event, self.last_frame, post_context_complete=False)
            if built is not None:
                output.append(built)
        return output

    def _finalize(self, event, available_through, post_context_complete):
        start = max(0, int(event.start_frame) - self.pre_frames, self.previous_emitted_end + 1)
        end = min(int(event.end_frame) + self.post_frames, available_through)
        if end < start:
            return None
        selected = [(fid, kp) for fid, kp in self.history if start <= fid <= end]
        if not selected:
            return None
        # A stream is expected to provide every frame. Raise rather than quietly
        # computing misleading statistical features from a gapped sequence.
        ids = [x[0] for x in selected]
        if ids != list(range(start, end + 1)):
            raise RuntimeError(f'History does not cover requested [{start}, {end}] (available [{ids[0]}, {ids[-1]}])')
        self.previous_emitted_end = end
        return BufferedRep(
            keypoints=np.stack([x[1] for x in selected]),
            start_frame=start,
            end_frame=end,
            detected_start_frame=int(event.start_frame),
            detected_end_frame=int(event.end_frame),
            peak_depth=float(event.peak_depth),
            min_knee_angle=float(event.min_knee_angle),
            post_context_complete=post_context_complete,
        )
