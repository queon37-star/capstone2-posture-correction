"""Hardware-independent bridge from Hailo detections to squat JSONL events.

The adapter and SquatStreamPipeline are loaded from this project's scripts.
The GStreamer callback owns frame_id, detection extraction and video caps.
"""

import json

from hailo_pose_adapter import single_person_coco17
from squat_stream_pipeline import missing_pose


class HailoSquatBridge:
    def __init__(self, pipeline, writer):
        self.pipeline = pipeline
        self.writer = writer
        self.processed_frames = 0
        self.valid_person_frames = 0
        self.missing_person_frames = 0

    def process(self, *, frame_id, detections, width, height, landmarks_type):
        """Process every frame, even on missing caps or ambiguous person detections.

        The supplied width/height must refer to the callback pad, not camera
        source resolution or Hailo model tensor resolution.
        """
        kp = None
        if width is not None and height is not None:
            kp = single_person_coco17(
                detections, width, height, landmarks_type=landmarks_type,
            )
        if kp is None:
            kp = missing_pose()
            self.missing_person_frames += 1
        else:
            self.valid_person_frames += 1

        events = self.pipeline.push(kp, frame_id=frame_id)
        self.processed_frames += 1
        self._publish(events)
        return events

    def finish(self):
        """Flush pending post-context events when capture ends."""
        events = self.pipeline.flush()
        self._publish(events)
        return events

    def _publish(self, events):
        for event in events:
            line = json.dumps(event, ensure_ascii=False)
            self.writer.write(line + '\n')
            self.writer.flush()
            print('REP', line, flush=True)
