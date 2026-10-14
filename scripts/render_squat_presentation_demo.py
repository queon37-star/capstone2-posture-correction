#!/usr/bin/env python3
"""Render true recorded squat-repetition outputs on their original video.

This is a visualization of PREVIOUSLY RUN inference: it reads actual `rep`
JSONL events. It does not make new posture classifications.

Examples:
  python scripts/render_squat_presentation_demo.py --video-file data/.../PM_029.mp4 \
    --events results/demo/PM_029_events.jsonl --output results/demo/PM_029_full.mp4 --pose-model yolov8m-pose.pt
  python scripts/render_squat_presentation_demo.py --video-file data/.../PM_029.mp4 \
    --events results/demo/PM_029_events.jsonl --output results/demo/PM_029_bad.mp4 \
    --auto-clip bad --pose-model yolov8m-pose.pt
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import cv2


def read_events(jsonl_path: str):
    result = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("type") != "rep":
                continue
            # end_frame includes the trailing classification buffer in our
            # pipeline (detected_end_frame is earlier). Never display at the
            # detected end before the actual post-context is complete.
            frame = item.get("end_frame", item.get("detected_end_frame"))
            if frame is None:
                raise ValueError(f"Event line {lineno} is missing an end frame")
            item["_render_at"] = int(frame)
            result.append(item)
    result.sort(key=lambda x: x["_render_at"])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video-file", required=True, help="Original video for the same events")
    p.add_argument("--events", required=True, help="JSONL from the actual squat stream runner")
    p.add_argument("--output", default="results/demo/squat_demo.mp4")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0, help="0 means until the source ends")
    p.add_argument("--auto-clip", choices=["good", "bad"], help="Auto-trim around first matching real event")
    p.add_argument("--seconds-before", type=float, default=3.5)
    p.add_argument("--seconds-after", type=float, default=3.5)
    p.add_argument("--pose-model", help="Optional YOLOv8m-pose.pt for skeleton visualization only")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--resize-width", type=int, default=1280, help="Max output width; 0 preserves original")
    p.add_argument("--preview", action="store_true", help="Show preview (press q to end; requires GUI)")
    p.add_argument("--list-events", action="store_true", help="Print event timeline and exit")
    p.add_argument("--no-h264", action="store_true", help="Skip automatic H.264 transcoding for PowerPoint")
    args = p.parse_args()

    events = read_events(args.events)
    good = sum(str(e.get("posture", "")).lower() == "good" for e in events)
    bad = sum(str(e.get("posture", "")).lower() == "bad" for e in events)
    print(f"Recorded rep events: {len(events)} | GOOD {good} | BAD {bad}")
    if args.list_events:
        for e in events:
            print(f'frame={e["_render_at"]:5}  count={e.get("count")}  posture={e.get("posture")}')
        return
    if not events:
        raise RuntimeError("No rep events in the JSONL file.")

    video = cv2.VideoCapture(args.video_file)
    if not video.isOpened():
        raise FileNotFoundError(f"Could not open original video: {args.video_file}")
    fps = float(video.get(cv2.CAP_PROP_FPS))
    if not (0.5 < fps < 240.0):
        fps = 30.0
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.auto_clip:
        event = next((e for e in events if str(e.get("posture", "")).lower() == args.auto_clip), None)
        if event is None:
            video.release()
            raise RuntimeError(f'No {args.auto_clip.upper()} events found. Choose another video or model.')
        start = max(0, event["_render_at"] - round(fps * args.seconds_before))
        end = event["_render_at"] + round(fps * args.seconds_after) + 1
        if total > 0:
            end = min(end, total)
        print(f'Auto-clip {args.auto_clip.upper()} around actual event frame {event["_render_at"]}: frames {start}..{end-1}')
    else:
        start = args.start_frame
        end = start + args.max_frames if args.max_frames > 0 else (total if total > 0 else 10**12)
        if total > 0:
            end = min(end, total)
    if start < 0 or end <= start:
        video.release()
        raise ValueError("Invalid video frame interval")

    video.set(cv2.CAP_PROP_POS_FRAMES, start)
    ret, frame = video.read()
    if not ret:
        video.release()
        raise RuntimeError(f"Cannot decode source frame {start}")
    h, w = frame.shape[:2]
    if args.resize_width > 0 and w > args.resize_width:
        scale = args.resize_width / w
        out_w = args.resize_width
        out_h = max(2, int(round(h * scale / 2) * 2))
    else:
        out_w, out_h = w, h
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    if not writer.isOpened():
        video.release()
        raise RuntimeError(f"Cannot open MP4 output: {output}")

    pose = None
    if args.pose_model:
        from ultralytics import YOLO
        pose = YOLO(args.pose_model)
        print('Note: skeleton is rendered by a separate YOLO pass for illustration.')
        print('      Good/Bad labels and counts are from the original JSONL, not this YOLO pass.')

    event_index = 0
    count = 0
    last = "NO RESULT YET"
    for e in events:
        if e["_render_at"] < start:
            count = int(e.get("count", count))
            last = str(e.get("posture", "unknown")).upper()
            event_index += 1

    written = 0
    try:
        for frame_idx in range(start, end):
            if frame_idx != start:
                ret, frame = video.read()
                if not ret:
                    break
            just_published = False
            while event_index < len(events) and events[event_index]["_render_at"] <= frame_idx:
                e = events[event_index]
                count = int(e.get("count", count))
                last = str(e.get("posture", "unknown")).upper()
                event_index += 1
                just_published = True
            # Plot optional pose on ORIGINAL pixels first. This does not
            # influence Good/Bad labels (they come from the recorded log).
            if pose is not None:
                plotted = pose.predict(frame, imgsz=args.imgsz, conf=0.25, verbose=False)[0].plot()
            else:
                plotted = frame.copy()
            if (out_w, out_h) != (w, h):
                plotted = cv2.resize(plotted, (out_w, out_h))

            scale = out_w / 1280.0
            scale = max(0.58, min(1.5, scale))
            margin = max(12, round(19 * scale))
            pad_w = min(out_w - 2 * margin, round(510 * scale))
            pad_h = round(150 * scale)
            panel = plotted.copy()
            cv2.rectangle(panel, (margin, margin), (margin + pad_w, margin + pad_h), (24, 33, 46), -1)
            cv2.addWeighted(panel, 0.82, plotted, 0.18, 0, plotted)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(plotted, "SQUAT | REAL MODEL OUTPUT", (margin+12, margin+round(30*scale)), font,
                        .59*scale, (225, 233, 244), max(1, round(2*scale)), cv2.LINE_AA)
            cv2.putText(plotted, f"REP COUNT  {count}", (margin+12, margin+round(76*scale)), font,
                        1.08*scale, (255, 255, 255), max(1, round(2*scale)), cv2.LINE_AA)
            cl = (105, 223, 130) if last == "GOOD" else (110, 130, 251) if last == "BAD" else (197, 207, 218)
            cv2.putText(plotted, f"LAST REP  {last}", (margin+12, margin+round(120*scale)), font,
                        .77*scale, cl, max(1, round(2*scale)), cv2.LINE_AA)
            # Make it clear results appear only after rep completion and the
            # trailing context frames. Never use GT labels in the video.
            footer = "OFFLINE VIDEO REPLAY | POSTURE DISPLAYED AFTER REP CLASSIFICATION"
            footer_scale = max(0.32, .48*scale)
            cv2.rectangle(plotted, (0, out_h-max(24, round(37*scale))), (out_w, out_h), (24,33,46), -1)
            cv2.putText(plotted, footer, (margin, out_h - max(7, round(12*scale))), font,
                        footer_scale, (218,231,241), 1, cv2.LINE_AA)
            writer.write(plotted)
            written += 1
            if args.preview:
                cv2.imshow("Squat presentation demo (press q to stop)", plotted)
                if cv2.waitKey(1) & 0xff == ord('q'):
                    break
            if just_published:
                print(f"frame={frame_idx}, count={count}, posture={last}")
    finally:
        video.release()
        writer.release()
        if args.preview:
            cv2.destroyAllWindows()

    if written == 0:
        raise RuntimeError("No frames were written")
    # cv2 mp4v is a reliable intermediate output. Re-encode to H.264 for
    # presentation players/PowerPoint when ffmpeg is installed.
    if not args.no_h264:
        encoder = shutil.which("ffmpeg")
        if encoder:
            h264_temp = output.with_name(output.stem + ".h264_temp" + output.suffix)
            try:
                subprocess.run([encoder, "-y", "-hide_banner", "-loglevel", "error", "-i", str(output),
                                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(h264_temp)], check=True)
                os.replace(h264_temp, output)
            except (subprocess.CalledProcessError, OSError) as err:
                h264_temp.unlink(missing_ok=True)
                print(f"WARNING: H.264 conversion failed; original MP4V video retained: {err}")
        else:
            print("WARNING: ffmpeg not found. Output is MP4V; install ffmpeg to use H.264 for PowerPoint.")
    print(f"SAVED: {output} | {written} frames | {fps:.2f} fps | {written/fps:.2f} sec")
    print("Results copied from actual recorded model events; not ground truth or new classification.")


if __name__ == "__main__":
    main()
