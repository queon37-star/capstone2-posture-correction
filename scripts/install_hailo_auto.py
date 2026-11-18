from pathlib import Path
source = Path('scripts/run_squat_hailo_pose.py')
target = Path('scripts/run_squat_hailo_auto.py')
text = source.read_text(encoding='utf-8')
imp = 'from squat_stream_pipeline import SquatStreamPipeline'
anchor = '    # The original Hailo sample sets HAILO_ENV_FILE'
assert text.count(imp) == 1, 'Cannot find unique SquatStreamPipeline import'
assert text.count(anchor) == 1, 'Cannot find unique anchor, do not modify runner'
text = text.replace(imp, imp + '\nfrom auto_detect_hailo_bridge import AutoDetectPipeline', 1)
text = text.replace(anchor, '    pipeline = AutoDetectPipeline(pipeline, fps)\n\n' + anchor, 1)
if target.exists():
    raise SystemExit(f'Refusing to overwrite {target}. Remove it intentionally before reinstalling.')
target.write_text(text, encoding='utf-8')
print('CREATED', target)
