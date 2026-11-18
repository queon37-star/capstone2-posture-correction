"""inventory_rehab24.py on a tiny fake REHAB24-6 layout (no real data needed)."""
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_inventory_counts(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    rows = []
    for ex, person, label, vid in [(6, 1, 1, 'PM_A'), (6, 1, 0, 'PM_A'), (6, 2, 1, 'PM_B'),
                                   (3, 1, 0, 'PM_C'), (3, 2, 1, 'PM_C')]:
        rows.append({'video_id': vid, 'repetition_number': len(rows) + 1, 'exercise_id': ex,
                     'person_id': person, 'first_frame': 10 * len(rows),
                     'last_frame': 10 * len(rows) + 59, 'correctness': label,
                     'cam17_orientation': 'front'})
    with (raw / 'Segmentation.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=';')
        w.writeheader()
        w.writerows(rows)
    (raw / 'PM_A-Camera17-30fps.mp4').write_bytes(b'')
    (raw / 'PM_C-Camera18-30fps.mp4').write_bytes(b'')
    out = tmp_path / 'out'
    subprocess.run([sys.executable, str(ROOT / 'scripts' / 'inventory_rehab24.py'), '--raw', str(raw),
                    '--processed-index', str(tmp_path / 'none.csv'), '--out', str(out)],
                   check=True, capture_output=True)
    per_ex = {r['exercise_id']: r for r in csv.DictReader((out / 'per_exercise.csv').open())}
    assert per_ex['6']['reps'] == '3' and per_ex['6']['correct(1)'] == '2'
    assert per_ex['3']['name'] == 'pushup_table' and per_ex['3']['persons'] == '2'
    videos = {r['video_id']: r for r in csv.DictReader((out / 'videos.csv').open())}
    assert videos['PM_B']['cameras_on_disk'] == 'MISSING' and videos['PM_C']['cameras_on_disk'] == '18'
