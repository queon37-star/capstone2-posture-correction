"""Run the *existing* C922 browser viewer, plus auto exercise detection and posture panel.

No modification to run_squat_pose_client_render.py; no extra camera or Hailo pipeline.
"""
import os
import runpy
from pathlib import Path

from auto_dashboard_shim import install_pipeline_hook, start_status_server


def main():
    render_script = Path(__file__).with_name('run_squat_pose_client_render.py')
    if not render_script.is_file():
        raise FileNotFoundError(f'Missing original browser renderer: {render_script}')
    if os.getenv('SQUAT_COUNT_ONLY', '0') == '1':
        print('WARNING: SQUAT_COUNT_ONLY=1 means Good/Bad is disabled.', flush=True)
    if os.getenv('SQUAT_AUTO_TIMEBASE', 'wall').lower() != 'wall':
        print('WARNING: USB live camera should use SQUAT_AUTO_TIMEBASE=wall.', flush=True)
    install_pipeline_hook()
    start_status_server()
    print('AUTO_DASHBOARD now launching original browser renderer...', flush=True)
    # run_path gives the original script its own __file__, __name__ and CLI args.
    runpy.run_path(str(render_script), run_name='__main__')


if __name__ == '__main__':
    main()
