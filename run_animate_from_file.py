# Pass in - name of file, quality level, sort parameters etc.

import numpy as np
import subprocess


def run_animation(job_id: str = "data", manim_params='-plq'):
    data = np.load(f'{job_id}.npy')
    # Write to file read from
    np.save('data.npy', data)
    subprocess.Popen(
        f"manim {manim_params} animate_from_file.py VectorRender", shell=True)
