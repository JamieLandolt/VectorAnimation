# Pass in - name of file, quality level, sort parameters etc.

import numpy as np
import subprocess


def run_animation(job_id: str = "data", manim_params='-plq'):
    # renormalise the canvas size
    data = np.load(f'{job_id}.npy')
    # x -> x - 450
    # y -> -y + 350
    data = np.array([[(row[0]-450)/20, (-row[1]+350)/20] for row in data])
    # Write to file read from
    np.save('data.npy', data)
    subprocess.Popen(
        f"manim {manim_params} animate_from_file.py VectorRender", shell=True)
