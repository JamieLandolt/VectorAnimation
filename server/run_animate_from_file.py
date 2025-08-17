# Pass in - name of file, quality level, sort parameters etc.

import numpy as np
import subprocess

import json


def run_animation(job_id: str = "data", render_params={
    "vectorJMin": -20,  # must be < 0
    "vectorJMax": 20,  # must be > 0
    "sortStyle": "size",  # can be size or speed
    "sortAscending": True,  # true or false
}, manim_params='-p -l'):
    data = np.load(f'points/{job_id}.npy')
    data = np.array([[(row[0]-450)/20, (-row[1]+350)/20] for row in data])
    # Write to file read from
    np.save('data.npy', data)

    # TODO pass in

    json_params = json.dumps(render_params)
    with open('params.txt', 'w') as file:
        file.write(json_params)

    subprocess.Popen(
        f"manim {manim_params} -o  {job_id}.mp4 animate_from_file.py VectorRender", shell=True)
    # Higher quality
    # subprocess.Popen(
    #     f"manim -p --high_quality --frame_rate 60 -o  {job_id}.mp4 animate_from_file.py VectorRender", shell=True)


if __name__ == "__main__":
    run_animation()
