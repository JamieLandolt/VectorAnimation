from asjdflkajsflkjasf import VectorData
from manim import BLUE_A
from manimlib.imports import *
import numpy as np


class VectorTest(VectorScene):
    # TODO: don't do ql for low quality when running, probs do qh or qk
    def construct(self):
        self.camera.frame_rate = 30
        num_vecs = 20
        # radii = np.array([i/(2*num_vecs) for i in range(num_vecs)])
        # freqs = np.array([i for i in range(int(-num_vecs/2), int(num_vecs/2))])
        # vector_info = []

        # for freq, rad in zip(freqs, radii):
        #     vector_info.append((freq, self.get_x_y(rad, freq)))

        global vector_info

        self.add_axes(animate=True, run_time=0.5)
        self.add_plane(animate=True, run_time=0.5)
        self.wait()

        def rotate_constantly(freq):
            def rotate(mob, dt):
                SPEED_SCALE_FACTOR = 1 / (num_vecs)
                # Rotate at π/2 radians per second
                mob.rotate(dt * PI / 2 * freq * SPEED_SCALE_FACTOR,
                           about_point=mob.get_start())
            return rotate

        def update_position(prev_vec):
            def move_vector(mob, dt):
                end = prev_vec.get_end()
                components = mob.get_vector()
                mob.put_start_and_end_on(
                    end, [end[0] + components[0], end[1] + components[1], 0])
            return move_vector

        # generate vectors (currently test data)
        vecs = []
        for i, (freq, (x, y)) in enumerate(reversed(vector_info)):
            # This puts vectors on top of each other unless it is the first vector
            vector_pos = [x + vector_info[i - 1][1][0], y +
                          vector_info[i - 1][1][1]] if i != 0 else [x, y]
            vec = Vector(vector_pos, colour=BLUE_A)
            vecs.append(vec)

        # Animate placing vectors
        self.play(*[GrowArrow(vec) for vec in vecs], run_time=0.5)

        def get_freqs(x): return x[0]
        # Make vectors rotate + move
        for i, (freq, vec) in enumerate(zip(map(get_freqs, vector_info), vecs)):
            vec.add_updater(rotate_constantly(freq))
            vec.add_updater(update_position(
                vecs[i-1] if i != 0 else Vector([0, 0])))

        # Add path tracing to last vec
        path = TracedPath(vecs[-1].get_end, dissipating_time=0.5, stroke_width=4,
                          stroke_color=BLUE_A, min_distance_to_new_point=0.01)
        self.add(path)

        self.wait(60)

    def get_x_y(self, rad, exp):
        return rad * np.cos(exp), rad * np.sin(exp)


# Jasper testing

data_heart = np.array([
    (16 * np.sin(t)**3,
     13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t))
    for t in np.linspace(0, 2*np.pi, 100)
])

vectorData = VectorData(data_heart)
vector_info = vectorData.get_c_vals()

if __name__ == "__main__":
    VectorTest().construct()
