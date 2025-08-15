from manim import BLUE_A
from manimlib.imports import *
import numpy as np

class VectorTest(VectorScene):
    # TODO: don't do ql for low quality when running, probs do qh
    def construct(self):
        radii = np.array([i/2 for i in range(1, 11)])
        freqs = np.array([i for i in range(-5, 0)] + [i for i in range(1, 6)])

        self.add_axes(animate=True)
        self.add_plane(animate=True)
        self.wait()

        def rotate_constantly(freq):
            def rotate(mob, dt):
                SPEED_SCALE_FACTOR = 0.1
                mob.rotate(dt * PI / 2 * freq * SPEED_SCALE_FACTOR, about_point=mob.get_start())  # Rotate at π/2 radians per second
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
        for rad, exp in zip(radii, freqs):
            vec = Vector([*self.get_x_y(rad, exp)], colour=BLUE_A)
            vecs.append(vec)

        # Animate placing vectors
        self.play(*[GrowArrow(vec) for vec in vecs], run_time=2)

        # Make vectors rotate + move
        for i, (vec, freq) in enumerate(zip(vecs, freqs)):
            vec.add_updater(rotate_constantly(freq))
            vec.add_updater(update_position(vecs[i-1] if i != 0 else Vector([0, 0])))

        # Add path tracing to last vec
        path = TracedPath(vecs[-1].get_end, dissipating_time=0.5, stroke_width=4, stroke_color=BLUE_A)
        self.add(path)

        self.wait(10)

    def get_x_y(self, rad, exp):
        return rad * np.cos(exp), rad * np.sin(exp)

VectorTest().construct()