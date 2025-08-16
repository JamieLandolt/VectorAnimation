from manim import *
from manimlib.imports import *
import numpy as np
import random

class VectorTest(VectorScene):
    CONFIG = {
        "axes_config": {
            "stroke_opacity": 0.4,
        },
        "camera_config": {
            "background_color": "#1E1E1E",
        },
    }

    # TODO: don't do ql for low quality when running, probs do qh or qk
    def construct(self):
        self.camera.frame_rate = 30
        self.camera.background_color = RED_A
        self.camera.set_frame_width(20)
        self.camera.set_frame_height(20)

        self.num_vecs = 100
        self.num_vecs += 1 # Avoid making n-1 vecs


        # radii = np.array([0.4 * (self.num_vecs - i) for i in range(self.num_vecs)])
        radii = np.array([np.round(np.sin(i) + 0.1, 2) for i in range(self.num_vecs)])
        freqs = np.array([i*random.randint(1, 10) for i in range(int(-self.num_vecs/2), int(self.num_vecs/2))])
        vector_info = []

        for freq, rad in zip(freqs, radii):
            vector_info.append((freq, self.get_x_y(rad, freq)))

        self.add_axes(animate=True, run_time=0.5)
        plane = NumberPlane(
            background_line_style={  # Faint sub-grid
                "stroke_color": WHITE,
                "stroke_width": 0.5,
                "stroke_opacity": 0.9
            }
        )
        self.add(plane)
        self.wait()

        def rotate_constantly(freq):
            def rotate(mob, dt):
                SPEED_SCALE_FACTOR = 1 / (self.num_vecs)
                mob.rotate(dt * PI / 2 * freq * SPEED_SCALE_FACTOR, about_point=mob.get_start())  # Rotate at π/2 radians per second
            return rotate

        def update_position(prev_vec):
            def move_vector(mob, dt):
                end = prev_vec.get_end()
                components = mob.get_vector()
                mob.put_start_and_end_on(
                    end, [end[0] + components[0], end[1] + components[1], 0])
            return move_vector

        def update_circle_position(vec):
            def move_circle(mob, dt):
                mob.move_to(vec.get_start())
            return move_circle

        # generate vectors (currently test data)
        vecs = []
        # offsets for the vectors when first drawn
        cumulative_vector_offsets = np.zeros(3)
        circle_colours = color_gradient([RED, ORANGE, YELLOW_D], self.num_vecs)

        for i, (freq, (x, y)) in enumerate(vector_info):
            vec = Vector([x, y], stroke_width=3, tip_length=0.2)

            # Shift each vector for the initial drawing
            cumulative_vector_offsets += np.array([vector_info[i - 1][1][0], vector_info[i - 1][1][1], 0]) if i != 0 else cumulative_vector_offsets
            vec.shift(cumulative_vector_offsets)

            # Set vector attributes
            vec.set_color(GREEN_C)
            vec.set_fill(GREEN)
            vec.set_opacity(0.6)

            vecs.append(vec)

            # Circumscribe vectors
            circle = Circle(radius=get_norm(vec.get_end() - vec.get_start()), color=circle_colours[i], stroke_width=0.8, stroke_opacity=0.8)
            circle.move_to(vec.get_start())
            circle.add_updater(update_circle_position(vec))
            self.add(circle)

        # Animate placing vectors
        self.play(*[GrowArrow(vec) for vec in vecs], run_time=0.5)

        get_freqs = lambda x: x[0]
        # Make vectors rotate + move
        for i, (freq, vec) in enumerate(zip(map(get_freqs, vector_info), vecs)):
            vec.add_updater(rotate_constantly(freq))
            vec.add_updater(update_position(vecs[i-1] if i != 0 else Vector([0, 0])))

        # Add path tracing to last vec
        tail = FadingTail(vecs[-1].get_end, fade_time=10, stroke_width=6, stroke_color=ORANGE, min_distance_to_new_point=0.01)
        self.add(tail)

        self.wait(10)

    def get_x_y(self, rad, exp):
        return rad * np.cos(exp), rad * np.sin(exp)



class FadingTail(VGroup):
    def __init__(
            self,
            mobject_or_func,
            fade_time: float = 3.0,
            stroke_width: float = 4,
            stroke_color=YELLOW,
            **kwargs
    ):
        super().__init__(**kwargs)

        if isinstance(mobject_or_func, Mobject):
            self.tracked_func = mobject_or_func.get_center
        else:
            self.tracked_func = mobject_or_func

        self.fade_time = fade_time
        self.base_stroke_width = stroke_width
        self.stroke_color = stroke_color
        self.path_history = []

        self.add_updater(self.update_fading_tail)

    def update_fading_tail(self, mob, dt=None):
        # Get current time (approximate if scene time not available)
        current_time = getattr(self, '_internal_time', 0)
        self._internal_time = current_time + (dt or 1 / 60)

        current_point = self.tracked_func()

        # Add current point
        self.path_history.append((current_point.copy(), current_time))

        # Remove old points
        while (self.path_history and
               current_time - self.path_history[0][1] > self.fade_time):
            self.path_history.pop(0)

        # Clear existing segments
        self.remove(*self.submobjects)

        # Create new segments with fading
        if len(self.path_history) > 1:
            for i in range(len(self.path_history) - 1):
                point1, time1 = self.path_history[i]
                point2, time2 = self.path_history[i + 1]

                # Calculate fade
                age = current_time - time1
                alpha = max(0, 1 - age / self.fade_time)

                if alpha > 0:
                    segment = Line(
                        point1, point2,
                        stroke_color=self.stroke_color,
                        stroke_width=self.base_stroke_width * alpha,
                        stroke_opacity=alpha
                    )
                    self.add(segment)

if __name__ == "__main__":
    VectorTest().construct()