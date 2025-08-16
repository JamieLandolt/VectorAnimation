from manim import *
from manimlib.imports import *
import numpy as np
from typing import List
import random


def get_average(vals: List[float]) -> float:
    if len(vals) < 3:
        return "error"
    interval_width: float = 1
    # trapezium method
    middle = vals[1:-1]
    # divide by b-a = len(int)-1
    integral = 0.5*interval_width*(vals[0] + 2*sum(middle) + vals[-1])
    return integral/(len(vals)-1)


def get_c_j(x_y_vals, j: int) -> float:
    # J is the c value we want to find

    # first get the average of x and y independently
    N = len(x_y_vals)

    def fx(t):
        angle = -2 * np.pi * j * t / N
        return x_y_vals[t][0] * np.cos(angle) - x_y_vals[t][1] * np.sin(angle)

    def fy(t):
        angle = -2 * np.pi * j * t / N
        return x_y_vals[t][1] * np.cos(angle) + x_y_vals[t][0] * np.sin(angle)

    fx_vals = [fx(t) for t in range(N)]
    fy_vals = [fy(t) for t in range(N)]

    return (get_average(fx_vals), get_average(fy_vals))


class VectorData:

    def __init__(self, data: List[List[float]]):
        """Data is of the form of a list of x and y tuples/lists"""
        self.data = data
        self.update_internal_c_vals()

    def update_internal_c_vals(self, min_j=-20, max_j=20):
        self.c_vals = [(j, get_c_j(self.data, j))
                       for j in range(min_j, max_j+1)]

    # def plot_data(self):
    #     plt.scatter(x=self.data[:, 0], y=self.data[:, 1])
    #     plt.axis('equal')
    #     plt.show()

    def sort_by_speed(self, asc=True):
        """Assumes continuity between max and min j
        sorts by speed ascending"""

        # i = 1
        output = []

        zero_index = 0
        # find the index of 0
        while (self.c_vals[zero_index][0] != 0):
            zero_index += 1

        output.append(self.c_vals[zero_index])

        i = 1

        while zero_index-i >= 0 and zero_index+i < len(self.c_vals):
            output.extend([self.c_vals[zero_index-i],
                          self.c_vals[zero_index+i]])
            i += 1

        while zero_index-i >= 0:
            output.append(self.c_vals[zero_index-i])
            i += 1

        while zero_index+i < len(self.c_vals):
            output.append(self.c_vals[zero_index+i])
            i += 1

        if asc:
            self.c_vals = output
        else:
            self.c_vals = output[::-1]

    def sort_by_mag(self, byLargest=True):
        def mag(tup): return tup[1][0]**2 + tup[1][1]**2
        if (byLargest):
            sortedMags = sorted(self.c_vals, key=mag, reverse=True)
        else:
            sortedMags = sorted(self.c_vals, key=mag)

        return sortedMags

    def get_c_vals(self):
        return self.c_vals

    def get_worst_case_length(self):
        return sum([np.sqrt(row[1][0]**2 + row[1][1]**2) for row in self.c_vals])

    def get_num_vectors(self):
        return len(self.c_vals)


class VectorRender(VectorScene):
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
        self.low_quality = False
        self.enable_circles = False

        self.camera.frame_rate = 30
        self.camera.background_color = RED_A

        # # Jasper testing (Testy Heart Data)
        # data = np.array([
        #     (16 * np.sin(t) ** 3,
        #      13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
        #     for t in np.linspace(0, 2 * np.pi, 100)
        # ])

        data = np.load('data.npy')

        vectorData = VectorData(data)

        vectorData.update_internal_c_vals(-10, 10)
        vectorData.sort_by_speed()
        vector_info = vectorData.get_c_vals()
        self.num_vecs = vectorData.get_num_vectors()

        # Determine how big frame is
        self.max_vector_height = 0
        for _, (i, j) in vector_info:
            self.max_vector_height += np.sqrt(i ** 2 + j ** 2)
        self.max_vector_height *= 4/5

        self.camera.set_frame_width(2*self.max_vector_height)
        self.camera.set_frame_height(2*self.max_vector_height)

        axes = Axes(
            x_min=-self.max_vector_height,
            x_max=self.max_vector_height,
            x_step=1,
            y_min=-self.max_vector_height,
            y_max=self.max_vector_height,
            y_step=1,

        )
        self.play(ShowCreation(axes))
        plane = NumberPlane(
            background_line_style={  # Faint sub-grid
                "stroke_color": WHITE,
                "stroke_width": 0.5,
                "stroke_opacity": 0.9
            },
            x_min=-self.max_vector_height,
            x_max=self.max_vector_height,
            x_step=1,
            y_min=-self.max_vector_height,
            y_max=self.max_vector_height,
            y_step=1,
        )
        self.play(ShowCreation(plane))
        self.wait()

        def rotate_constantly(freq):
            def rotate(mob, dt):
                SPEED_SCALE_FACTOR = 4 / (self.num_vecs)
                mob.rotate(dt * TAU * freq * SPEED_SCALE_FACTOR,
                           about_point=mob.get_start())

            return rotate

        def update_position(prev_vec):
            def move_vector(mob, dt):
                end = prev_vec.get_end()
                components = mob.get_vector()
                mob.put_start_and_end_on(
                    end, end + components
                )

            return move_vector

        def update_circle_position(vec):
            def move_circle(mob, dt):
                mob.move_to(vec.get_start())
            return move_circle

        # Create vectors starting from origin pointing to their c_j at t=0
        vecs = []
        circle_colours = color_gradient([RED, ORANGE, YELLOW_D], self.num_vecs)
        biggest_magnitude = max(
            vector_info, key=lambda x: x[1][0] ** 2 + x[1][1] ** 2)
        biggest_magnitude = np.sqrt(
            biggest_magnitude[1][0] ** 2 + biggest_magnitude[1][1] ** 2)
        for freq, (x, y) in vector_info:
            vec_magnitude = np.sqrt((x ** 2) + (y ** 2))
            vec = Vector([x, y, 0], stroke_width=6 * vec_magnitude /
                         biggest_magnitude, tip_length=vec_magnitude / biggest_magnitude * 1)

            # Set vector attributes
            vec.set_color(GREEN_C)
            vec.set_fill(GREEN)
            vec.set_opacity(0.6)

            vecs.append(vec)

            if self.enable_circles:
                # Circumscribe vectors
                # circle = Circle(radius=get_norm(vec.get_end() - vec.get_start()), color=circle_colours[i], stroke_width=0.8, stroke_opacity=0.8)
                # circle.move_to(vec.get_start())
                # circle.add_updater(update_circle_position(vec))
                # self.add(circle)
                pass

        # Animate placing vectors
        self.play(*[GrowArrow(vec) for vec in vecs], run_time=0.5)

        def get_freqs(x): return x[0]
        # Make vectors rotate + move
        for i, (freq, vec) in enumerate(zip(map(get_freqs, vector_info), vecs)):
            vec.add_updater(rotate_constantly(freq))
            if i == 0:
                # First vector starts at origin always
                def stay_at_origin(mob, dt):
                    mob.put_start_and_end_on([0, 0, 0], mob.get_end())

                vec.add_updater(stay_at_origin)
            else:
                # Each vector starts at the end of the previous vector
                vec.add_updater(update_position(vecs[i - 1]))

        # Add path tracing to last vec
        tail = FadingTail(vecs[-1].get_end, fade_time=15, stroke_width=6,
                          stroke_color=ORANGE, min_distance_to_new_point=0.01)
        self.add(tail)

        self.wait(40)

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
    VectorRender().construct()
