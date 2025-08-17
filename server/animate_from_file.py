from manim import *
from manimlib.imports import *
import numpy as np
from typing import List
import json

# TODO: Zoom, fix circles, Button on website for selecting sort, Button for selecting quality (numVectors)
# TODO: Adjust length of video based on number/time of timestamps, Add high quality option on website

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


class VectorRender(ZoomedScene):
    CONFIG = {
        "axes_config": {
            "stroke_opacity": 0.4,
        },
        "camera_config": {
            "background_color": "#1E1E1E",
        },
    }

    def __init__(self, **kwargs):
        ZoomedScene.__init__(
            self,
            zoom_factor=0.1,  # Size of zoom window relative to screen
            zoomed_display_height=10,  # Height of zoom display
            zoomed_display_width=20,  # Width of zoom display
            image_frame_stroke_width=20,
            zoomed_camera_config={
                "default_frame_stroke_width": 3,
                "background_color": BLACK,
            },
            **kwargs
        )

    def construct(self):
        self.high_quality = True
        self.enable_circles = False
        self.animation_length = 5 if not self.high_quality else 60

        self.camera.frame_rate = 30 if not self.high_quality else 60
        self.camera.background_color = RED_A

        vectorData, vector_info = self.get_vector_info()
        self.num_vecs = vectorData.get_num_vectors()

        self.aspect_ratio = 9/16
        self.set_fov(vector_info, self.aspect_ratio)
        self.draw_plane_axes()

        vecs = self.create_vectors(vector_info)
        frame = self.zoom_in_on_vectors(vecs[-1])

        self.animate_vectors(vector_info, vecs)

        self.wait(self.animation_length / 3)
        frame.scale(0.5)
        self.wait(self.animation_length / 3)
        frame.scale(0.5)
        self.wait(self.animation_length / 3)

    def set_fov(self, vector_info, aspect_ratio):
        # Determine how big frame is
        self.max_vector_height = 0

        # Calc max possible vector distance from origin
        for _, (i, j) in vector_info:
            self.max_vector_height += np.sqrt(i ** 2 + j ** 2)
        self.max_vector_height *= 4/5 # This is a random number lol seems to work

        self.camera.set_frame_width(2*self.max_vector_height)
        self.camera.set_frame_height(2*self.max_vector_height*self.aspect_ratio)


    def zoom_in_on_vectors(self, last_vec):
        def track_last_vector(vec):
            def move_to_last_vector(mob, dt):
                mob.move_to(vec.get_end())
            return move_to_last_vector

        # What should be showed in the zoomed frame
        frame = self.zoomed_camera.frame
        frame.move_to(last_vec.get_end())
        frame.set_color(GREY)
        frame.add_updater(track_last_vector(last_vec))

        # The zoomed frame (int the top left)
        zoomed_display = self.zoomed_display
        zoomed_display.set_height(self.max_vector_height * 0.3)
        zoomed_display.set_width(self.max_vector_height * 0.6)
        zoomed_display_frame = zoomed_display.display_frame
        zoomed_display_frame.set_color(WHITE)

        # zoomed_display.shift(LEFT * self.max_vector_height / 2.2)
        # zoomed_display.shift(DOWN * self.max_vector_height / 3.8) # UP 11.5
        # zoomed_display.to_corner(DL)
        zoomed_display.move_to([-self.max_vector_height + zoomed_display.get_width() / 2 + 0.3,
                                -self.max_vector_height * self.aspect_ratio + zoomed_display.get_height() / 2 + 0.3, 0])

        # Animate making the frame
        self.add(frame)
        self.activate_zooming()

        self.play(self.get_zoomed_display_pop_out_animation())
        return frame

    def get_vector_info(self):
        data = np.load('data.npy')

        with open("params.txt", "r") as file:
            json_text = file.read()

        params = json.loads(json_text)

        vectorData = VectorData(data)
        vectorData.update_internal_c_vals(
            params['vectorJMin'] if not self.high_quality else -100,
            params['vectorJMax'] if not self.high_quality else 100)

        if params['sortStyle'] == 'size':
            vectorData.sort_by_mag(byLargest=params['sortAscending'])
        else:
            vectorData.sort_by_speed(asc=params['sortAscending'])

        # Keep tail or not
        self.keepTail = params['keepTail']

        vector_info = vectorData.get_c_vals()
        return vectorData, vector_info

    def create_vectors(self, vector_info):
        # Create vectors starting from origin pointing to their c_j at t=0
        circle_colours = color_gradient([RED, ORANGE, YELLOW_D], self.num_vecs)

        cumulative_vector_offsets = np.zeros(3)
        biggest_magnitude = max(
            vector_info, key=lambda x: x[1][0] ** 2 + x[1][1] ** 2)
        biggest_magnitude = np.sqrt(
            biggest_magnitude[1][0] ** 2 + biggest_magnitude[1][1] ** 2)

        vecs = []
        for i, (freq, (x, y)) in enumerate(vector_info):
            vec_magnitude = np.sqrt((x ** 2) + (y ** 2))
            thickness_scale = lambda x: -(-(np.log(x + 0.5) / np.log(1000)) + 0.4) ** 4 + 0.5
            vec = Vector([x, y, 0], stroke_width=12 * thickness_scale(vec_magnitude / biggest_magnitude),
                         tip_length=vec_magnitude / biggest_magnitude)

            # Shift each vector for the initial drawing
            cumulative_vector_offsets += np.array(
                [vector_info[i - 1][1][0], vector_info[i - 1][1][1], 0]) if i != 0 else cumulative_vector_offsets
            vec.shift(cumulative_vector_offsets)

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

        return vecs

    def animate_vectors(self, vector_info, vecs):
        def get_freqs(x):
            return x[0]

        def rotate_constantly(freq):
            def rotate(mob, dt):
                SPEED_SCALE_FACTOR = 2 / (self.num_vecs)
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
        if self.keepTail:
            tail = TracedPath(
                traced_point_func = vecs[-1].get_end,
                stroke_width = 6,
                stroke_color=ORANGE
            )
        else:
            tail = FadingTail(vecs[-1].get_end, fade_time=15, stroke_width=6,
                          stroke_color=ORANGE, min_distance_to_new_point=0.01)
        self.add(tail)

    def get_x_y(self, rad, exp):
        return rad * np.cos(exp), rad * np.sin(exp)

    def draw_plane_axes(self):
        axes = Axes(
            x_min=-self.max_vector_height,
            x_max=self.max_vector_height,
            x_step=1,
            y_min=-self.max_vector_height*self.aspect_ratio,
            y_max=self.max_vector_height*self.aspect_ratio,
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
            y_min=-self.max_vector_height*self.aspect_ratio,
            y_max=self.max_vector_height*self.aspect_ratio,
            y_step=1,
        )
        self.play(ShowCreation(plane))


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
