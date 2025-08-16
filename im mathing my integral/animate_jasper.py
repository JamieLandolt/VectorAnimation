# from asjdflkajsflkjasf import VectorData
from manim import BLUE_A
from manimlib.imports import *
import numpy as np

from typing import List
# import matplotlib.pyplot as plt


def get_average(vals: List[int]) -> int:
    if len(vals) < 3:
        return "error"
    interval_width: int = 1
    # trapezium method
    middle = vals[1:-1]
    # divide by b-a = len(int)-1
    integral = 0.5*interval_width*(vals[0] + 2*sum(middle) + vals[-1])
    return integral/(len(vals)-1)


def get_c_j(x_y_vals, j: int) -> float:
    # J is the c value we want to find

    # first get the average of x and y independently

    def fx(t): return x_y_vals[t][0]*np.cos(-2 *
                                            np.pi*j*t) - x_y_vals[t][1]*np.sin(-2*np.pi*j*t)
    def fy(t): return x_y_vals[t][1]*np.cos(-2 *
                                            np.pi*j*t) + x_y_vals[t][0]*np.sin(-2*np.pi*j*t)

    fx_vals = np.array([fx(t) for t in range(len(x_y_vals))])
    fy_vals = np.array([fy(t) for t in range(len(x_y_vals))])

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

    def sort_by_mag(c_data, byLargest):
        def mag(tup): return tup[1][0]**2 + tup[1][1]**2
        if (byLargest):
            sortedMags = sorted(c_data, key=mag, reverse=True)
        else:
            sortedMags = sorted(c_data, key=mag)

        return sortedMags

    def get_c_vals(self):
        return self.c_vals


# Jasper testing
data_heart = np.array([
    (16 * np.sin(t)**3,
     13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t))
    for t in np.linspace(0, 2*np.pi, 100)
])

vectorData = VectorData(data_heart)
# vector_info = vectorData.get_c_vals()
# print(vector_info)


class VectorTest(VectorScene):
    # TODO: don't do ql for low quality when running, probs do qh or qk
    def construct(self):
        self.camera.set
        num_vecs = 20
        # radii = np.array([i/(2*num_vecs) for i in range(num_vecs)])
        # freqs = np.array([i for i in range(int(-num_vecs/2), int(num_vecs/2))])
        # vector_info = []

        # for freq, rad in zip(freqs, radii):
        #     vector_info.append((freq, self.get_x_y(rad, freq)))

        vector_info = vectorData.get_c_vals()

        self.camera.set
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


if __name__ == "__main__":
    VectorTest().construct()
