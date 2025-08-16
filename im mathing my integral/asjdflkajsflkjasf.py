from typing import List
# import matplotlib.pyplot as plt
import numpy as np


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


if __name__ == "__main__":
    data_heart = np.array([
        (16 * np.sin(t)**3,
         13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t))
        for t in np.linspace(0, 2*np.pi, 100)
    ])

    vectorData = VectorData(data_heart)
    vector_info = vectorData.get_c_vals()
    print(vector_info)
