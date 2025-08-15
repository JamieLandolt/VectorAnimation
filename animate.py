from manimlib.imports import *
import numpy as np

class VectorTest(VectorScene):
    # TODO: don't do ql for low quality when running, probs do qh
    def construct(self):
        r = 1
        data = np.array([(r * np.cos(theta), r * np.sin(theta)) for theta in np.linspace(0, 2 * np.pi, 20)])
        print(data)

        self.add_axes(animate=True)
        self.add_plane(animate=True)
        self.wait()

        myVec = Vector([2, 3])
        self.add_vector(myVec) # add_vector grows the vector
        self.wait()

VectorTest().construct()