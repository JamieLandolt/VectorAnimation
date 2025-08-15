from manimlib.imports import *

class Vectors(VectorScene):
    CONFIG = {
        "show_basis_vectors": True,
        "foreground_plane_kwargs": {
            "x_max": 10,
            "x_min": -10,
            "y_max": 10,
            "y_min": -10,
        }
    }
    def construct(self):
        self.add_axes(animate=True)
        self.add_plane(animate=True)
        self.wait()

        myVec = Vector([3, 2])
        # self.add(myVec) # places it
        self.add_vector(myVec) # Grows it

        self.wait()


class RotatingVector(VectorScene):
    def construct(self):
        plane = NumberPlane()
        self.add(plane)

        vector = Vector([3, 2], color="blue")
        self.add(vector)

        # Rotate by 90 degrees (π/2 radians)
        self.play(Rotate(vector, PI / 2))
        self.wait()

        # Rotate by another 45 degrees
        self.play(Rotate(vector, PI / 4))
        self.wait()


class ConstantRotation(VectorScene):
    def construct(self):
        plane = NumberPlane()
        self.add(plane)

        vector = Vector([3, 2])
        self.add(vector)
        vector2 = Vector([3, 2])
        vector2.shift([3, 4, 0])
        self.add(vector)
        # Get end coords of vec
        tip_coords = vector.get_end()

        # Add updater for constant rotation
        def rotate_constantly(mob, dt):
            mob.rotate(dt * PI / 2, about_point=ORIGIN)  # Rotate at π/2 radians per second

        # Add updater to shift the base
        def move_to_tip(mob, dt):
            mob.move_to()

        vector.add_updater(rotate_constantly)
        vector2.add_updater(rotate_constantly)

        self.wait(8)  # Let it rotate for 8 seconds (4 full rotations)