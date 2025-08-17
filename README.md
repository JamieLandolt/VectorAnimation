Fourier Series Vector Animations by the Fourier Four (Jamie, John, Jasper & Will)

Tech Stack:
Python
- Flask & Flask_Restx (Web) (John)
- Manim & ManimLib (Animations) (Jamie)
- Numpy (Math) (Jasper)

This project allows anyone to draw an image which will then be recreated by stacked rotating vectors.
The catch is that all vectors are rotating at predetermined constant rates and the only thing we control is the size of each vector.

There are three core components in the functionality pipeline:
- Website
  - The user draws an image
  - The (x, y) coordinates are sampled at each timestamp to track where the pen was at each timestamp.
- Math
  - Given an ordered list of (x, y) coordinates, you generate n vectors with frequencies ranging between [-(n-1), (n-1)]
  - Then you perform a fourier transform to determine the magnitudes of the n vectors.
- Animation
  - Given a list of frequencies and i, j components [(freq, (i, j)] generate each vector with size (i, j)
  - Generated base to tip moving the base of the k+1th vector to the tip of the kth vector
  - At each frame, move the base of the k+1th vector to the tip of the kth vector and rotate the kth vector based on the kth frequency and the time passed since the last frame
And Thats It!

Whiteboard explanation (More in depth on the maths): (Will)
![IMG_7118](https://github.com/user-attachments/assets/0a3171e3-1219-42ba-aba3-402e3eb5832c)

