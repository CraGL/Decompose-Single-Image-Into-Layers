# Decomposing Images into Layers via RGB-space Geometry

This code implements the pipeline described in the paper "Decomposing Images into Layers via RGB-space Geometry" by Jianchao Tan, Jyh-Ming Lien, and Yotam Gingold from TOG 2016.

The pipeline is divided into two steps.

### 1. Convex Hull Simplification

Input:

* Source Image

Output:

* Simplified Hull Vertices (color palettes) for multiple simplification levels (4 to 10 vertices)

Users can then choose what simplification level (number of vertices) is reasonable. They can visualize the output by dragging-and-dropping it onto [our web GUI](http://yig.github.io/image-rgb-in-3D/).


### 2. Layer Extraction

Input: 

* Source Image
* Simplified convex hull vertices (the color palettes)
* Vertex order (users choose which they like)

Output:

* Translucent layers as PNG's and weights (barycentric coordinates).

Users can perform global recoloring in [our web GUI](http://yig.github.io/image-rgb-in-3D/). First load the original image, then drag-and-drop the convex hull `.js` file, and finally drag-and-drop the Barycentric Coordinates weights `.js` file. For a more detailed usage guide to the web GUI, please see the [supplemental materials](http://cs.gmu.edu/~ygingold/singleimage/) of our paper.

We provide two choices for this layer extraction step:

* Our main global optimization-based approach, including RGB and RGBA versions. The RGB optimization assumes that the first layer is opaque, which means its color is the background color. The RGBA version is appropriate when there is no obvious background color, such as a photograph.
* Our As-Sparse-As-Possible (ASAP) approach, which is independent per-pixel and therefore much faster but generates noisier results.

## Dependencies:
* NumPy
* Scipy
* cvxopt (as an interface to the [GLPK](https://www.gnu.org/software/glpk/) linear programming solver)
* [GLPK](https://www.gnu.org/software/glpk/) (`brew install glpk`)
* matplotlib
* OpenCV 3.1
* PIL or Pillow (Python Image Library): `pip install Pillow`
