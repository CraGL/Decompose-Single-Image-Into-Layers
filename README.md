Decomposing Images into Layers via RGB-space Geometry

This code implements the pipeline described in the paper "Decomposing Images into Layers via RGB-space Geometry" by Jianchao Tan, Jyh-Ming Lien, Yotam Gingold from TOG 2016.


The pipeline is divided into two steps.

1 Convexhull Simplification:
	Input:
		Source Image
	Output:
		Simplified Hull Vertices(color palettes) on different simplification level(4 to 10 vertices)
		Users need to choose how numbers of vertices is reasonable, with help of our Web GUI.



2 Layers Extraction:
	Input: 
		Source Image, Simplified Hull vertices(color palettes, user choose numbers of vertices) and vertices order (Users choose which they like).
	Output:
	    Translucent Layers and Barycentric Coordinates. 
	    Users can use Barycentric Coordinates to do global recoloring on our Web GUI.


	We provide two choices in this step:
		One is our Main recovering method, including RGB version and RGBA version.
		The other is our ASAP recovering method. 

	Users can try our Main recovering method with RGB version first. RGB version will assume the first layer to be opaque, which means its color is also the background color. RGBA version is only used when you do not know what is background color. 



Dependencies:
	NumPy
	Scipy
	cvxopt (with 'glpk' solver)
	matplotlib
	OpenCV 3.1
	PIL or Pillow (Python Image Library): pip install Pillow


	
