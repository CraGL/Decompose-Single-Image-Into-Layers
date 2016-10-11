Input: 
	Source Image; 
	Color palettes ordering json file(User Specified); 
	Simplified ConvexHull Vertices(color palettes) json file; 
	Regularization Term Weights json file.

Output: 
	Translucent layers; 
	Barycentric Coordintate Weights.


Our Main method:
	(1) RGB version(first layer is default to be opaque, usually use this version):
        
        cd results
		python ../Layers_Extraction/SILD_RGB.py    apple.png  apple-final_simplified_hull_clip-06-color_order1.js apple-final_simplified_hull_clip-06.js  apple-final_simplified_hull_clip-06-color_order1-RGB_lap_adjusted_weights-poly3-opaque400-dynamic40000  --weights weights-poly3-opaque400-dynamic40000.js  --solve-smaller-factor 2


	(2) RGBA version(first layer is also translucent):
		
		cd results
		python ../Layers_Extraction/SILD_RGBA.py   apple.png  apple-final_simplified_hull_clip-06-color_order1.js apple-final_simplified_hull_clip-06.js  apple-final_simplified_hull_clip-06-color_order1-RGBA_lap_adjusted_weights-poly3-opaque400-dynamic40000  --weights weights-poly3-opaque400-dynamic40000.js  --solve-smaller-factor 2


Our ASAP method:

    cd results
	python ../Layers_Extraction/SILD_ASAP.py  apple-06-ASAP.json none

