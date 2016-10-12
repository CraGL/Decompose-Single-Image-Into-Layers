import json
import PIL.Image as Image
import numpy as np
import sys

def covnert_from_barycentric_weights_to_alphas(weights,order,epsilon=1.0/512):
#     weights[weights<=(epsilon/len(order))]=0.0
    def get_alpha_i(index, weight,epsilon=0.0):
        total=sum(weight[0:index+1])
        if total<=epsilon:
    #     if total==0:
            return 0
        else:
            return weight[index]/total

    weights_ordered=weights[:,order]
    alphas=np.ones(weights_ordered.shape)
    for ind in range(len(weights_ordered)):
        weight=weights_ordered[ind]
        for i in range(1,len(weight)): ### alpha_0 is default 1.0, opaque white.
            alphas[ind,i]=get_alpha_i(i,weight)
    return alphas




#####example usage: 
##### python ../Layer_Extraction/from_weights_to_alphas.py apple-06-layers-RGB-layer_optimization_all_weights.js apple-final_simplified_hull_clip-06.js apple-06-vertex_order2.js 

if __name__=="__main__":

    weights_filename=sys.argv[1]
    vertices_filename=sys.argv[2]
    order_filename=sys.argv[3]

    with open(weights_filename) as weightsfile:
        weights=np.asarray(json.load(weightsfile)['weights'])

    with open(vertices_filename) as verticesfile:
        vertices=np.asarray(json.load(verticesfile)['vs'])

    with open(order_filename) as order_file:
        order=np.asarray(json.load(order_file))
    
    shape=weights.shape
    num=shape[-1]

    print order
    alphas=covnert_from_barycentric_weights_to_alphas(weights.reshape((-1,num)),order)
    print alphas.shape
    print vertices
    vertices=vertices[order,:]
    print vertices
    alphas=alphas.reshape(shape)
    layer=np.ones((shape[0],shape[1],4))
    for i in range(num):
        alpha=alphas[:,:,i]
        layer[:,:,-1]=alpha
        layer[:,:,:-1]=vertices[i].reshape((1,1,-1))/255.0
        Image.fromarray((layer*255).clip(0,255).round().astype(np.uint8)).save('Converted_layers-%02d.png'%i)






