##### copying from SILD_6b_Generating_transparent_layers_assuming_pixelchangeonly3times_using_optimization.ipynb. 
####  created at 2015.06.08

import numpy as np
import os
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from PIL import Image
from SILD_RGB import *
import pyximport
pyximport.install(reload_support=True)
from GteDistPointTriangle import *
from pprint import pprint 
import time


start=time.clock()

def Get_ASAP_alphas(img_label,tetra_prime,weights):

    img_shape=img_label.shape
    img_label=img_label.reshape((-1,3))
    img_label_backup=img_label.copy()

    num_layers=3
    # num_layers=len(tetra_prime)-1

    ### adjust the weights:
    if 'w_polynomial' in weights:
        # weights['w_polynomial'] *= 50000.0 
        weights['w_polynomial'] /= img_shape[2]

    if 'w_opaque' in weights:
        weights['w_opaque'] /= num_layers

    if 'w_spatial_static' in weights:
        weights['w_spatial_static'] /= num_layers

    if 'w_spatial_dynamic' in weights:
        weights['w_spatial_dynamic'] /= num_layers

    pprint(weights)


    hull=ConvexHull(tetra_prime)
    test_inside=Delaunay(tetra_prime)
    label=test_inside.find_simplex(img_label,tol=1e-8)
    # print len(label[label==-1])

    ### modify img_label[] to make all points are inside the simplified convexhull
    for i in range(img_label.shape[0]):
    #     print i
        if label[i]<0:
            dist_list=[]
            cloest_points=[]
            for j in range(hull.simplices.shape[0]):
                result = DCPPointTriangle( img_label[i], hull.points[hull.simplices[j]] )
                dist_list.append(result['distance'])
                cloest_points.append(result['closest'])
            dist_list=np.asarray(dist_list)
            index=np.argmin(dist_list)
            img_label[i]=cloest_points[index]

    ### assert
    test_inside=Delaunay(tetra_prime)
    label=test_inside.find_simplex(img_label,tol=1e-8)
    # print len(label[label==-1])
    assert(len(label[label==-1])==0)


    ### colors2xy dict
    colors2xy ={}
    unique_image_label=list(set(list(tuple(element) for element in img_label)))

    for element in unique_image_label:
        colors2xy.setdefault(tuple(element),[])
        
    for index in range(len(img_label)):
        element=img_label[index]
        colors2xy[tuple(element)].append(index)

    unique_colors=np.array(colors2xy.keys())
    # print len(img_label)
    img_label=unique_colors.copy()
    # print len(img_label)
    vertices_list=tetra_prime



    tetra_pixel_dict={}
    for face_vertex_ind in hull.simplices:
        # print face_vertex_ind
        if (face_vertex_ind!=0).all():
            i,j,k=face_vertex_ind
            tetra_pixel_dict.setdefault(tuple((i,j,k)),[])

    index_list=np.array(list(np.arange(len(img_label))))

    for face_vertex_ind in hull.simplices:
        if (face_vertex_ind!=0).all():
            # print face_vertex_ind
            i,j,k=face_vertex_ind
            tetra=np.array([vertices_list[0],vertices_list[i],vertices_list[j],vertices_list[k]])
            test_Del=Delaunay(tetra)
            # print len(index_list)
            if len(index_list)!=0:
                label=test_Del.find_simplex(img_label[index_list],tol=1e-8)
                chosen_index=list(index_list[label>=0])
                tetra_pixel_dict[tuple((i,j,k))]+=chosen_index
                index_list=np.array(list(set(index_list)-set(chosen_index)))

    # print index_list
    assert(len(index_list)==0)
 
            # temp_index_list=list(index_list)
            # for m in temp_index_list:
            #     label=test_Del.find_simplex(img_label[m])
            #     if label>=0:
            #         tetra_pixel_dict[tuple((i,j,k))].append(m)
            #         index_list.remove(m)


    pixel_num=0
    for key in tetra_pixel_dict:
        pixel_num+=len(tetra_pixel_dict[key])
    # print pixel_num
    assert(pixel_num==img_label.shape[0])



    ### input is like (0,1,2,3,4) then shortest_path_order is (1,2,3,4), 0th is background color, usually is white
    shortest_path_order=tuple(np.arange(len(tetra_prime))[1:])
    # print shortest_path_order

    alpha_list=np.zeros((img_label.shape[0],len(shortest_path_order)))
    color_list=vertices_list[np.asarray(list(shortest_path_order))]

    for vertice_tuple in tetra_pixel_dict:
        # print vertice_tuple
        vertice_index_inglobalorder=np.asarray(shortest_path_order)[np.asarray(sorted(list(shortest_path_order).index(s) for s in vertice_tuple))]
        vertice_index_inglobalorder_tuple=tuple(list(vertice_index_inglobalorder))
        # print vertice_index_inglobalorder_tuple
        
        colors=np.array([vertices_list[0],
                         vertices_list[vertice_index_inglobalorder_tuple[0]],
                         vertices_list[vertice_index_inglobalorder_tuple[1]],
                         vertices_list[vertice_index_inglobalorder_tuple[2]]
                        ])
                        
        pixel_index=np.array(tetra_pixel_dict[vertice_tuple])
        if len(pixel_index)!=0:
            arr=img_label[pixel_index]
            arr=arr.reshape((arr.shape[0],1,arr.shape[1]))
            Y0=np.ones((arr.shape[0]*3))*0.5
            Y=optimize(arr,colors,Y0,weights)
            sub_alpha_list=1.0-Y.reshape((arr.shape[0],3))            
            alpha_list[pixel_index[:,None],np.array([np.asarray(sorted(list(shortest_path_order).index(s) for s in vertice_tuple))])]=sub_alpha_list


    # print alpha_list.shape

    #### orignal shape alpha list
    total_alpha_list=np.zeros((len(img_label_backup),len(shortest_path_order)))
    for index in range(len(img_label)):
        element=img_label[index]
        index_list=colors2xy[tuple(element)]
        total_alpha_list[index_list,:]=alpha_list[index,:]

    total_alpha_list=total_alpha_list.reshape((img_shape[0], img_shape[1], len(shortest_path_order)))

    return total_alpha_list




def covnert_from_alphas_to_barycentricweights(alphas,epsilon=0.0):
    import numpy as np
#### first column of alphas should be all 1.0 (canvas is set to be opaque)
    def get_weight_from_alpha(alpha,epsilon=0.0):
        weight=np.ones(len(alpha))
        for i in range(len(weight)-1):
            temp1=1.0
            temp2=1.0
            for j in range(i,len(weight)):
                temp1*=(1.0-alpha[j])
            for j in range(i+1,len(weight)):
                temp2*=(1.0-alpha[j])
            weight[i]=temp2-temp1
        weight[-1]=alpha[-1]
        return weight

    weights=np.zeros(alphas.shape)
    for ind in range(len(weights)):
        alpha=alphas[ind]
        weights[ind]=get_weight_from_alpha(alpha)
    return weights



###### save as layers and composited layers.
def composite_layers( layers ):
    layers = np.asfarray( layers )
    ## Start with opaque white.
    out = 255*np.ones( layers[0].shape )[:,:,:3]
    for layer in layers:
        out += layer[:,:,3:]/255.*( layer[:,:,:3] - out )
    return out


def covnert_from_barycentric_weights_to_alphas(weights,order,epsilon=0.0/512):
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



if __name__ =='__main__':

    import sys
    
    filename=sys.argv[1]
    
    resize_flag = None
    if len( sys.argv ) > 2:
        resize_flag=sys.argv[2]

    # filename="test_fourcolors.json"


    import json
    with open(filename) as json_file:
        json_data = json.load(json_file)
        
    input_image_name=json_data["input_image_name"]
    input_vertices_file_name=json_data["input_vertices_file_name"]
    input_vertices_order_file_name=json_data["input_vertices_order_file_name"]
    output_prefix=json_data["output_prefix"]
    weights=json_data["weights"]
    pprint(weights)

    # output_prefix=os.path.splitext(input_vertices_file_name)[0]+"-ASAP-"+output_prefix+"-poly"+str(weights["w_polynomial"])+"-opaque"+str(weights["w_opaque"])

    images=Image.open(input_image_name).convert('RGB')
    if resize_flag=='resize_small':
        ### small size for test use#### 
        ratio=5.0
        images=images.resize((int(images.size[0]/ratio),int(images.size[1]/ratio)))
        ##### test end
    img_label=np.asfarray(images)
    img_shape=img_label.shape
    img_label_backup=img_label.copy()
    img_label=img_label/255.0

    input_color_order=np.asarray(json.load(open(input_vertices_order_file_name)))
    # print input_color_order 

    tetra_prime=np.asfarray(json.load(open(input_vertices_file_name))['vs'])
    tetra_prime=tetra_prime/255.0
    tetra_prime_backup=tetra_prime.copy()
    # print tetra_prime
    ##### reorder the vertices(colors).
    tetra_prime=tetra_prime[input_color_order,:]
    # print tetra_prime


    total_alpha_list=Get_ASAP_alphas(img_label,tetra_prime,weights)


    ##### save alphas as barycentric coordinates
    alphas=total_alpha_list.reshape((img_shape[0]*img_shape[1], -1 ))
    extend_alphas=ones((alphas.shape[0],alphas.shape[1]+1))
    extend_alphas[:,1:]=alphas
    #### first columns of extend_alphas are all 1.0
    barycentric_weights=covnert_from_alphas_to_barycentricweights(extend_alphas)
    # reconstruct_alphas=covnert_from_barycentric_weights_to_alphas(barycentric_weights,np.arange(barycentric_weights.shape[1]))
    # diff=reconstruct_alphas-extend_alphas
    # print abs(diff).max()


    temp=255.0*np.sum(barycentric_weights.reshape((barycentric_weights.shape[0],barycentric_weights.shape[1],1))*tetra_prime, axis=1)
    print temp.max()
    diff=temp-img_label_backup.reshape((-1,3))
    print abs(diff).max()
    print sqrt(square(diff).sum()/diff.shape[0])

    # barycentric_weights=barycentric_weights.reshape((img_shape[0],img_shape[1],-1))
    origin_order_barycentric_weights=np.ones(barycentric_weights.shape)

    #### to make the weights order is same as orignal input vertex order
    origin_order_barycentric_weights[:,input_color_order]=barycentric_weights

    # test_weights_diff1=origin_order_barycentric_weights-barycentric_weights
    # test_weights_diff2=barycentric_weights-barycentric_weights
    # print len(test_weights_diff1[test_weights_diff1==0])
    # print len(test_weights_diff2[test_weights_diff2==0])

    ####assert
    temp=255.0*np.sum(origin_order_barycentric_weights.reshape((origin_order_barycentric_weights.shape[0],origin_order_barycentric_weights.shape[1],1))*tetra_prime_backup, axis=1)
    diff=temp-img_label_backup.reshape((-1,3))
    print abs(diff).max()
    print diff.shape[0]
    print sqrt(square(diff).sum()/diff.shape[0])


    origin_order_barycentric_weights=origin_order_barycentric_weights.reshape((img_shape[0],img_shape[1],-1))


    import json
    output_all_weights_filename=output_prefix+"-layer_all_weights.js"
    with open(output_all_weights_filename,'wb') as myfile:
        json.dump({'weights': origin_order_barycentric_weights.tolist()}, myfile)

    for i in range(origin_order_barycentric_weights.shape[-1]):
        output_all_weights_map_filename=output_prefix+"-layer_optimization_all_weights_map-%02d.png" % i
        Image.fromarray((origin_order_barycentric_weights[:,:,input_color_order[i]]*255).round().clip(0,255).astype(uint8)).save(output_all_weights_map_filename)



    layers = []
    for li, color in enumerate( tetra_prime ):  ### colors are now in range[0.0,1.0] not [0,255]
        layer = ones( ( img_shape[0], img_shape[1], 4 ), dtype = np.uint8 )
        layer[:,:,:3] = np.asfarray(color*255.0).round().clip( 0,255 ).astype( np.uint8 )
        layer[:,:,3] = 255 if ( li == 0 ) else (total_alpha_list[:,:,li-1]*255.).round().clip( 0,255 ).astype( np.uint8 )
        layers.append( layer )
        outpath = output_prefix + '-layer%02d.png' % li
        Image.fromarray( layer ).save( outpath )
        print 'Saved layer:', outpath

    composited = composite_layers( layers )
    composited = composited.round().clip( 0, 255 ).astype( np.uint8 )
    outpath = output_prefix + '-composite.png'
    Image.fromarray( composited ).save( outpath )
    print 'Saved composite:', outpath


    img_diff=composited-np.asfarray(images)
    RMSE=sqrt(sum(square(img_diff))/(composited.shape[0]*composited.shape[1]))

    print 'img_shape is: ', img_diff.shape
    print 'max dist: ', sqrt(square(img_diff).sum(axis=2)).max()
    print 'median dist', median(sqrt(square(img_diff).sum(axis=2)))
    print 'RMSE: ', RMSE



    stop=time.clock()

    print 'time: ', stop-start

