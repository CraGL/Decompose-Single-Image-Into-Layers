from numpy import *
from itertools import izip as zip

## Normally it is bad practice to put a mutable value as the default parameter,
## because it is shared across all function calls, so its changed state will persist.
## In this case, though, I want that behavior.

def E_opaque( Y, scratches = {} ):
    return -dot( Y, Y )

def grad_E_opaque( Y, out, scratches = {} ):
    multiply( -2, Y, out )

def E_spatial_static( Y, Ytarget, scratches = {} ):
    if 'Y' not in scratches: scratches['Y'] = Y.copy()
    scratch = scratches['Y']
    
    subtract( Y, Ytarget, scratch )
    return dot( scratch, scratch )

def grad_E_spatial_static( Y, Ytarget, out, scratches = {} ):
    subtract( Y, Ytarget, out )
    out *= 2

def E_spatial_dynamic( Y, LTL, scratches = {} ):
    ## I don't see how to specify the output memory
    return dot( Y, LTL.dot( Y ) )

def grad_E_spatial_dynamic( Y, LTL, out, scratches = {} ):
    ## I don't see how to specify the output memory
    out[:] = LTL.dot( Y )
    out *= 2

def E_polynomial_pieces( Y, C, P, scratches = {} ):
    '''
    Y is a #pix-by-#layers flattened array
    C is a (#layers+1)-by-#channels not-flattened array (the 0-th layer is the background color)
    P is a #pix-by-#channels not-flattened array
    '''
    
    ### Reshape Y the way we want it.
    Y = Y.reshape( ( P.shape[0], C.shape[0]-1 ) )
    
    
    ## Allocate scratch space
    if 'F' not in scratches:
        scratches['F'] = empty( P.shape, dtype = Y.dtype )
    F = scratches['F']
    
    if 'M' not in scratches:
        ## We want the non-flattened Y's shape.
        assert len( Y.shape ) > 1
        scratches['M'] = empty( Y.shape, dtype = Y.dtype )
    M = scratches['M']
    
    if 'D' not in scratches:
        scratches['D'] = empty( ( C.shape[0]-1, C.shape[1] ), dtype = Y.dtype )
    D = scratches['D']
    
    if 'DM' not in scratches:
        scratches['DM'] = empty( ( P.shape[0], D.shape[0], D.shape[1] ), dtype = Y.dtype )
    DM = scratches['DM']
    
    if 'energy_presquared' not in scratches:
        scratches['energy_presquared'] = empty( F.shape, dtype = Y.dtype )
    energy_presquared = scratches['energy_presquared']
    
    
    ## Compute F
    subtract( C[newaxis,-1,:], P, F )
    
    ## Compute M
    cumprod( Y[:,::-1], axis = 1, out = M )
    M = M[:,::-1]
    
    ## Compute D
    subtract( C[:-1,:], C[1:,:], D )
    
    ## Finish the computation
    multiply( D[newaxis,...], M[...,newaxis], DM )
    DM.sum( 1, out = energy_presquared )
    energy_presquared += F

def E_polynomial( Y, C, P, scratches = {} ):
    E_polynomial_pieces( Y, C, P, scratches )
    
    energy_presquared = scratches['energy_presquared']
    
    square( energy_presquared, energy_presquared )
    return energy_presquared.sum()

def gradY_E_polynomial( Y, C, P, out, scratches = {} ):
    E_polynomial_pieces( Y, C, P, scratches )
    
    ### Reshape Y the way we want it.
    Y = Y.reshape( ( P.shape[0], C.shape[0]-1 ) )
    
    energy_presquared = scratches['energy_presquared']
    D = scratches['D']
    M = scratches['M']
    DM = scratches['DM']
    
    if 'Mi' not in scratches:
        scratches['Mi'] = empty( DM.shape, dtype = Y.dtype )
    Mi = scratches['Mi']
    assert Mi.shape[1] == Y.shape[1]
    
    if 'Yli' not in scratches:
        scratches['Yli'] = empty( Y.shape[0], dtype = Y.dtype )
    Yli = scratches['Yli']
    
    for li in range( Y.shape[1] ):
        Yli[:] = Y[:,li]
        Y[:,li] = 1.
        ## UPDATE: I cannot use cumprod() when aliasing
        ## the input and output parameters and one is the reverse of the other.
        cumprod( Y[:,::-1], axis = 1, out = M )
        Y[:,li] = Yli
        Mr = M[:,::-1]
        Mr[:,li+1:] = 0.
        
        multiply( D[newaxis,...], Mr[...,newaxis], DM )
        DM.sum( 1, out = Mi[:,li,:] )
    
    multiply( energy_presquared[:,newaxis,:], Mi, Mi )
    out.shape = Y.shape
    Mi.sum( 2, out = out )
    out *= 2.
    out.shape = ( prod( Y.shape ), )

def gen_energy_and_gradient( img, layer_colors, weights, img_spatial_static_target = None, scratches = None ):
    '''
    Given a rows-by-cols-by-#channels 'img', where channels are the 3 color channels,
    and (#layers+1)-by-#channels 'layer_colors' (the 0-th color is the background color),
    and a dictionary of floating-point or None weights { w_spatial, w_opacity },
    and an optional parameter 'img_spatial_static_target' which are the target values for 'w_spatial_static' (if not flattened, it will be),
    and an optional parameter 'scratches' which should be a dictionary that will be used to store scratch space between calls to this function (use only *if* arguments are the same size),
    returns a tuple of functions:
        ( e, g )
        where e( Y ) computes the scalar energy of a flattened rows-by-cols-by-#layers array of (1-alpha) values,
        and g( Y ) computes the gradient of e.
    '''
    
    img = asfarray( img )
    layer_colors = asfarray( layer_colors )
    
    assert len( img.shape ) == 3
    assert len( layer_colors.shape ) == 2
    assert img.shape[2] == layer_colors.shape[1]
    
    # from pprint import pprint
    # pprint( weights )
    assert set( weights.keys() ).issubset( set([ 'w_polynomial', 'w_opaque', 'w_spatial_static', 'w_spatial_dynamic' ]) )
    
    C = layer_colors
    P = img.reshape( -1, img.shape[2] )
    
    num_layers = C.shape[0]-1
    Ylen = P.shape[0] * num_layers
    
    if 'w_spatial_static' in weights:
        assert img_spatial_static_target is not None
        Yspatial_static_target = img_spatial_static_target.ravel()
    
    if 'w_spatial_dynamic' in weights:
        # print 'Preparing a Laplacian matrix for E_spatial_dynamic...'
        import fast_energy_laplacian
        import scipy.sparse
        # print '    Generating L...'
        LTL = fast_energy_laplacian.gen_grid_laplacian( img.shape[0], img.shape[1] )
        # print '    Computing L.T*L...'
        # LTL = LTL.T * LTL
        # print '    Replicating L.T*L for all layers...'
        ## Now repeat LTL #layers times.
        ## Because the layer values are the innermost dimension,
        ## every entry (i,j, val) in LTL should be repeated
        ## (i*#layers + k, j*#layers + k, val) for k in range(#layers).
        LTL = LTL.tocoo()
        ## Store the shape. It's a good habit, because there may not be a nonzero
        ## element in the last row and column.
        shape = LTL.shape
        
        ## There is a "fastest" version below.
        '''
        rows = zeros( LTL.nnz * num_layers, dtype = int )
        cols = zeros( LTL.nnz * num_layers, dtype = int )
        vals = zeros( LTL.nnz * num_layers )
        count = 0
        ks = arange( num_layers )
        for r, c, val in zip( LTL.row, LTL.col, LTL.data ):
            ## Slow
            #for k in range( num_layers ):
            #    rows.append( r*num_layers + k )
            #    cols.append( c*num_layers + k )
            #    vals.append( val )
            
            ## Faster
            rows[ count : count + num_layers ] = r*num_layers + ks
            cols[ count : count + num_layers ] = c*num_layers + ks
            vals[ count : count + num_layers ] = val
            count += num_layers
            
        assert count == LTL.nnz * num_layers
        '''
        
        ## Fastest
        ks = arange( num_layers )
        rows = ( repeat( asarray( LTL.row ).reshape( LTL.nnz, 1 ) * num_layers, num_layers, 1 ) + ks ).ravel()
        cols = ( repeat( asarray( LTL.col ).reshape( LTL.nnz, 1 ) * num_layers, num_layers, 1 ) + ks ).ravel()
        vals = ( repeat( asarray( LTL.data ).reshape( LTL.nnz, 1 ), num_layers, 1 ) ).ravel()
        
        LTL = scipy.sparse.coo_matrix( ( vals, ( rows, cols ) ), shape = ( shape[0]*num_layers, shape[1]*num_layers ) ).tocsr()
        # print '...Finished.'
    
    if scratches is None:
        scratches = {}
    
    def e( Y ):
        e = 0.
        
        if 'w_polynomial' in weights:
            e += weights['w_polynomial'] * E_polynomial( Y, C, P, scratches )
        
        if 'w_opaque' in weights:
            e += weights['w_opaque'] * E_opaque( Y, scratches )
        
        if 'w_spatial_static' in weights:
            e += weights['w_spatial_static'] * E_spatial_static( Y, Yspatial_static_target, scratches )
        
        if 'w_spatial_dynamic' in weights:
            e += weights['w_spatial_dynamic'] * E_spatial_dynamic( Y, LTL, scratches )
        
        # print 'Y:', Y
        # print 'e:', e
        
        return e
    
    ## Preallocate this memory
    gradient_space = [ zeros( Ylen ), zeros( Ylen ) ]
    # total_gradient = zeros( Ylen )
    # gradient_term = zeros( Ylen )
    
    def g( Y ):
        total_gradient = gradient_space[0]
        gradient_term = gradient_space[1]
        
        total_gradient[:] = 0.
        
        if 'w_polynomial' in weights:
            gradY_E_polynomial( Y, C, P, gradient_term, scratches )
            gradient_term *= weights['w_polynomial']
            total_gradient += gradient_term
        
        if 'w_opaque' in weights:
            grad_E_opaque( Y, gradient_term, scratches )
            gradient_term *= weights['w_opaque']
            total_gradient += gradient_term
        
        if 'w_spatial_static' in weights:
            grad_E_spatial_static( Y, Yspatial_static_target, gradient_term, scratches )
            gradient_term *= weights['w_spatial_static']
            total_gradient += gradient_term
        
        if 'w_spatial_dynamic' in weights:
            grad_E_spatial_dynamic( Y, LTL, gradient_term, scratches )
            gradient_term *= weights['w_spatial_dynamic']
            total_gradient += gradient_term
        
        # print 'Y:', Y
        # print 'total_gradient:', total_gradient
        
        return total_gradient
    
    return e, g

# def composite_layers( layers ):
#     layers = asfarray( layers )
    
#     ## Start with opaque white.
#     out = 255*ones( layers[0].shape )[:,:,:3]
#     for layer in layers:
#         out += layer[:,:,3:]/255.*( layer[:,:,:3] - out )
    
#     return out


def composite_layers( layers ):
    layers = asfarray( layers )
    
    out=zeros(layers[0].shape)
    out_RGB_pre = zeros( layers[0].shape )[:,:,:3]
    out_alpha=zeros(layers[0].shape)[:,:,3:]
    for layer in layers:
        out_alpha=out_alpha*(1-layer[:,:,3:]/255.0)+layer[:,:,3:]/255.0
        out_RGB_pre = out_RGB_pre*(1-layer[:,:,3:]/255.0)+layer[:,:,3:]/255.0*layer[:,:,:3]

    out_alpha_reshape=out_alpha.reshape((out_alpha.shape[:2]))
    out[:,:,3:]=(out_alpha*255.0).round().clip(0,255)
    out[:,:,:3][out_alpha_reshape==0.0]=array([0,0,0])
    out[:,:,:3][out_alpha_reshape!=0.0]=(out_RGB_pre[out_alpha_reshape!=0.0]/out_alpha[out_alpha_reshape!=0.0]).round().clip(0,255)

    return out



def optimize( arr, colors, Y0, weights, img_spatial_static_target = None, scratches = None, saver = None ):
    '''
    Given a rows-by-cols-by-#channels array 'arr', where channels are the 4 color channels,
    and (#layers+1)-by-#channels 'colors' (the 0-th color is the background color),
    and rows-by-cols-by-#layers array 'Y0' of initial (1-alpha) values for each pixel (flattened or not),
    and a dictionary of floating-point or None weights { w_polynomial, w_opacity, w_spatial_dynamic, w_spatial_static },
    and an optional parameter 'img_spatial_static_target' which are the target values for 'w_spatial_static' (if not flattened, it will be),
    and an optional parameter 'scratches' which should be a dictionary that will be used to store scratch space between calls to this function (use only *if* arguments are the same size),
    and an optional parameter 'saver' which will be called after every iteration with the current state of Y.
    returns a rows-by-cols-#layers array of optimized Y values, which are (1-alpha).
    '''
    
    import scipy.optimize
    import time
    start=time.clock()
    
    Y0 = Y0.ravel()
    
    Ylen = len( Y0 )
    
    e, g = gen_energy_and_gradient( arr, colors, weights, img_spatial_static_target = img_spatial_static_target, scratches = scratches )
    
    bounds = zeros( ( Ylen, 2 ) )
    bounds[:,1] = 1.
    
    ## Save the result-in-progress in case the users presses control-C.
    ## [number of iterations, last Y]
    Ysofar = [0,None]
    def callback( xk ):
        Ysofar[0] += 1
        ## Make a copy
        xk = array( xk )
        Ysofar[1] = xk
        
        if saver is not None: saver( xk )
    
    # print 'Optimizing...'
    # start = time.clock()
    
    try:
        ## WOW! TNC does a really bad job on our problem.
        # opt_result = scipy.optimize.minimize( e, Y0, method = 'TNC', jac = g, bounds = bounds )
        ## I did an experiment with the 'tol' parameter.
        ## I checked in the callback for a max/total absolute difference less than 1./255.
        ## Passing tol directly doesn't work, because the solver we are using (L-BFGS-B)
        ## normalizes it by the maximum function value, whereas we want an
        ## absolute stopping criteria.
        ## Max difference led to stopping with visible artifacts.
        ## Total absolute difference terminated on the very iteration that L-BFGS-B did
        ## anyways.
        opt_result = scipy.optimize.minimize( e, Y0, jac = g, bounds = bounds, callback = callback 
             # ,method='L-BFGS-B'
             # ,options={'ftol': 1e-4, 'gtol': 1e-4}
            )
    
    except KeyboardInterrupt:
        ## If the user 
        print 'KeyboardInterrupt after %d iterations!' % Ysofar[0]
        Y = Ysofar[1]
        ## Y will be None if we didn't make it through 1 iteration before a KeyboardInterrupt.
        if Y is None:
            Y = -31337*ones( ( arr.shape[0], arr.shape[1], len( colors )-1 ) )
    
    else:
        # print opt_result
        Y = opt_result.x
    
    # duration = time.clock() - start
    # print '...Finished optimizing in %.3f seconds.' % duration

    end = time.clock()
    print 'Optimize an image of size ', Y.shape, ' took ', (end-start), ' seconds.'
    
    Y = Y.reshape( arr.shape[0], arr.shape[1], len( colors )-1 )
    return Y

def run_one( imgpath, orderpath, colorpath, outprefix, weightspath = None, save_every = None, solve_smaller_factor = None, too_small = None ):
    '''
    Given a path `imgpath` to an image,
    a path `colorpath` to a JSON file containing an array of RGBA triplets of layer colors (the 0-th color is the background color),
    a prefix `outprefix` to use for saving files,
    an optional path `weightspath` to a JSON file containing a dictionary of weight values,
    an optional positive number `save_every` which specifies how often to save progress,
    an optional positive integer `solve_smaller_factor` which, if specified,
    will first solve on a smaller image whose dimensions are `1/solve_smaller_factor` the full size image,
    and an optional positive integer `too_small` which, if specified, determines
    the limit of the `solve_smaller_factor` recursion as the minimum image size (width or height),
    runs optimize() on it and saves the output to e.g. `outprefix + "-layer01.png"`.
    '''
    
    import json, os
    from PIL import Image
    import time
    
    #### origin_arr is RGBA format!!! origin_colors are hull vertices. C0 t0 Cn, and is RGB format
    origin_arr = asfarray( Image.open( imgpath ).convert( 'RGBA' ) )
    order=asarray(json.load(open(orderpath)))
    # print order
    origin_colors = asfarray(json.load(open(colorpath))['vs'])
    # print origin_colors
    origin_colors_backup=origin_colors.copy()
    origin_colors=origin_colors[order,:]
    # print origin_colors
    
    ### arr is R*alpha, G*alpha, B*alpha, X*alpha format. X=255.0
    arr=zeros((origin_arr.shape[0],origin_arr.shape[1],4))
    arr[:,:,:3]=origin_arr[:,:,:3]*origin_arr[:,:,3:]/255.0
    arr[:,:,3:]=255.0*origin_arr[:,:,3:]/255.0

    # colors are modified vertices. It is RGBX format. Now, first vertices are 0,0,0,0. others are R,G,B,255.0
    colors=zeros((len(origin_colors)+1,4))
    colors[1:,:3]=origin_colors
    colors[0,:]=0.0
    colors[1:,3]=255.0
    

    colors=colors/255.0
    arr=arr/255.0

    # print colors





    assert solve_smaller_factor is None or int( solve_smaller_factor ) == solve_smaller_factor
    
    if save_every is None:
        save_every = 100.
    
    if solve_smaller_factor is None:
        solve_smaller_factor = 2
    
    if too_small is None:
        too_small = 5
    
    # arr = arr[:1,:1,:]
    # colors = colors[:3]
    
    kSaveEverySeconds = save_every
    ## [ number of iterations, time of last save, arr.shape ]
    last_save = [ None, None, None ]
    def reset_saver( arr_shape ):
        last_save[0] = 0
        last_save[1] = time.clock()
        last_save[2] = arr_shape
    def saver( xk ):
        arr_shape = last_save[2]
        
        last_save[0] += 1
        now = time.clock()
        ## Save every 10 seconds!
        if now - last_save[1] > kSaveEverySeconds:
            print 'Iteration', last_save[0]
            save_results( xk, origin_colors, arr_shape, outprefix )
            ## Get the time again instead of using 'now', because that doesn't take into
            ## account the time to actually save the images, which is a lot for large images.
            last_save[1] = time.clock()
    
    Ylen = arr.shape[0]*arr.shape[1]*( len(colors) - 1 )
    
    # Y0 = random.random( Ylen )
    # Y0 = zeros( Ylen ) + 0.0001
    Y0 = .5*ones( Ylen )
    # Y0 = ones( Ylen )
    
    static = None
    if weightspath is not None:
        weights = json.load( open( weightspath ) )
    else:
        weights = { 'w_polynomial': 375., 'w_opaque': 1., 'w_spatial_dynamic': 100. }
        # weights = { 'w_polynomial': 1., 'w_opaque': 100. }
        # weights = { 'w_opaque': 100. }
        # weights = { 'w_spatial_static': 100. }
        # static = 0.75 * ones( Ylen )
        # weights = { 'w_spatial_dynamic': 100. }
        # weights = { 'w_spatial_dynamic': 100., 'w_opaque': 100. }

    
    num_layers=len(colors)-1
    # print arr.shape[2]
    # print num_layers
    ### adjust the weights:
    if 'w_polynomial' in weights:
        # weights['w_polynomial'] *= 50000.0 #### old one is 255*255
        weights['w_polynomial'] /= arr.shape[2]
    
    if 'w_opaque' in weights:
        weights['w_opaque'] /= num_layers
    
    if 'w_spatial_static' in weights:
        weights['w_spatial_static'] /= num_layers
    
    if 'w_spatial_dynamic' in weights:
        weights['w_spatial_dynamic'] /= num_layers


    
    if solve_smaller_factor != 1:
        assert solve_smaller_factor > 1
        
        def optimize_smaller( solve_smaller_factor, large_arr, large_Y0, large_img_spatial_static_target ):
            ## Terminate recursion if the image is too small.
            if large_arr.shape[0]//solve_smaller_factor < too_small or large_arr.shape[1]//solve_smaller_factor < too_small:
                return large_Y0
            
            ## small_arr = downsample( large_arr )
            small_arr = large_arr[::solve_smaller_factor,::solve_smaller_factor]
            ## small_Y0 = downsample( large_Y0 )
            small_Y0 = large_Y0.reshape( large_arr.shape[0], large_arr.shape[1], -1 )[::solve_smaller_factor,::solve_smaller_factor].ravel()
            ## small_img_spatial_static_target = downsample( large_img_spatial_static_target )
            small_img_spatial_static_target = None
            if large_img_spatial_static_target is not None:
                small_img_spatial_static_target = large_img_spatial_static_target.reshape( arr.shape[0], arr.shape[1], -1 )[::solve_smaller_factor,::solve_smaller_factor].ravel()
            
            ## get an improved Y by recursively shrinking
            small_Y1 = optimize_smaller( solve_smaller_factor, small_arr, small_Y0, small_img_spatial_static_target )
            
            ## solve on the downsampled problem
            print '==> Optimizing on a smaller image:', small_arr.shape, 'instead of', large_arr.shape
            reset_saver( small_arr.shape )
            small_Y = optimize( small_arr, colors, small_Y1, weights, img_spatial_static_target = small_img_spatial_static_target, saver = saver )
            
            ## save the intermediate solution.
            saver( small_Y )
            
            ## large_Y1 = upsample( small_Y )
            ### 1 Make a copy
            large_Y1 = array( large_Y0 ).reshape( large_arr.shape[0], large_arr.shape[1], -1 )
            ### 2 Fill in as much as will fit using numpy.repeat()
            small_Y = small_Y.reshape( small_arr.shape[0], small_arr.shape[1], -1 )
            small_Y_upsampled = repeat( repeat( small_Y, solve_smaller_factor, 0 ), solve_smaller_factor, 1 )
            large_Y1[:,:] = small_Y_upsampled[ :large_Y1.shape[0], :large_Y1.shape[1] ]
            # large_Y1[ :small_Y.shape[0]*solve_smaller_factor, :small_Y.shape[1]*solve_smaller_factor ] = repeat( repeat( small_Y, solve_smaller_factor, 0 ), solve_smaller_factor, 1 )
            ### 3 The right and bottom edges may have been missed due to rounding
            # large_Y1[ small_Y.shape[0]*solve_smaller_factor:, : ] = large_Y1[ small_Y.shape[0]*solve_smaller_factor - 1 : small_Y.shape[0]*solve_smaller_factor, : ]
            # large_Y1[ :, small_Y.shape[1]*solve_smaller_factor: ] = large_Y1[ :, small_Y.shape[1]*solve_smaller_factor - 1 : small_Y.shape[1]*solve_smaller_factor ]
            
            return large_Y1.ravel()
        
        Y0 = optimize_smaller( solve_smaller_factor, arr, Y0, static )
    
    reset_saver( arr.shape )
    Y = optimize( arr, colors, Y0, weights, img_spatial_static_target = static, saver = saver )
    

    composited_img=save_results( Y, origin_colors, arr.shape, outprefix )
    img_diff=composited_img-origin_arr
    RMSE=sqrt(sum(square(img_diff))/(origin_arr.shape[0]*origin_arr.shape[1]))

    print 'img_shape is: ', img_diff.shape
    print 'max dist: ', sqrt(square(img_diff).sum(axis=2)).max()
    print 'median dist', median(sqrt(square(img_diff).sum(axis=2)))
    print 'RMSE: ', RMSE


    ##### save alphas as barycentric coordinates
    alphas=1. - Y.reshape((arr.shape[0]*arr.shape[1], -1 ))
 
    barycentric_weights=covnert_from_alphas_to_barycentricweights(alphas)
    # print barycentric_weights.shape

    # print barycentric_weights.max()
    # print barycentric_weights.min()
    # print barycentric_weights.sum(axis=1).max()
    # print barycentric_weights.sum(axis=1).min()
    # print len(arange(len(barycentric_weights))[abs(barycentric_weights.sum(axis=1)-1.0)<=1e-2])

    origin_order_barycentric_weights=ones(barycentric_weights.shape)
    #### to make the weights order is same as orignal input vertex order
    origin_order_barycentric_weights[:,order]=barycentric_weights

    # test_weights_diff1=origin_order_barycentric_weights-barycentric_weights
    # test_weights_diff2=barycentric_weights-barycentric_weights
    # print len(test_weights_diff1[test_weights_diff1==0])
    # print len(test_weights_diff2[test_weights_diff2==0])


    ####assert
    temp=sum(origin_order_barycentric_weights.reshape((origin_order_barycentric_weights.shape[0],origin_order_barycentric_weights.shape[1],1))*origin_colors_backup, axis=1)
    diff=temp-255.0*arr.reshape((-1,4))[:,:3]
    # assert(abs(diff).max()<0.5)
    print diff.shape[0]
    print abs(diff).max()
    print median(abs(diff))
    print sqrt(square(diff).sum()/diff.shape[0])


    # #### assert2
    # test_origin_order_barycentric_weights=origin_order_barycentric_weights/origin_order_barycentric_weights.sum(axis=1).reshape((-1,1))
    # temp=sum(test_origin_order_barycentric_weights.reshape((test_origin_order_barycentric_weights.shape[0],test_origin_order_barycentric_weights.shape[1],1))*origin_colors_backup, axis=1)
    # diff=temp-origin_arr.reshape((-1,4))[:,:3]
    # # assert(abs(diff).max()<0.5)
    # print abs(diff).max()
    # print diff.shape[0]
    # print median(abs(diff))
    # print sqrt(square(diff).sum(axis=1)).sum()/diff.shape[0]


    # print test_origin_order_barycentric_weights.max()
    # print test_origin_order_barycentric_weights.min()
    # print test_origin_order_barycentric_weights.sum(axis=1).max()
    # print test_origin_order_barycentric_weights.sum(axis=1).min()


    # #### assert3
    # test_origin_order_barycentric_weights=origin_order_barycentric_weights/arr[:,:,3].reshape((-1,1))
    # temp=sum(test_origin_order_barycentric_weights.reshape((test_origin_order_barycentric_weights.shape[0],test_origin_order_barycentric_weights.shape[1],1))*origin_colors_backup, axis=1)
    # diff=temp-origin_arr.reshape((-1,4))[:,:3]
    # # assert(abs(diff).max()<0.5)
    # print abs(diff).max()
    # print diff.shape[0]
    # print median(abs(diff))
    # print sqrt(square(diff).sum(axis=1)).sum()/diff.shape[0]


    # print test_origin_order_barycentric_weights.max()
    # print test_origin_order_barycentric_weights.min()
    # print test_origin_order_barycentric_weights.sum(axis=1).max()
    # print test_origin_order_barycentric_weights.sum(axis=1).min()



#### expand the weights. because first weights are w_-1, which is not zeros.
    origin_order_barycentric_weights_expand=zeros((origin_order_barycentric_weights.shape[0],origin_order_barycentric_weights.shape[1]+1))
    print origin_order_barycentric_weights_expand.shape
    origin_order_barycentric_weights_expand[:,0]=1.0-origin_order_barycentric_weights.sum(axis=1)
    origin_order_barycentric_weights_expand[:,1:]=origin_order_barycentric_weights
    origin_order_barycentric_weights_expand=origin_order_barycentric_weights_expand.reshape((arr.shape[0],arr.shape[1],-1))



    import json
    output_all_weights_filename=outprefix+"-layer_optimization_all_weights.js"
    with open(output_all_weights_filename,'wb') as myfile:
        json.dump({'weights': origin_order_barycentric_weights_expand.tolist()}, myfile)
    
    expand_order=ones(len(order)+1)
    expand_order[0]=0
    expand_order[1:]=order+1
    for i in range(origin_order_barycentric_weights_expand.shape[-1]):
        output_all_weights_map_filename=outprefix+"-layer_optimization_all_weights_map-%02d.png" % i
        Image.fromarray((origin_order_barycentric_weights_expand[:,:,expand_order[i]]*255).round().clip(0,255).astype(uint8)).save(output_all_weights_map_filename)
    return Y
    



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



## here, colors are read from file. it is #layer by 3 format.
def save_results( Y, colors, img_shape, outprefix ):
    from PIL import Image
    
    alphas = 1. - Y.reshape( img_shape[0], img_shape[1], -1 )
    layers = []
    for li, color in enumerate( colors ):
        layer = ones( ( img_shape[0], img_shape[1], 4 ), dtype = uint8 )
        layer[:,:,:3] = asfarray(color).round().clip( 0,255 ).astype( uint8 )
        layer[:,:,3] = (alphas[:,:,li]*255.).round().clip( 0,255 ).astype( uint8 )
        layers.append( layer )
        outpath = outprefix + '-layer%02d.png' % li
        Image.fromarray( layer ).save( outpath )
        print 'Saved layer:', outpath
    
    composited = composite_layers( layers )
    composited = composited.round().clip( 0, 255 ).astype( uint8 )

    outpath = outprefix + '-composite.png'
    Image.fromarray( composited ).save( outpath )
    print 'Saved composite:', outpath

    return composited

if __name__ == '__main__':
    import sys
    

    ##### example:
    ##### python -u ../../../fast_energy_RGBA_lap_adjusted_weights.py  eye.png  eye_final_simplified_hull-05-color_order.js  eye_final_simplified_hull-05.js  eye_final_simplified_hull-05-RGBA_lap_adjusted_weights-poly3-opaque400-dynamic40000  --weights weights-poly3-opaque400-dynamic40000.js
    
    def usage():
        print >> sys.stderr, "Usage:", sys.argv[0], "path/to/image path/to/layer_color_list.js path/to/output [--weights /path/to/weights.js] [--save-every save_every_N_seconds N] [--solve-smaller-factor F] [--too-small T]"
        print >> sys.stderr, "NOTE: The 0-th element of layer_color_list is the background color."
        print >> sys.stderr, 'NOTE: Files will be saved to "path/to/output-composite.png" and "path/to/output-layer01.png"'
        sys.exit(-1)
    
    args = list( sys.argv[1:] )
    
    try:
        
        weightspath = None
        try:
            index = args[:-1].index( '--weights' )
            weightspath = args[ index+1 ]
            del args[ index : index+2 ]
        except ValueError: pass
        
        save_every = None
        try:
            index = args[:-1].index( '--save-every' )
            save_every = int( args[ index+1 ] )
            del args[ index : index+2 ]
        except ValueError: pass
        
        solve_smaller_factor = None
        try:
            index = args[:-1].index( '--solve-smaller-factor' )
            solve_smaller_factor = int( args[ index+1 ] )
            del args[ index : index+2 ]
        except ValueError: pass
        
        too_small = None
        try:
            index = args[:-1].index( '--too-small' )
            too_small = int( args[ index+1 ] )
            del args[ index : index+2 ]
        except ValueError: pass
    
    except Exception:
        usage()
    
    if len( args ) != 4: usage()
    
    image_path, orderpath, color_path, output_prefix = args
    import time
    start=time.clock()
    run_one( image_path, orderpath, color_path, output_prefix, weightspath = weightspath, save_every = save_every, solve_smaller_factor = solve_smaller_factor, too_small = too_small )
    end=time.clock()
    print 'time: ', end-start