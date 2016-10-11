'''
Ported from: http://www.geometrictools.com/GTEngine/Include/GteDistPointTriangle.h
'''
from libc.math cimport sqrt

ctypedef double Real

cdef scale( Real s, Real v[3], Real out[3] ):
    out[0] = s*v[0]
    out[1] = s*v[1]
    out[2] = s*v[2]

cdef subtract( Real a[3], Real b[3], Real out[3] ):
    out[0] = a[0] - b[0]
    out[1] = a[1] - b[1]
    out[2] = a[2] - b[2]

cdef add( Real a[3], Real b[3], Real out[3] ):
    out[0] = a[0] + b[0]
    out[1] = a[1] + b[1]
    out[2] = a[2] + b[2]


cdef scale2( Real s, Real v[2], Real out[2] ):
    out[0] = s*v[0]
    out[1] = s*v[1]

cdef subtract2( Real a[2], Real b[2], Real out[2] ):
    out[0] = a[0] - b[0]
    out[1] = a[1] - b[1]


cdef add2( Real a[2], Real b[2], Real out[2] ):
    out[0] = a[0] + b[0]
    out[1] = a[1] + b[1]




cdef Real dot3( Real a[3], Real b[3] ):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

cdef GetMinEdge02( Real a11, Real b1, Real p[2] ):
    p[0] = 0.
    p[1] = 0.
    
    if b1 >= 0.:
        p[1] = 0.
    elif (a11 + b1 <= 0.):
        p[1] = 1.
    else:
        p[1] = -b1 / a11

cdef GetMinEdge12( Real a01, Real a11, Real b1, Real f10, Real f01, Real p[2] ):
    cdef Real h0
    cdef Real h1

    p[0] = 0.
    p[1] = 0.

    h0 = a01 + b1 - f10
    if (h0 >= 0):
        p[1] = 0
    else:
        h1 = a11 + b1 - f01
        if (h1 <= 0):
            p[1] = 1
        else:
            p[1] = h0 / (h0 - h1)
    p[0] = 1. - p[1]

cdef GetMinInterior( Real p0[2], Real h0, Real p1[2], Real h1, Real p[2] ):
    cdef Real z = h0 / (h0 - h1)
    
    # p = (1. - z) * p0 + z * p1

    cdef Real tmp1[2]
    cdef Real tmp2[2]

    scale2( 1. - z, p0, tmp1 )
    scale2( z, p1, tmp2 )

    add2( tmp1, tmp2, p )


cdef struct Result:
     Real parameter[3]
     Real closest[3]
     Real distance
     Real sqrDistance



cpdef Result DCPPointTriangle( object[Real, ndim=1, mode="c"] point_p, object[Real, ndim=2, mode="c"] triangle_p ):
    '''
    Given a 3-dimensional point as a numpy.array
    and a triangle as a sequence of 3 same-dimensional points (also numpy.arrays),
    returns an object with properties:
        .distance: the distance from the point to the triangle
        .sqrDistance: the square of .distance
        .parameter[3]: the three barycentric coordinates for the closest point in the triangle (i.e. .closest = \sum_{i=0}^2 .parameter[i]*triangle[i])
        .closest: the closest point on the triangle to 'point'
    '''
    cdef Real point[3]
    cdef Real triangle[3][3]

    for i in range(3):
        point[i]=point_p[i]

    for i in range(3):
        for j in range(3):
            triangle[i][j]=triangle_p[i,j]


    cdef Real p[2]
    cdef Real p0[2]
    cdef Real p1[2]
    
    cdef Real diff[3]
    subtract( point, triangle[0], diff )

    cdef Real edge0[3]
    cdef Real edge1[3]
    subtract( triangle[1], triangle[0], edge0 )
    subtract( triangle[2], triangle[0], edge1 )

 
    
    cdef Real a00 = dot3(edge0, edge0)
    cdef Real a01 = dot3(edge0, edge1)
    cdef Real a11 = dot3(edge1, edge1)
    cdef Real b0 = -dot3(diff, edge0)
    cdef Real b1 = -dot3(diff, edge1)
    
    cdef Real f00 = b0
    cdef Real f10 = b0 + a00
    cdef Real f01 = b0 + a01

    # print f00, f10, f01

    cdef Real h0, h1, dt1
    
    ## Compute the endpoints p0 and p1 of the segment.  The segment is
    ## parameterized by L(z) = (1-z)*p0 + z*p1 for z in [0,1] and the
    ## directional derivative of half the quadratic on the segment is
    ## H(z) = dot(p1-p0,gradient[Q](L(z))/2), where gradient[Q]/2 = (F,G).
    ## By design, F(L(z)) = 0 for cases (2), (4), (5), and (6).  Cases (1) and
    ## (3) can correspond to no-intersection or intersection of F = 0 with the
    ## triangle.
    if (f00 >= 0.):
        if (f01 >= 0.):
            ## (1) p0 = (0,0), p1 = (0,1), H(z) = G(L(z))
            GetMinEdge02(a11, b1, p)
        else:
            ## (2) p0 = (0,t10), p1 = (t01,1-t01), H(z) = (t11 - t10)*G(L(z))
            p0[0] = 0.
            p0[1] = f00 / (f00 - f01)
            p1[0] = f01 / (f01 - f10)
            p1[1] = 1. - p1[0]
            dt1 = p1[1] - p0[1]
            h0 = dt1 * (a11 * p0[1] + b1)
            if (h0 >= 0.):
                GetMinEdge02(a11, b1, p)
            else:
                h1 = dt1 * (a01 * p1[0] + a11 * p1[1] + b1)
                if (h1 <= 0.):
                    GetMinEdge12(a01, a11, b1, f10, f01, p)
                else:
                    GetMinInterior(p0, h0, p1, h1, p)
    elif (f01 <= 0.):
        if (f10 <= 0.):
            ## (3) p0 = (1,0), p1 = (0,1), H(z) = G(L(z)) - F(L(z))
            GetMinEdge12(a01, a11, b1, f10, f01, p)
        else:
            ## (4) p0 = (t00,0), p1 = (t01,1-t01), H(z) = t11*G(L(z))
            p0[0] = f00 / (f00 - f10)
            p0[1] = 0.
            p1[0] = f01 / (f01 - f10)
            p1[1] = 1. - p1[0]
            h0 = p1[1] * (a01 * p0[0] + b1)

            # print h0


            if (h0 >= 0.):
                p[0] = p0[0]  ## GetMinEdge01
                p[1] = p0[1]
            else:
                h1 = p1[1] * (a01 * p1[0] + a11 * p1[1] + b1)
                if (h1 <= 0.):
                    GetMinEdge12(a01, a11, b1, f10, f01, p)
                else:
                    GetMinInterior(p0, h0, p1, h1, p)
    elif (f10 <= 0.):
        ## (5) p0 = (0,t10), p1 = (t01,1-t01), H(z) = (t11 - t10)*G(L(z))
        p0[0] = 0.
        p0[1] = f00 / (f00 - f01)
        p1[0] = f01 / (f01 - f10)
        p1[1] = 1. - p1[0]
        dt1 = p1[1] - p0[1]
        h0 = dt1 * (a11 * p0[1] + b1)
        if (h0 >= 0.):
            GetMinEdge02(a11, b1, p)
        else:
            h1 = dt1 * (a01 * p1[0] + a11 * p1[1] + b1)
            if (h1 <= 0.):
                GetMinEdge12(a01, a11, b1, f10, f01, p)
            else:
                GetMinInterior(p0, h0, p1, h1, p)
    else:
        ## (6) p0 = (t00,0), p1 = (0,t11), H(z) = t11*G(L(z))
        p0[0] = f00 / (f00 - f10)
        p0[1] = 0.
        p1[0] = 0.
        p1[1] = f00 / (f00 - f01)
        h0 = p1[1] * (a01 * p0[0] + b1)
        if (h0 >= 0.):
            p[0] = p0[0]  ## GetMinEdge01
            p[1] = p0[1]
        else:
            h1 = p1[1] * (a11 * p1[1] + b1)
            if (h1 <= 0.):
                GetMinEdge02(a11, b1, p)
            else:
                GetMinInterior(p0, h0, p1, h1, p)

    # print p
    
    cdef Result result
    result.parameter[0] = 1. - p[0] - p[1]
    result.parameter[1] = p[0]
    result.parameter[2] = p[1]

    # result.closest = triangle[0] + p[0] * edge0 + p[1] * edge1
    cdef Real tmp1[3]
    cdef Real tmp2[3]
    scale( p[0], edge0, tmp1 )
    scale( p[1], edge1, tmp2 )
    add( tmp1, tmp2, result.closest)
    add( triangle[0], result.closest, result.closest)
    

    subtract(point, result.closest, diff)
    result.sqrDistance = dot3(diff, diff)
    result.distance = sqrt(result.sqrDistance)

    return result
