'''
From: https://github.com/yig/trimesh/
'''

from numpy import *

def mag2( vec ):
    return dot( vec, vec )
def mag( vec ):
    return sqrt(mag2(vec))
def asarrayf( *args, **kwargs ):
    kwargs['dtype'] = float
    return asarray( *args, **kwargs )
def zerosf( *args, **kwargs ):
    kwargs['dtype'] = float
    return zeros( *args, **kwargs )

class TriMesh( object ):
    def __init__( self ):
        self.vs = []
        self.faces = []
        
        self.__face_normals = None
        self.__face_areas = None
        self.__vertex_normals = None
        self.__vertex_areas = None
        self.__edges = None
        
        self.__halfedges = None
        self.__vertex_halfedges = None
        self.__face_halfedges = None
        self.__edge_halfedges = None
        self.__directed_edge2he_index = None
        
        self.lifetime_counter = 0
    
    def copy( self ):
        import copy
        return copy.deepcopy( self )
    
    def __deepcopy__( self, memodict ):
        result = TriMesh()
        
        ## Make a copy of vs and faces using array().
        ## But in case they weren't stored as arrays, return them as the type they were.
        ## This is important if they were lists, and someone expected to be able to call
        ## .append() or .extend() on them.
        result.vs = array( self.vs )
        if not isinstance( self.vs, ndarray ):
            result.vs = type( self.vs )( result.vs )
        
        result.faces = array( self.faces )
        if not isinstance( self.faces, ndarray ):
            result.faces = type( self.faces )( result.faces )
        
        if hasattr( self, 'uvs' ):
            result.uvs = array( self.uvs )
            if not isinstance( self.uvs, ndarray ):
                result.uvs = type( self.uvs )( result.uvs )
        
        ## I could skip copying these cached values, but they are usually needed for rendering
        ## and copy quickly.
        if self.__face_normals is not None:
            result.__face_normals = self.__face_normals.copy()
        if self.__face_areas is not None:
            result.__face_areas = self.__face_areas.copy()
        if self.__vertex_normals is not None:
            result.__vertex_normals = self.__vertex_normals.copy()
        if self.__vertex_areas is not None:
            result.__vertex_areas = self.__vertex_areas.copy()
        if self.__edges is not None:
            result.__edges = list( self.__edges )
        
        ## I will skip copying these cached values, because they copy slowly and are
        ## not as commonly needed.  They'll still be regenerated as needed.
        '''
        if self.__halfedges is not None:
            from copy import copy
            result.__halfedges = [ copy( he ) for he in self.__halfedges ]
        if self.__vertex_halfedges is not None:
            result.__vertex_halfedges = list( self.__vertex_halfedges )
        if self.__face_halfedges is not None:
            result.__face_halfedges = list( self.__face_halfedges )
        if self.__edge_halfedges is not None:
            result.__edge_halfedges = list( self.__edge_halfedges )
        if self.__directed_edge2he_index is not None:
            result.__directed_edge2he_index = dict( self.__directed_edge2he_index )
        '''
        
        result.lifetime_counter = self.lifetime_counter
        
        return result
    
    def update_face_normals_and_areas( self ):
        if self.__face_normals is None: self.__face_normals = zerosf( ( len( self.faces ), 3 ) )
        if self.__face_areas is None: self.__face_areas = zerosf( len( self.faces ) )
        
        ## We need subtraction between vertices.
        ## Convert vertices to arrays once here, or else we'd have to call asarray()
        ## ~6 times for each vertex.
        ## NOTE: If self.vs is already an array, then this code does nothing.
        ## TODO Q: Should I set self.vs = asarray( self.vs )?  It might violate someone's
        ##         assumption that self.vs is whatever indexable type they left it.
        ##         In particular, this violates the ability of someone to .append() or .extend()
        ##         self.vs.
        vs = asarray( self.vs )
        fs = asarray( self.faces, dtype = int )
        
        ## Slow:
        '''
        for f in xrange( len( self.faces ) ):
            face = self.faces[f]
            n = cross(
                vs[ face[1] ] - vs[ face[0] ],
                vs[ face[2] ] - vs[ face[1] ]
                )
            nmag = mag( n )
            self.__face_normals[f] = (1./nmag) * n
            self.__face_areas[f] = .5 * nmag
        '''
        ## ~Slow
        
        ## Fast:
        self.__face_normals = cross( vs[ fs[:,1] ] - vs[ fs[:,0] ], vs[ fs[:,2] ] - vs[ fs[:,1] ] )
        self.__face_areas = sqrt((self.__face_normals**2).sum(axis=1))
        self.__face_normals /= self.__face_areas[:,newaxis]
        self.__face_areas *= 0.5
        ## ~Fast
        
        assert len( self.faces ) == len( self.__face_normals )
        assert len( self.faces ) == len( self.__face_areas )
    
    
    def get_face_normals( self ):
        if self.__face_normals is None: self.update_face_normals_and_areas()
        return self.__face_normals
    
    face_normals = property( get_face_normals )
    
    
    def get_face_areas( self ):
        if self.__face_areas is None: self.update_face_normals_and_areas()
        return self.__face_areas
    
    face_areas = property( get_face_areas )
    
    
    def update_vertex_normals( self ):
        if self.__vertex_normals is None: self.__vertex_normals = zerosf( ( len(self.vs), 3 ) )
        
        ## Slow:
        '''
        for vi in xrange( len( self.vs ) ):
            self.__vertex_normals[vi] = 0.
            
            for fi in self.vertex_face_neighbors( vi ):
                ## This matches the OpenMesh FAST vertex normals.
                #self.__vertex_normals[vi] += self.face_normals[ fi ]
                ## Area weighted
                self.__vertex_normals[vi] += self.face_normals[ fi ] * self.face_areas[ fi ]
        
        ## Now normalize the normals
        #self.__vertex_normals[vi] *= 1./mag( self.__vertex_normals[vi] )
        self.__vertex_normals *= 1./sqrt( ( self.__vertex_normals**2 ).sum(1) ).reshape( (len(self.vs), 1) )
        '''
        ## ~Slow
        
        ## Fast:
        fs = asarray( self.faces, dtype = int )
        ## This matches the OpenMesh FAST vertex normals.
        #fns = self.face_normals
        ## Area weighted
        fns = self.face_normals * self.face_areas[:,newaxis]
        
        self.__vertex_normals[:] = 0.
        ## I wish this worked, but it doesn't do the right thing with aliasing
        ## (when the same element appears multiple times in the slice).
        #self.__vertex_normals[ fs[:,0] ] += fns
        #self.__vertex_normals[ fs[:,1] ] += fns
        #self.__vertex_normals[ fs[:,2] ] += fns
        import itertools
        for c in (0,1,2):
            for i, n in itertools.izip( fs[:,c], fns ):
                self.__vertex_normals[ i ] += n
        
        self.__vertex_normals /= sqrt( ( self.__vertex_normals**2 ).sum(axis=1) )[:,newaxis]
        ## ~Fast
        
        assert len( self.vs ) == len( self.__vertex_normals )
    
    def get_vertex_normals( self ):
        if self.__vertex_normals is None: self.update_vertex_normals()
        return self.__vertex_normals
    
    vertex_normals = property( get_vertex_normals )
    
    
    def update_vertex_areas( self ):
        if self.__vertex_areas is None: self.__vertex_areas = zerosf( len(self.vs) )
        
        ## Slow:
        '''
        for vi in xrange( len( self.vs ) ):
            ## Try to compute proper area (if we have laplacian editing around).
            ## (This only matters for obtuse triangles.)
            try:
                #raise ImportError
                import laplacian_editing
                cot_alpha, cot_beta, area = laplacian_editing.cotangentWeights(
                    self.vs[ vi ],
                    [ self.vs[ vni ] for vni in self.vertex_vertex_neighbors( vi ) ],
                    self.vertex_is_boundary( vi )
                    )
                self.__vertex_areas[vi] = area
            
            ## Otherwise use 1/3 of the incident faces' areas
            except ImportError:
                self.__vertex_areas[vi] = 0.
                for fi in self.vertex_face_neighbors( vi ):
                    self.__vertex_areas[vi] += self.face_areas[ fi ]
                
                self.__vertex_areas[vi] *= 1./3.
        '''
        ## ~Slow
        
        ## Fast:
        ## NOTE: This does not use laplacian_editing's so-called mixed area
        ##       computation even if the module is present!
        ##       (This only matters for obtuse triangles.)
        self.__vertex_areas[:] = 0.
        
        fs = asarray( self.faces, dtype = int )
        fas = self.__face_areas
        ## I wish this worked, but it doesn't do the right thing with aliasing
        ## (when the same element appears multiple times in the slice).
        #self.__vertex_areas[ fs[:,0] ] += fas
        #self.__vertex_areas[ fs[:,1] ] += fas
        #self.__vertex_areas[ fs[:,2] ] += fas
        import itertools
        for c in (0,1,2):
            for i, area in itertools.izip( fs[:,c], fas ):
                self.__vertex_areas[ i ] += area
        
        self.__vertex_areas /= 3.
        ## ~Fast
        
        assert len( self.vs ) == len( self.__vertex_areas )
    
    def get_vertex_areas( self ):
        if self.__vertex_areas is None: self.update_vertex_areas()
        return self.__vertex_areas
    
    vertex_areas = property( get_vertex_areas )
    
    
    def update_edge_list( self ):
        #from sets import Set, ImmutableSet
        Set, ImmutableSet = set, frozenset
        
        ## We need a set of set-pairs of vertices, because edges are bidirectional.
        edges = Set()
        for face in self.faces:
            edges.add( ImmutableSet( ( face[0], face[1] ) ) )
            edges.add( ImmutableSet( ( face[1], face[2] ) ) )
            edges.add( ImmutableSet( ( face[2], face[0] ) ) )
        
        self.__edges = [ tuple( edge ) for edge in edges ]
    
    def get_edges( self ):
        if self.__edges is None: self.update_edge_list()
        return self.__edges
    
    edges = property( get_edges )
    
    
    class HalfEdge( object ):
        def __init__( self ):
            self.to_vertex = -1
            self.face = -1
            self.edge = -1
            self.opposite_he = -1
            self.next_he = -1
    
    def update_halfedges( self ):
        '''
        Generates all half edge data structures for the mesh given by its vertices 'self.vs'
        and faces 'self.faces'.
        
        untested
        '''
        
        self.__halfedges = []
        self.__vertex_halfedges = None
        self.__face_halfedges = None
        self.__edge_halfedges = None
        self.__directed_edge2he_index = {}
        
        __directed_edge2face_index = {}
        for fi, face in enumerate( self.faces ):
            __directed_edge2face_index[ (face[0], face[1]) ] = fi
            __directed_edge2face_index[ (face[1], face[2]) ] = fi
            __directed_edge2face_index[ (face[2], face[0]) ] = fi
        
        def directed_edge2face_index( edge ):
            result = __directed_edge2face_index.get( edge, -1 )
            
            ## If result is -1, then there's no such face in the mesh.
            ## The edge must be a boundary edge.
            ## In this case, the reverse orientation edge must have a face.
            if -1 == result:
                assert edge[::-1] in __directed_edge2face_index
            
            return result
        
        self.__vertex_halfedges = [None] * len( self.vs )
        self.__face_halfedges = [None] * len( self.faces )
        self.__edge_halfedges = [None] * len( self.edges )
        
        for ei, edge in enumerate( self.edges ):
            he0 = self.HalfEdge()
            ## The face will be -1 if it is a boundary half-edge.
            he0.face = directed_edge2face_index( edge )
            he0.to_vertex = edge[1]
            he0.edge = ei
            
            he1 = self.HalfEdge()
            ## The face will be -1 if it is a boundary half-edge.
            he1.face = directed_edge2face_index( edge[::-1] )
            he1.to_vertex = edge[0]
            he1.edge = ei
            
            ## Add the HalfEdge structures to the list.
            he0index = len( self.__halfedges )
            self.__halfedges.append( he0 )
            he1index = len( self.__halfedges )
            self.__halfedges.append( he1 )
            
            ## Now we can store the opposite half-edge index.
            he0.opposite_he = he1index
            he1.opposite_he = he0index
            
            ## Also store the index in our __directed_edge2he_index map.
            assert edge not in self.__directed_edge2he_index
            assert edge[::-1] not in self.__directed_edge2he_index
            self.__directed_edge2he_index[ edge ] = he0index
            self.__directed_edge2he_index[ edge[::-1] ] = he1index
            
            ## If the vertex pointed to by a half-edge doesn't yet have an out-going
            ## halfedge, store the opposite halfedge.
            ## Also, if the vertex is a boundary vertex, make sure its
            ## out-going halfedge a boundary halfedge.
            ## NOTE: Halfedge data structure can't properly handle butterfly vertices.
            ##       If the mesh has butterfly vertices, there will be multiple outgoing
            ##       boundary halfedges.  Because we have to pick one as the vertex's outgoing
            ##       halfedge, we can't iterate over all neighbors, only a single wing of the
            ##       butterfly.
            if self.__vertex_halfedges[ he0.to_vertex ] is None or -1 == he1.face:
                self.__vertex_halfedges[ he0.to_vertex ] = he0.opposite_he
            if self.__vertex_halfedges[ he1.to_vertex ] is None or -1 == he0.face:
                self.__vertex_halfedges[ he1.to_vertex ] = he1.opposite_he
            
            ## If the face pointed to by a half-edge doesn't yet have a
            ## halfedge pointing to it, store the halfedge.
            if -1 != he0.face and self.__face_halfedges[ he0.face ] is None:
                self.__face_halfedges[ he0.face ] = he0index
            if -1 != he1.face and self.__face_halfedges[ he1.face ] is None:
                self.__face_halfedges[ he1.face ] = he1index
            
            ## Store one of the half-edges for the edge.
            assert self.__edge_halfedges[ ei ] is None
            self.__edge_halfedges[ ei ] = he0index
        
        ## Now that all the half-edges are created, set the remaining next_he field.
        ## We can't yet handle boundary halfedges, so store them for later.
        boundary_heis = []
        for hei, he in enumerate( self.__halfedges ):
            ## Store boundary halfedges for later.
            if -1 == he.face:
                boundary_heis.append( hei )
                continue
            
            face = self.faces[ he.face ]
            i = he.to_vertex
            j = face[ ( list(face).index( i ) + 1 ) % 3 ]
            
            he.next_he = self.__directed_edge2he_index[ (i,j) ]
        
        ## Make a map from vertices to boundary halfedges (indices) originating from them.
        ## NOTE: There will only be multiple originating boundary halfedges at butterfly vertices.
        vertex2outgoing_boundary_hei = {}
        #from sets import Set
        Set = set
        for hei in boundary_heis:
            originating_vertex = self.__halfedges[ self.__halfedges[ hei ].opposite_he ].to_vertex
            vertex2outgoing_boundary_hei.setdefault(
                originating_vertex, Set()
                ).add( hei )
            if len( vertex2outgoing_boundary_hei[ originating_vertex ] ) > 1:
                print 'Butterfly vertex encountered'
        
        ## For each boundary halfedge, make its next_he one of the boundary halfedges
        ## originating at its to_vertex.
        for hei in boundary_heis:
            he = self.__halfedges[ hei ]
            for outgoing_hei in vertex2outgoing_boundary_hei[ he.to_vertex ]:
                he.next_he = outgoing_hei
                vertex2outgoing_boundary_hei[ he.to_vertex ].remove( outgoing_hei )
                break
        
        assert False not in [ 0 == len( out_heis ) for out_heis in vertex2outgoing_boundary_hei.itervalues() ]
    
    def he_index2directed_edge( self, he_index ):
        '''
        Given the index of a HalfEdge, returns the corresponding directed edge (i,j).
        
        untested
        '''
        
        he = self.halfedges[ he_index ]
        return ( self.halfedges[ he.opposite_he ].to_vertex, he.to_vertex )
    
    def directed_edge2he_index( self, edge ):
        '''
        Given a directed edge (i,j), returns the index of the HalfEdge class in
        halfedges().
        
        untested
        '''
        
        if self.__directed_edge2he_index is None: self.update_halfedges()
        
        edge = tuple( edge )
        return self.__directed_edge2he_index[ edge ]
    
    def get_halfedges( self ):
        '''
        Returns a list of all HalfEdge classes.
        
        untested
        '''
        
        if self.__halfedges is None: self.update_halfedges()
        return self.__halfedges
    
    halfedges = property( get_halfedges )
    
    def vertex_vertex_neighbors( self, vertex_index ):
        '''
        Returns the vertex neighbors (as indices) of the vertex 'vertex_index'.
        
        untested
        '''
        
        ## It's important to access self.halfedges first (which calls get_halfedges()),
        ## so that we're sure all halfedge info is generated.
        halfedges = self.halfedges
        result = []
        start_he = halfedges[ self.__vertex_halfedges[ vertex_index ] ]
        he = start_he
        while True:
            result.append( he.to_vertex )
            
            he = halfedges[ halfedges[ he.opposite_he ].next_he ]
            if he is start_he: break
        
        return result
    
    def vertex_valence( self, vertex_index ):
        '''
        Returns the valence (number of vertex neighbors) of vertex with index 'vertex_index'.
        
        untested
        '''
        
        return len( self.vertex_vertex_neighbors( vertex_index ) )
    
    def vertex_face_neighbors( self, vertex_index ):
        '''
        Returns the face neighbors (as indices) of the vertex 'vertex_index'.
        
        untested
        '''
        
        ## It's important to access self.halfedges first (which calls get_halfedges()),
        ## so that we're sure all halfedge info is generated.
        halfedges = self.halfedges
        result = []
        start_he = halfedges[ self.__vertex_halfedges[ vertex_index ] ]
        he = start_he
        while True:
            if -1 != he.face: result.append( he.face )
            
            he = halfedges[ halfedges[ he.opposite_he ].next_he ]
            if he is start_he: break
        
        return result
    
    def vertex_is_boundary( self, vertex_index ):
        '''
        Returns whether the vertex with given index is on the boundary.
        
        untested
        '''
        
        ## It's important to access self.halfedges first (which calls get_halfedges()),
        ## so that we're sure all halfedge info is generated.
        halfedges = self.halfedges
        return -1 == halfedges[ self.__vertex_halfedges[ vertex_index ] ].face
    
    def boundary_vertices( self ):
        '''
        Returns a list of the vertex indices on the boundary.
        
        untested
        '''
        
        result = []
        for hei, he in enumerate( self.halfedges ):
            if -1 == he.face:
                # result.extend( self.he_index2directed_edge( hei ) )
                result.append( he.to_vertex )
                result.append( self.halfedges[ he.opposite_he ].to_vertex )
        
        #from sets import ImmutableSet
        ImmutableSet = frozenset
        return list(ImmutableSet( result ))
    
    def boundary_edges( self ):
        '''
        Returns a list of boundary edges (i,j).  If (i,j) is in the result, (j,i) will not be.
        
        untested
        '''
        
        result = []
        for hei, he in enumerate( self.halfedges ):
            if -1 == he.face:
                result.append( self.he_index2directed_edge( hei ) )
        return result
    
    def positions_changed( self ):
        '''
        Notify the object that vertex positions changed.
        All position-related structures (normals, areas) will be marked for re-calculation.
        '''
        
        self.__face_normals = None
        self.__face_areas = None
        self.__vertex_normals = None
        self.__vertex_areas = None
        
        self.lifetime_counter += 1
    
    
    def topology_changed( self ):
        '''
        Notify the object that topology (faces or #vertices) changed.
        All topology-related structures (halfedges, edge lists) as well as position-related
        structures (normals, areas) will be marked for re-calculation.
        '''
        
        ## Set mesh.vs to an array so that subsequent calls to asarray() on it are no-ops.
        self.vs = asarray( self.vs )

        #### jianchao's modification begin
        self.vs=list(self.vs)
        #### jianchao's modification end
        
        self.__edges = None
        self.__halfedges = None
        self.__vertex_halfedges = None
        self.__face_halfedges = None
        self.__edge_halfedges = None
        self.__directed_edge2he_index = None
        
        self.positions_changed()
    
    def get_dangling_vertices( self ):
        '''
        Returns vertex indices in TriMesh 'mesh' that belong to no faces.
        '''
        
        ## Slow:
        '''
        brute_vertex_face_valence = [ 0 ] * len( self.vs )
        for i,j,k in self.faces:
            brute_vertex_face_valence[ i ] += 1
            brute_vertex_face_valence[ j ] += 1
            brute_vertex_face_valence[ k ] += 1
        return [ i for i in xrange( len( self.vs ) ) if 0 == brute_vertex_face_valence[i] ]
        '''
        ## ~Slow
        
        ## Fast:
        '''
        brute_vertex_face_valence = zeros( len( self.vs ), dtype = int )
        self.faces = asarray( self.faces )
        brute_vertex_face_valence[ self.faces[:,0] ] += 1
        brute_vertex_face_valence[ self.faces[:,1] ] += 1
        brute_vertex_face_valence[ self.faces[:,2] ] += 1
        return where( brute_vertex_face_valence == 0 )[0]
        '''
        ## ~Fast
        
        ## Faster:
        vertex_has_face = zeros( len( self.vs ), dtype = bool )
        self.faces = asarray( self.faces )
        vertex_has_face[ self.faces.ravel() ] = True
        return where( vertex_has_face == 0 )[0]
        ## ~Faster
    
    def remove_vertex_indices( self, vertex_indices_to_remove ):
        '''
        Removes vertices in the list of indices 'vertex_indices_to_remove'.
        Also removes faces containing the vertices and dangling vertices.
        
        Returns an array mapping vertex indices before the call
        to vertex indices after the call or -1 if the vertex was removed.
        
        used
        '''
        
        ## I can't assert this here because I call this function recursively to remove dangling
        ## vertices.
        ## Also, someone manipulating the mesh might want to do the same thing (call this
        ## function on dangling vertices).
        #assert 0 == len( self.get_dangling_vertices() )
        
        
        if 0 == len( vertex_indices_to_remove ): return arange( len( self.vs ) )
        
        
        ## Slow:
        '''
        ## Make a map from old to new vertices.  This is the return value.
        old2new = [ -1 ] * len( self.vs )
        last_index = 0
        for i in xrange( len( self.vs ) ):
            if i not in vertex_indices_to_remove:
                old2new[ i ] = last_index
                last_index += 1
        
        ## Remove vertices from vs, faces, edges, and optionally uvs.
        self.vs = [ pt for i, pt in enumerate( self.vs ) if old2new[i] != -1 ]
        if hasattr( self, 'uvs' ):
            self.uvs = [ uv for i, uv in enumerate( self.uvs ) if old2new[i] != -1 ]
        ## UPDATE: We have half-edge info, so we have to call 'topology_changed()' to
        ##         regenerate the half-edge info, and 'topology_changed()' implies
        ##         'geometry_changed()', so updating anything but '.vs', '.faces'
        ##         and '.uvs' is a waste unless I can precisely update the
        ##         halfedge data structures.
        #self.__vertex_normals = asarrayf( [ vn for i, vn in enumerate( self.__vertex_normals ) if old2new[i] != -1 ] )
        #self.__edges = [ ( old2new[i], old2new[j] ) for i,j in self.__edges ]
        #self.__edges = [ edge for edge in self.__edges if -1 not in edge ]
        self.faces = [ ( old2new[i], old2new[j], old2new[k] ) for i,j,k in self.faces ]
        #self.__face_normals = [ n for i,n in enumerate( self.__face_normals ) if -1 not in self.faces[i] ]
        #self.__face_areas = [ n for i,n in enumerate( self.__face_areas ) if -1 not in self.faces[i] ]
        self.faces = [ tri for tri in self.faces if -1 not in tri ]
        '''
        ## ~Slow
        
        
        ## Fast:
        ## Make a map from old to new vertices.  This is the return value.
        old2new = -ones( len( self.vs ), dtype = int )
        ## Later versions of numpy.setdiff1d(), such as 2.0, return a unique, sorted array
        ## and do not assume that inputs are unique.
        ## Earlier versions, such as 1.4, require unique inputs and don't say
        ## anything about sorted output.
        ## (We don't know that 'vertex_indices_to_remove' is unique!)
        keep_vertices = sort( setdiff1d( arange( len( self.vs ) ), unique( vertex_indices_to_remove ) ) )
        old2new[ keep_vertices ] = arange( len( keep_vertices ) )
        
        ## Remove vertices from vs, faces, edges, and optionally uvs.
        ## Fast:
        self.vs = asarray( self.vs )
        self.vs = self.vs[ keep_vertices, : ]
        if hasattr( self, 'uvs' ):
            self.uvs = asarray( self.uvs )
            self.uvs = self.uvs[ keep_vertices, : ]
        
        self.faces = asarray( self.faces )
        self.faces = old2new[ self.faces ]
        self.faces = self.faces[ ( self.faces != -1 ).all( axis = 1 ) ]
        ## ~Fast
        
        
        ## Now that we have halfedge info, just call topology changed and everything but
        ## 'vs' and 'faces' will be regenerated.
        self.topology_changed()
        
        ## Remove dangling vertices created by removing faces incident to vertices in 'vertex_indices_to_remove'.
        ## We only need to call this once, because a dangling vertex has no faces, so its removal
        ## won't remove any faces, so no new dangling vertices can be created.
        dangling = self.get_dangling_vertices()
        if len( dangling ) > 0:
            old2new_recurse = self.remove_vertex_indices( dangling )
            assert 0 == len( self.get_dangling_vertices() )
            
            
            for i in xrange( len( old2new ) ):
                if -1 != old2new[i]: old2new[i] = old2new_recurse[ old2new[ i ] ]
            
            # old2new[ old2new != -1 ] = old2new_recurse[ old2new ]
        
        
        ### jianchao's modification begin
        self.vs=list(self.vs)
        self.faces=list(self.faces)
        ### jianchao's modification end


        return old2new
    
    def remove_face_indices( self, face_indices_to_remove ):
        '''
        Removes faces in the list of indices 'face_indices_to_remove'.
        Also removes dangling vertices.
        
        Returns an array mapping face indices before the call
        to face indices after the call or -1 if the face was removed.
        
        used
        '''
        
        if 0 == len( face_indices_to_remove ): return arange( len( self.faces ) )
        
        
        ## Fast:
        ## Make a map from old to new faces.  This is the return value.
        old2new = -ones( len( self.faces ), dtype = int )
        ## Later versions of numpy.setdiff1d(), such as 2.0, return a unique, sorted array
        ## and do not assume that inputs are unique.
        ## Earlier versions, such as 1.4, require unique inputs and don't say
        ## anything about sorted output.
        ## (We don't know that 'face_indices_to_remove' is unique!)
        keep_faces = sort( setdiff1d( arange( len( self.faces ) ), unique( face_indices_to_remove ) ) )
        old2new[ keep_faces ] = arange( len( keep_faces ) )
        
        ## Remove vertices from vs, faces, edges, and optionally uvs.
        ## Fast:
        self.faces = asarray( self.faces )
        self.faces = self.faces[ keep_faces, : ]
        ## ~Fast
        
        
        ## Now that we have halfedge info, just call topology changed and everything but
        ## 'vs' and 'faces' will be regenerated.
        self.topology_changed()
        
        ## Remove dangling vertices created by removing faces incident to vertices.
        ## Since we are only removing dangling vertices, 'self.faces' can't be affected,
        ## so we don't need to worry about the 'old2new' map.
        dangling = self.get_dangling_vertices()
        if len( dangling ) > 0:
            self.remove_vertex_indices( dangling )
            assert 0 == len( self.get_dangling_vertices() )
        
        return old2new
    
    
    def append( self, mesh ):
        '''
        Given a mesh, with two properties,
            .vs, containing a list of 3d vertices
            .faces, containing a list of triangles as triplets of indices into .vs
        appends 'mesh's vertices and faces to self.vs and self.faces.
        '''
        
        ## mesh's vertices are going to be copied to the end of self.vs;
        ## All vertex indices in mesh.faces will need to be offset by the current
        ## number of vertices in self.vs.
        vertex_offset = len( self.vs )
        
        self.vs = list( self.vs ) + list( mesh.vs )
        self.faces = list( self.faces ) + list( asarray( mesh.faces, dtype = int ) + vertex_offset )
        
        
        ## If there are uvs, concatenate them.
        
        ## First, if self is an empty mesh (without uv's), and the mesh to append-to has uv's,
        ## create an empty .uvs property in self.
        if not hasattr( self, 'uvs' ) and hasattr( mesh, 'uvs' ) and len( self.vs ) == 0:
            self.uvs = []
        
        if hasattr( self, 'uvs' ) and hasattr( mesh, 'uvs' ):
            self.uvs = list( self.uvs ) + list( mesh.uvs )
        elif hasattr( self, 'uvs' ):
            del self.uvs
        
        
        ## We're almost done, we only need to call topology_changed().
        ## However, let's see if we can keep some properties that are slow to regenerate.
        self__face_normals = self.__face_normals
        self__face_areas = self.__face_areas
        self__vertex_normals = self.__vertex_normals
        self__vertex_areas = self.__vertex_areas
        
        self.topology_changed()
        
        if self__face_normals is not None and mesh.__face_normals is not None:
            self.__face_normals = append( self__face_normals, mesh.__face_normals, axis = 0 )
        if self__face_areas is not None and mesh.__face_areas is not None:
            self.__face_areas = append( self__face_areas, mesh.__face_areas, axis = 0 )
        if self__vertex_normals is not None and mesh.__vertex_normals is not None:
            self.__vertex_normals = append( self__vertex_normals, mesh.__vertex_normals, axis = 0 )
        if self__vertex_areas is not None and mesh.__vertex_areas is not None:
            self.__vertex_areas = append( self__vertex_areas, mesh.__vertex_areas, axis = 0 )
    
    
    def FromTriMeshes( meshes ):
        '''
        Given a sequence of meshes, each with two properties,
            .vs, containing a list of 3d vertices
            .faces, containing a list of triangles as triplets of indices into .vs
        returns a single TriMesh object containing all meshes concatenated together.
        '''
        
        result = TriMesh()
        for mesh in meshes:
            result.append( mesh )
        
        ## Reset the lifetime counter
        result.lifetime_counter = 0
        return result
    
    FromTriMeshes = staticmethod( FromTriMeshes )
    
    
    def FromOBJ_FileName( obj_fname ):
        if obj_fname.endswith( '.gz' ):
            import gzip
            f = gzip.open( obj_fname )
        else:
            f = open( obj_fname )
        return TriMesh.FromOBJ_Lines( f )
    
    FromOBJ_FileName = staticmethod( FromOBJ_FileName )
    
    
    def FromOBJ_Lines( obj_lines ):
        '''
        Given lines from an OBJ file, return a new TriMesh object.
        
        tested
        '''
        
        result = TriMesh()
        
        ## NOTE: We only handle faces and vertex positions.
        for line in obj_lines:
            line = line.strip()
            
            sline = line.split()
            ## Skip blank lines
            if not sline: continue
            
            elif sline[0] == 'v':
                result.vs.append( asarrayf( map( float, sline[1:] ) ) )
                ## Vertices must have three coordinates.
                assert len( result.vs[-1] ) == 3
            
            elif sline[0] == 'f':
                ## The split('/')[0] means we record only the vertex coordinate indices
                ## for each face.
                face_vertex_ids = [ int( c.split('/')[0] ) for c in sline[1:] ]
                ## Faces must be triangles.
                assert len( face_vertex_ids ) == 3
                
                ## Face vertex indices cannot be zero.
                assert True not in [ ind == 0 for ind in face_vertex_ids ]
                
                ## Subtract one from positive indices, and use relative addressing for negative
                ## indices.
                face_vertex_ids = [
                    ## This awkward "make a tuple with both results and select it based on
                    ## truth" technique is to make up for Python's lack of a functional 'if'.
                    ( ind-1, len(result.vs) + ind )[ ind < 0 ]
                    for ind in face_vertex_ids
                    ]
                
                assert False not in [ ind < len( result.vs ) for ind in face_vertex_ids ]
                result.faces.append( face_vertex_ids )
        
        return result
    
    FromOBJ_Lines = staticmethod( FromOBJ_Lines )
    
    
    def write_OBJ( self, fname, header_comment = None ):
        '''
        Writes the data out to an OBJ file named 'fname'.
        Optional comment 'header_comment' is printed at the
        top of the OBJ file, after prepending the OBJ comment
        marker at the head of each line.
        
        tested
        '''
        
        
        ## Estimate for mesh size:
        ## 16 bytes for a vertex row,
        ## optionally 16 bytes for a uv row,
        ## 12/20 bytes for a face row with/without uv's.
        ## Assuming no uv's and 2 faces per vertex,
        ## a 1MB mesh is made of (1024*1024/(16+2*12)) = 26214 vertices.
        ## If we have uv's, then we will reach 1MB with (1024*1024/(2*16+2*20)) = 14563 vertices.
        ## Print a warning if we're going to save a mesh much larger than a megabyte.
        if len( self.vs ) > 15000:
            print '[Writing a large OBJ to "%s"...]' % (fname,)
        
        
        out = file( fname, 'w' )
        
        if header_comment is None:
            import sys
            header_comment = 'Written by ' + ' '.join([ arg.replace('\n',r'\n') for arg in sys.argv ])
        
        ## Print the header comment.
        for line in header_comment.split('\n'):
            out.write( '## %s\n' % (line,) )
        out.write( '\n' )
        
        
        ## Print vertices.
        for v in self.vs:
            out.write( 'v %r %r %r\n' % tuple(v) )
        out.write( '\n' )
        
        
        ## Print uv's if we have them.
        if hasattr( self, 'uvs' ):
            for uv in self.uvs:
                out.write( 'vt %r %r\n' % tuple(uv) )
            out.write( '\n' )
            
            ## Print faces with uv's.
            for f in self.faces:
                #out.write( 'f %s/%s %s/%s %s/%s\n' % tuple( ( asarray(f,dtype=int) + 1 ).repeat(2) ) )
                out.write( 'f %s/%s %s/%s %s/%s\n' % ( f[0]+1,f[0]+1, f[1]+1,f[1]+1, f[2]+1,f[2]+1 ) )
        else:
            ## Print faces without uv's.
            for f in self.faces:
                #out.write( 'f %s %s %s\n' % tuple(asarray(f,dtype=int) + 1) )
                out.write( 'f %s %s %s\n' % ( f[0]+1, f[1]+1, f[2]+1 ) )
        
        
        out.close()
        
        print '[OBJ written to "%s"]' % (fname,)
    
    def write_OFF( self, fname ):
        '''
        Writes the data out to an OFF file named 'fname'.
        '''
        
        out = file( fname, 'w' )
        
        out.write( 'OFF\n' )
        out.write( '%d %d 0\n' % ( len( self.vs ), len( self.faces ) ) )
        
        for v in self.vs:
            out.write( '%r %r %r\n' % tuple(v) )
        for f in self.faces:
            out.write( '3 %s %s %s\n' % tuple(f) )
        
        out.close()
        
        print '[OFF written to "%s"]' % (fname,)

## We can't pickle anything that doesn't have a name visible at module scope.
## In order to allow pickling of class TriMesh, we'll make a reference to the inner HalfEdge class
## here at the module level.
HalfEdge = TriMesh.HalfEdge
