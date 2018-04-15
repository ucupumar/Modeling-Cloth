# You are at the top. If you attempt to go any higher
#   you will go beyond the known limits of the code
#   universe where there are most certainly monsters

# might be able to get a speedup where I'm appending move and -move

# to do:
#  use point raycaster to make a cloth_wrap option
#  fix multiple sew spring error
#  set up better indexing so that edges only get calculated once
#  self colisions
#  object collisions
#  add bending springs
#  add curl by shortening bending springs on one axis or diagonal
#  independantly scale bending springs and structural to create buckling
#  run on frame update as an option?
#  option to cache animation?
#  collisions need to properly exclude pinned and vertex pinned
#  virtual springs do something wierd to the velocity
#  multiple converging springs go to far. Need to divide by number of springs at a vert or move them all towards a mean

# now!!!!
#  refresh self collisions.

# collisions:
# Onlny need to check on of the edges for groups connected to a vertex    
# for edge to face intersections...
# figure out where the edge hit the face
# figure out which end of the edge is inside the face
# move along the face normal to the surface for the point inside.
# if I reflect by flipping the vel around the face normal
#   if it collides on the bounce it will get caught on the next iteration


'''??? Would it make sense to do self collisions with virtual edges ???'''

bl_info = {
    "name": "Modeling Cloth",
    "author": "Rich Colburn (the3dadvantage@gmail.com), Yusuf Umar (@ucupumar)",
    "version": (1, 0),
    "blender": (2, 79, 0),
    "location": "View3D > Extended Tools > Modeling Cloth",
    "description": "Maintains the surface area of an object so it behaves like cloth",
    "warning": "There might be an angry rhinoceros behind you",
    "wiki_url": "",
    "category": '3D View'}


import bpy
import bmesh
import numpy as np
from numpy import newaxis as nax
from bpy_extras import view3d_utils
from bpy.props import *
from bpy.app.handlers import persistent
from mathutils import *
import pickle
import codecs
import time
import sys

AUTO_UPDATE_WHEN_ERROR_HAPPENS = True

you_have_a_sense_of_humor = False
#you_have_a_sense_of_humor = True
if you_have_a_sense_of_humor:
    import antigravity


def get_co(ob, arr=None, key=None): # key
    """Returns vertex coords as N x 3"""
    c = len(ob.data.vertices)
    if arr is None:    
        arr = np.zeros(c * 3, dtype=np.float32)
    if key is not None:
        ob.data.shape_keys.key_blocks[key].data.foreach_get('co', arr.ravel())        
        arr.shape = (c, 3)
        return arr
    ob.data.vertices.foreach_get('co', arr.ravel())
    arr.shape = (c, 3)
    return arr


def get_proxy_co(ob, arr):
    """Returns vertex coords with modifier effects as N x 3"""
    me = ob.to_mesh(bpy.context.scene, True, 'PREVIEW')
    if arr is None:
        arr = np.zeros(len(me.vertices) * 3, dtype=np.float32)
        arr.shape = (arr.shape[0] //3, 3)    
    c = arr.shape[0]
    me.vertices.foreach_get('co', arr.ravel())
    bpy.data.meshes.remove(me)
    arr.shape = (c, 3)
    return arr


def triangulate(ob):
    """Requires a mesh. Returns an index array for viewing co as triangles"""
    me = ob.to_mesh(bpy.context.scene, True, 'PREVIEW')
    obm = bmesh.new()
    obm.from_mesh(me)        
    bmesh.ops.triangulate(obm, faces=obm.faces)
    obm.to_mesh(me)        
    count = len(me.polygons)    
    tri_idx = np.zeros(count * 3, dtype=np.int64)        
    me.polygons.foreach_get('vertices', tri_idx)        
    bpy.data.meshes.remove(me)
    obm.free()
    return tri_idx.reshape(count, 3)


def tri_normals_in_place(ob, col, tri_co):    
    """Takes N x 3 x 3 set of 3d triangles and 
    returns non-unit normals and origins"""
    #col = get_collider_data(ob)
    col.origins = tri_co[:,0]
    col.cross_vecs = tri_co[:,1:] - col.origins[:, nax]
    col.normals = np.cross(col.cross_vecs[:,0], col.cross_vecs[:,1])


def get_tri_normals(tr_co):
    """Takes N x 3 x 3 set of 3d triangles and 
    returns non-unit normals and origins"""
    origins = tr_co[:,0]
    cross_vecs = tr_co[:,1:] - origins[:, nax]
    return cross_vecs, np.cross(cross_vecs[:,0], cross_vecs[:,1]), origins


def closest_points_edge(vec, origin, p):
    '''Returns the location of the point on the edge'''
    vec2 = p - origin
    d = (vec2 @ vec) / (vec @ vec)
    cp = vec * d[:, nax]
    return cp, d


def proxy_in_place(ob, col):
    """Overwrite vert coords with modifiers in world space"""
    #col = get_collider_data(ob)
    me = ob.to_mesh(bpy.context.scene, True, 'PREVIEW')
    me.vertices.foreach_get('co', col.co.ravel())
    bpy.data.meshes.remove(me)
    col.co = apply_transforms(ob, col.co)


def apply_rotation(ob, col):
    """When applying vectors such as normals we only need
    to rotate"""
    #col = get_collider_data(ob)
    m = np.array(ob.matrix_world)
    mat = m[:3, :3].T
    col.v_normals = col.v_normals @ mat
    

def proxy_v_normals_in_place(ob, col, world=True):
    """Overwrite vert coords with modifiers in world space"""
    #col = get_collider_data(ob)
    me = ob.to_mesh(bpy.context.scene, True, 'PREVIEW')
    me.vertices.foreach_get('normal', col.v_normals.ravel())
    bpy.data.meshes.remove(me)
    if world:    
        apply_rotation(ob, col)


def proxy_v_normals(ob):
    """Overwrite vert coords with modifiers in world space"""
    me = ob.to_mesh(bpy.context.scene, True, 'PREVIEW')
    arr = np.zeros(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get('normal', arr)
    arr.shape = (arr.shape[0] //3, 3)
    bpy.data.meshes.remove(me)
    m = np.array(ob.matrix_world, dtype=np.float32)    
    mat = m[:3, :3].T # rotates backwards without T
    return arr @ mat


def apply_transforms(ob, co):
    """Get vert coords in world space"""
    m = np.array(ob.matrix_world, dtype=np.float32)    
    mat = m[:3, :3].T # rotates backwards without T
    loc = m[:3, 3]
    return co @ mat + loc


def apply_in_place(ob, arr, cloth):
    """Overwrite vert coords in world space"""
    m = np.array(ob.matrix_world, dtype=np.float32)    
    mat = m[:3, :3].T # rotates backwards without T
    loc = m[:3, 3]
    arr[:] = arr @ mat + loc
    #cloth.co = cloth.co @ mat + loc


def applied_key_co(ob, arr=None, key=None):
    """Get vert coords in world space"""
    c = len(ob.data.vertices)
    if arr is None:
        arr = np.zeros(c * 3, dtype=np.float32)
    ob.data.shape_keys.key_blocks[key].data.foreach_get('co', arr)
    arr.shape = (c, 3)
    m = np.array(ob.matrix_world)    
    mat = m[:3, :3].T # rotates backwards without T
    loc = m[:3, 3]
    return co @ mat + loc


def revert_transforms(ob, co):
    """Set world coords on object. 
    Run before setting coords to deal with object transforms
    if using apply_transforms()"""
    m = np.linalg.inv(ob.matrix_world)    
    mat = m[:3, :3].T # rotates backwards without T
    loc = m[:3, 3]
    return co @ mat + loc  


def revert_in_place(ob, co):
    """Revert world coords to object coords in place."""
    m = np.linalg.inv(ob.matrix_world)    
    mat = m[:3, :3].T # rotates backwards without T
    loc = m[:3, 3]
    co[:] = co @ mat + loc


def revert_rotation(ob, co):
    """When reverting vectors such as normals we only need
    to rotate"""
    #m = np.linalg.inv(ob.matrix_world)    
    m = np.array(ob.matrix_world)
    mat = m[:3, :3] # rotates backwards without T
    return co @ mat


def get_last_object():
    """Finds cloth objects for keeping settings active
    while selecting other objects like pins"""
    cloths = [i for i in bpy.data.objects if i.mc.enable] # so we can select an empty and keep the settings menu up
    if bpy.context.object.mc.enable:
        return cloths, bpy.context.object
    
    if len(cloths) > 0:
        ob = bpy.context.scene.mc.last_object
        return cloths, ob
    return None, None


def get_poly_centers(ob, type=np.float32):
    mod = False
    m_count = len(ob.modifiers)
    if m_count > 0:
        show = np.zeros(m_count, dtype=np.bool)
        ren_set = np.copy(show)
        ob.modifiers.foreach_get('show_render', show)
        ob.modifiers.foreach_set('show_render', ren_set)
        mod = True
    mesh = ob.to_mesh(bpy.context.scene, True, 'RENDER')
    p_count = len(mesh.polygons)
    center = np.zeros(p_count * 3)#, dtype=type)
    mesh.polygons.foreach_get('center', center)
    center.shape = (p_count, 3)
    bpy.data.meshes.remove(mesh)
    if mod:
        ob.modifiers.foreach_set('show_render', show)

    return center


def simple_poly_centers(ob, key=None):
    if key is not None:
        s_key = ob.data.shape_keys.key_blocks[key].data
        return np.squeeze([[np.mean([ob.data.vertices[i].co for i in p.vertices], axis=0)] for p in ob.data.polygons])


def get_poly_normals(ob, type=np.float32):
    mod = False
    m_count = len(ob.modifiers)
    if m_count > 0:
        show = np.zeros(m_count, dtype=np.bool)
        ren_set = np.copy(show)
        ob.modifiers.foreach_get('show_render', show)
        ob.modifiers.foreach_set('show_render', ren_set)
        mod = True
    mesh = ob.to_mesh(bpy.context.scene, True, 'RENDER')
    p_count = len(mesh.polygons)
    normal = np.zeros(p_count * 3)#, dtype=type)
    mesh.polygons.foreach_get('normal', normal)
    normal.shape = (p_count, 3)
    bpy.data.meshes.remove(mesh)
    if mod:
        ob.modifiers.foreach_set('show_render', show)

    return normal


def get_v_normals(ob, arr):
    """Since we're reading from a shape key we have to use
    a proxy mesh."""
    mod = False
    m_count = len(ob.modifiers)
    if m_count > 0:
        show = np.zeros(m_count, dtype=np.bool)
        ren_set = np.copy(show)
        ob.modifiers.foreach_get('show_render', show)
        ob.modifiers.foreach_set('show_render', ren_set)
        mod = True
    mesh = ob.to_mesh(bpy.context.scene, True, 'RENDER')
    #v_count = len(mesh.vertices)
    #normal = np.zeros(v_count * 3)#, dtype=type)
    mesh.vertices.foreach_get('normal', arr.ravel())
    #normal.shape = (v_count, 3)
    bpy.data.meshes.remove(mesh)
    if mod:
        ob.modifiers.foreach_set('show_render', show)

    #return normal


def get_v_nor(ob, nor_arr):
    ob.data.vertices.foreach_get('normal', nor_arr.ravel())
    return nor_arr


def closest_point_edge(e1, e2, p):
    '''Returns the location of the point on the edge'''
    vec1 = e2 - e1
    vec2 = p - e1
    d = np.dot(vec2, vec1) / np.dot(vec1, vec1)
    cp = e1 + vec1 * d 
    return cp


def generate_collision_data(ob, pins, means):
    """The mean of each face is treated as a point then
    checked against the cpoe of the normals of every face.
    If the distance between the cpoe and the center of the face
    it's checking is in the margin, do a second check to see
    if it's also in the virtual cylinder around the normal.
    If it's in the cylinder, back up along the normal until the
    point is on the surface of the cylinder.
    instead of backing up along the normal, could do a line-plane intersect
    on the path of the point against the normal. 
    
    Since I know the direction the points are moving... it might be possible
    to identify the back sides of the cylinders so the margin on the back
    side is infinite. This way I could never miss the collision.
    !!! infinite backsides !!! Instead... since that would work because
    there is no way to tell the difference between a point that's moving
    away from the surface and a point that's already crossed the surface...
    I could do a raycast onto the infinte plane of the normal still treating
    it like a cylinder so instead of all the inside triangle stuff, just check
    the distance from the intersection to the cpoe of the normal
    
    Get cylinder sizes by taking the center of each face,
    and measuring the distance to it's closest neighbor, then back up a bit
    
    !!!could save the squared distance around the cylinders and along the normal 
    to save a step while checking... !!!
    """
    
    # one issue: oddly shaped triangles can cause gaps where face centers could pass through
    # since both sids are being checked it's less likely that both sides line up and pass through the gap
    obm = bmesh.new()
    obm.from_mesh(ob.data)

    obm.faces.ensure_lookup_table()
    obm.verts.ensure_lookup_table()
    p_count = len(obm.faces)
    
    per_face_v =  [[v.co for v in f.verts] for f in obm.faces]
    
    # $$$$$ calculate means with add.at like it's set up already$$$$$
    #means = np.array([np.mean([v.co for v in f.verts], axis=0) for f in obm.faces], dtype=np.float32)
    ### !!! calculating means from below wich is dynamic. Might be better since faces change size anyway. Could get better collisions

    # get sqared distance to closest vert in each face. (this will still work if the mesh isn't flat)
    sq_dist = []
    for i in range(p_count):
        dif = np.array(per_face_v[i]) - means[i]
        sq_dist.append(np.min(np.einsum('ij,ij->i', dif, dif)))
    
    # neighbors for excluding point face collisions.
    neighbors = np.tile(np.ones(p_count, dtype=np.bool), (pins.shape[0], 1))
    #neighbors = np.tile(pins, (pins.shape[0], 1))
    p_neighbors = [[f.index for f in obm.verts[p].link_faces] for p in pins]
    for x in range(neighbors.shape[0]):
        neighbors[x][p_neighbors[x]] = False

    # returns the radius distance from the mean to the closest vert in the polygon
    return np.array(np.sqrt(sq_dist), dtype=np.float32), neighbors
        

def create_vertex_groups(groups=['common', 'not_used'], weights=[0.0, 0.0], ob=None):
    '''Creates vertex groups and sets weights. "groups" is a list of strings
    for the names of the groups. "weights" is a list of weights corresponding 
    to the strings. Each vertex is assigned a weight for each vertex group to
    avoid calling vertex weights that are not assigned. If the groups are
    already present, the previous weights will be preserved. To reset weights
    delete the created groups'''
    if ob is None:
        ob = bpy.context.object
    vg = ob.vertex_groups
    for g in range(0, len(groups)):
        if groups[g] not in vg.keys(): # Don't create groups if there are already there
            vg.new(groups[g])
            vg[groups[g]].add(range(0,len(ob.data.vertices)), weights[g], 'REPLACE')
        else:
            vg[groups[g]].add(range(0,len(ob.data.vertices)), 0, 'ADD') # This way we avoid resetting the weights for existing groups.


def get_bmesh(obj=None):
    ob = get_last_object()[1]
    if ob is None:
        ob = obj
    obm = bmesh.new()
    if ob.mode == 'OBJECT':
        obm.from_mesh(ob.data)
    elif ob.mode == 'EDIT':
        obm = bmesh.from_edit_mesh(ob.data)
    return obm


def get_minimal_edges(ob):
    obm = get_bmesh(ob)
    obm.edges.ensure_lookup_table()
    obm.verts.ensure_lookup_table()
    obm.faces.ensure_lookup_table()
    
    # get sew edges:
    sew = [i.index for i in obm.edges if len(i.link_faces)==0]
    
    # get linear edges
    e_count = len(obm.edges)
    eidx = np.zeros(e_count * 2, dtype=np.int32)
    e_bool = np.zeros(e_count, dtype=np.bool)
    e_bool[sew] = True
    ob.data.edges.foreach_get('vertices', eidx)
    eidx.shape = (e_count, 2)

    # get diagonal edges:
    diag_eidx = []
    print('new============')
    start = 0
    stop = 0
    step_size = [len(i.verts) for i in obm.faces]
    p_v_count = np.sum(step_size)
    p_verts = np.ones(p_v_count, dtype=np.int32)
    ob.data.polygons.foreach_get('vertices', p_verts)
    # can only be understood on a good day when the coffee flows (uses rolling and slicing)
    # creates uniqe diagonal edge sets
    for f in obm.faces:
        fv_count = len(f.verts)
        stop += fv_count
        if fv_count > 3: # triangles are already connected by linear springs
            skip = 2
            f_verts = p_verts[start:stop]
            for fv in range(len(f_verts)):
                if fv > 1:        # as we go around the loop of verts in face we start overlapping
                    skip = fv + 1 # this lets us skip the overlap so we don't have mirror duplicates
                roller = np.roll(f_verts, fv)
                for r in roller[skip:-1]:
                    diag_eidx.append([roller[0], r])

        start += fv_count    
    
    # eidx groups
    sew_eidx = eidx[e_bool]
    lin_eidx = eidx[-e_bool]
    diag_eidx = np.array(diag_eidx)
        
    return lin_eidx, diag_eidx, sew_eidx


def add_remove_virtual_springs(remove=False):
    ob = get_last_object()[1]
    cloth = get_cloth_data(ob)
    obm = get_bmesh()
    obm.verts.ensure_lookup_table()
    count = len(obm.verts)
    idxer = np.arange(count, dtype=np.int32)
    sel = np.array([v.select for v in obm.verts])    
    selected = idxer[sel]

    virtual_springs = np.array([[vs.vertex_id_1, vs.vertex_id_2] for vs in ob.mc.virtual_springs])
    if virtual_springs.shape[0] == 0:
        virtual_springs.shape = (0, 2)

    if remove:
        ls = virtual_springs[:, 0]
        
        in_sel = np.in1d(ls, idxer[sel])

        deleter = np.arange(ls.shape[0], dtype=np.int32)[in_sel]

        for i in reversed(deleter):
            ob.mc.virtual_springs.remove(i)

        #ob.mc.force_update = True
        return

    existing = np.append(cloth.eidx, virtual_springs, axis=0)
    flip = existing[:, ::-1]
    existing = np.append(existing, flip, axis=0)
    ls = existing[:,0]
        
    #springs = []
    for i in idxer[sel]:

        # to avoid duplicates:
        # where this vert occurs on the left side of the existing spring list
        v_in = existing[i == ls]
        v_in_r = v_in[:,1]
        not_in = selected[~np.in1d(selected, v_in_r)]
        idx_set = not_in[not_in != i]
        for sv in idx_set:
            #springs.append([i, sv])
            new_vs = ob.mc.virtual_springs.add()
            new_vs.vertex_id_1 = i
            new_vs.vertex_id_2 = sv

    # gets appended to eidx in the cloth_init function after calling get connected polys in case geometry changes
    #ob.mc.force_update = True


def generate_guide_mesh():
    """Makes the arrow that appears when creating pins"""
    verts = [[0.0, 0.0, 0.0], [-0.01, -0.01, 0.1], [-0.01, 0.01, 0.1], [0.01, -0.01, 0.1], [0.01, 0.01, 0.1], [-0.03, -0.03, 0.1], [-0.03, 0.03, 0.1], [0.03, 0.03, 0.1], [0.03, -0.03, 0.1], [-0.01, -0.01, 0.2], [-0.01, 0.01, 0.2], [0.01, -0.01, 0.2], [0.01, 0.01, 0.2]]
    edges = [[0, 5], [5, 6], [6, 7], [7, 8], [8, 5], [1, 2], [2, 4], [4, 3], [3, 1], [5, 1], [2, 6], [4, 7], [3, 8], [9, 10], [10, 12], [12, 11], [11, 9], [3, 11], [9, 1], [2, 10], [12, 4], [6, 0], [7, 0], [8, 0]]
    faces = [[0, 5, 6], [0, 6, 7], [0, 7, 8], [0, 8, 5], [1, 3, 11, 9], [1, 2, 6, 5], [2, 4, 7, 6], [4, 3, 8, 7], [3, 1, 5, 8], [12, 10, 9, 11], [4, 2, 10, 12], [3, 4, 12, 11], [2, 1, 9, 10]]
    name = 'ModelingClothPinGuide'
    if 'ModelingClothPinGuide' in bpy.data.objects:
        mesh_ob = bpy.data.objects['ModelingClothPinGuide']
    else:   
        mesh = bpy.data.meshes.new('ModelingClothPinGuide')
        mesh.from_pydata(verts, edges, faces)  
        mesh.update()
        mesh_ob = bpy.data.objects.new(name, mesh)
        bpy.context.scene.objects.link(mesh_ob)
        mesh_ob.show_x_ray = True
    return mesh_ob


def create_guide():
    """Spawns the guide"""
    if 'ModelingClothPinGuide' in bpy.data.objects:
        mesh_ob = bpy.data.objects['ModelingClothPinGuide']
        return mesh_ob
    mesh_ob = generate_guide_mesh()
    bpy.context.scene.objects.active = mesh_ob
    bpy.ops.object.material_slot_add()
    if 'ModelingClothPinGuide' in bpy.data.materials:
        mat = bpy.data.materials['ModelingClothPinGuide']
    else:    
        mat = bpy.data.materials.new(name='ModelingClothPinGuide')
    mat.use_transparency = True
    mat.alpha = 0.35            
    mat.emit = 2     
    mat.game_settings.alpha_blend = 'ALPHA_ANTIALIASING'
    mat.diffuse_color = (1, 1, 0)
    mesh_ob.material_slots[0].material = mat
    return mesh_ob


def delete_guide():
    """Deletes the arrow"""
    if 'ModelingClothPinGuide' in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects['ModelingClothPinGuide'])
    if 'ModelingClothPinGuide' in bpy.data.meshes:        
        guide_mesh = bpy.data.meshes['ModelingClothPinGuide']
        guide_mesh.user_clear()
        bpy.data.meshes.remove(guide_mesh)
    

def scale_source(multiplier):
    """grow or shrink the source shape"""
    ob = get_last_object()[1]
    if ob is not None:
        if ob.mc.enable:
            count = len(ob.data.vertices)
            co = np.zeros(count*3, dtype=np.float32)
            ob.data.shape_keys.key_blocks['modeling cloth source key'].data.foreach_get('co', co)
            co.shape = (count, 3)
            mean = np.mean(co, axis=0)
            co -= mean
            co *= multiplier
            co += mean
            ob.data.shape_keys.key_blocks['modeling cloth source key'].data.foreach_set('co', co.ravel())                
            cloth = get_cloth_data(ob)
            if hasattr(cloth, 'cy_dists'):
                cloth.cy_dists *= multiplier
            

def reset_shapes(ob=None):
    """Sets the modeling cloth key to match the source key.
    Will regenerate shape keys if they are missing"""
    if ob is None:    
        if bpy.context.object.mc.enable:
            ob = bpy.context.object
        else:    
            ob = bpy.context.scene.mc.last_object

    if ob.data.shape_keys == None:
        ob.shape_key_add('Basis')    
    if 'modeling cloth source key' not in ob.data.shape_keys.key_blocks:
        ob.shape_key_add('modeling cloth source key')        
    if 'modeling cloth key' not in ob.data.shape_keys.key_blocks:
        ob.shape_key_add('modeling cloth key')        
        ob.data.shape_keys.key_blocks['modeling cloth key'].value=1
    
    keys = ob.data.shape_keys.key_blocks
    count = len(ob.data.vertices)
    co = np.zeros(count * 3, dtype=np.float32)
    keys['modeling cloth source key'].data.foreach_get('co', co)
    #co = applied_key_co(ob, None, 'modeling cloth source key')
    keys['modeling cloth key'].data.foreach_set('co', co)
    
    # reset the data stored in the class
    cloth = get_cloth_data(ob)
    cloth.vel[:] = 0
    co.shape = (co.shape[0]//3, 3)
    cloth.co = co
    
    keys['modeling cloth key'].mute = True
    keys['modeling cloth key'].mute = False


def get_spring_mix(ob, eidx):
    rs = []
    ls = []
    minrl = []
    for i in eidx:
        r = eidx[eidx == i[1]].shape[0]
        l = eidx[eidx == i[0]].shape[0]
        rs.append (min(r,l))
        ls.append (min(r,l))
    mix = 1 / np.array(rs + ls, dtype=np.float32) ** 1.2
    #mix2 = 1 / np.array(rs + ls)
    #print(mix.shape, "mix shape")
    
    return mix
        

#def update_pin_group():
#    """Updates the cloth data after changing mesh or vertex weight pins"""
#    ob = get_last_object()[1]
#    create_cloth_data(ob) #, new=False)


def collision_data_update(self, context):
    ob = self.id_data
    if ob.mc.self_collision:    
        #ob = get_last_object()[1]
        create_cloth_data(ob) #, new=False)    


def refresh_noise(self, context):
    cloth = get_cloth_data(self)
    if cloth:
        zeros = np.zeros(cloth.count, dtype=np.float32)
        random = np.random.random(cloth.count)
        zeros[:] = random
        cloth.noise = ((zeros + -0.5) * self.mc.noise * 0.1)[:, nax]


def generate_wind(wind_vec, ob, nor_arr, wind, vel):
    """Maintains a wind array and adds it to the cloth vel"""    
    wind *= 0.9
    if np.any(wind_vec):
        turb = ob.mc.turbulence
        w_vec = revert_rotation(ob, wind_vec)
        wind += w_vec * (1 - np.random.random(nor_arr.shape) * -turb) 
        
        # only blow on verts facing the wind
        perp = nor_arr @ w_vec 
        wind *= np.abs(perp[:, nax])
        vel += wind    


def check_and_get_pins_and_hooks(ob):
    scene = bpy.context.scene
    pins = []
    hooks = []
    cull_ids = []
    for i, pin in enumerate(ob.mc.pins):
        # Only one user means actual pin object already deleted
        if pin.hook.users == 1:
            cull_ids.append(i)
        else:
            #vert = ob.data.vertices[pin.vertex_id]
            pins.append(pin.vertex_id)
            hooks.append(pin.hook)

    # Delete missing hooks data
    for i in cull_ids:
        pin = ob.mc.pins[i]
        bpy.data.objects.remove(pin.hook)
        ob.mc.pins.remove(i)

    return pins, hooks
        
    
class ClothData:
    def __init__(self):
        pass

        # Optionally preserved

        #self.idxer = None #np (vert ids)
        #self.co = None #np (vert coords)
        #self.eidx = None #np (linear and diagonal edge vertex ids)
        #self.eidx_tiler = None #np (transpose ravel of eidx)
        #self.pindexer = None #np (vert ids of not weight pinned verts)
        #self.sco = None #np (shape key coordinates)
        #self.sew_edges = None #np (vert ids of sew edges)
        #self.mix = None #np
        #self.noise = None #np
        #self.c_peat = None #np
        #self.t_peat = None #np
        #self.tridex = None #np (vert ids of triangulate meshes)
        #self.tridexer = None #np (triangle ids of triangulate meshes)
        #self.unpinned = None #np (edge vert ids(?) of not weight pinned verts)
        #self.v_normals = None #np (vertex normals)
        #self.vel_start = None #np (vertex coord at start of every frame)

        ## Better be preserved

        #self.vel = None #np (vertex velocity vectors)
        #self.wind = None #np (wind vectors)

        ###

        #self.count = 0 #int (num_vertices)
        ##self.pcount = 0 #int (number of not weight pinned verts)
        ##self.clicked = False #bool (drag event)
        ##self.name = '' #string
        #self.ob = None #object pointer
        ##self.waiting = False #bool

def create_cloth_data(ob): #, new=True):
    """Creates instance of cloth object with attributes needed for engine"""
    data = bpy.context.window_manager.modeling_cloth_data_set
    #cloth = get_cloth_data(ob)

    # Try to get the cloth data first
    try:
        cloth =  data[ob.name]
    except:
        # Search for possible name changes
        cloth = None
        for ob_name, c in data.items():
            if c.ob == ob:

                # Rename the key
                data[ob.name] = data.pop(ob_name)
                cloth = data[ob.name]
                break

        # If cloth still not found
        if not cloth:
            cloth = ClothData()
            data[ob.name] = cloth
            cloth.ob = ob

    scene = bpy.context.scene
    scene.objects.active = ob
    
    cloth.idxer = np.arange(len(ob.data.vertices), dtype=np.int32)
    # data only accesible through object mode
    mode = ob.mode
    if mode == 'EDIT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # data is read from a source shape and written to the display shape so we can change the target springs by changing the source shape
    #cloth.name = ob.name
    if ob.data.shape_keys == None:
        ob.shape_key_add('Basis')    
    if 'modeling cloth source key' not in ob.data.shape_keys.key_blocks:
        ob.shape_key_add('modeling cloth source key')        
    if 'modeling cloth key' not in ob.data.shape_keys.key_blocks:
        ob.shape_key_add('modeling cloth key')        
        ob.data.shape_keys.key_blocks['modeling cloth key'].value=1
    cloth.count = len(ob.data.vertices)
    
    # we can set a large group's pin state using the vertex group. No hooks are used here
    if 'modeling_cloth_pin' not in ob.vertex_groups:
        cloth.pin_group = create_vertex_groups(groups=['modeling_cloth_pin'], weights=[0.0], ob=None)
    for i in range(cloth.count):
        try:
            ob.vertex_groups['modeling_cloth_pin'].weight(i)
        except RuntimeError:
            # assign a weight of zero
            ob.vertex_groups['modeling_cloth_pin'].add(range(0,len(ob.data.vertices)), 0.0, 'REPLACE')
    cloth.pin_bool = -np.array([ob.vertex_groups['modeling_cloth_pin'].weight(i) for i in range(cloth.count)], dtype=np.bool)

    # unique edges------------>>>
    uni_edges = get_minimal_edges(ob)
    if len(uni_edges[1]) > 0:   
        cloth.eidx = np.append(uni_edges[0], uni_edges[1], axis=0)
    else:
        cloth.eidx = uni_edges[0]
    #cloth.eidx = uni_edges[0][0]
    cloth.sew_edges = uni_edges[2]

    #if cloth.virtual_springs.shape[0] > 0:
        #cloth.eidx = np.append(cloth.eidx, cloth.virtual_springs, axis=0)
    if len(ob.mc.virtual_springs) > 0:
        virtual_springs = np.array([[vs.vertex_id_1, vs.vertex_id_2] for vs in ob.mc.virtual_springs])
        cloth.eidx = np.append(cloth.eidx, virtual_springs, axis=0)
    cloth.eidx_tiler = cloth.eidx.T.ravel()    

    #eidx1 = np.copy(cloth.eidx)
    cloth.pindexer = np.arange(cloth.count, dtype=np.int32)[cloth.pin_bool]
    cloth.unpinned = np.in1d(cloth.eidx_tiler, cloth.pindexer)
    cloth.eidx_tiler = cloth.eidx_tiler[cloth.unpinned]    
    #cloth.unpinned = unpinned
    #cloth.pcount = pindexer.shape[0]
    #cloth.pindexer = pindexer
    #uni = np.unique(cloth.eidx_tiler, return_inverse=True, return_counts=True)

    #cloth.mix = (1/uni[2][uni[1]])[:, nax].astype(np.float32) # force gets divided by number of springs

    # this helps with extra springs behaving as if they had more mass---->>>
    mixology = get_spring_mix(ob, cloth.eidx)
    cloth.mix = mixology[cloth.unpinned][:, nax]
    #cloth.mix = (cloth.mix ** .87) * 0.35
    
    # unique edges------------>>>
    
    # Source shape key coordinates
    cloth.sco = np.zeros(cloth.count * 3, dtype=np.float32)
    ob.data.shape_keys.key_blocks['modeling cloth source key'].data.foreach_get('co', cloth.sco)
    cloth.sco.shape = (cloth.count, 3)

    # Current shape key coordinates
    cloth.co = np.zeros(cloth.count * 3, dtype=np.float32)
    ob.data.shape_keys.key_blocks['modeling cloth key'].data.foreach_get('co', cloth.co)
    cloth.co.shape = (cloth.count, 3)    
    co = cloth.co

    # Velocity
    cloth.vel = np.zeros(cloth.count * 3, dtype=np.float32)
    cloth.vel.shape = (cloth.count, 3)
    cloth.vel_start = np.zeros(cloth.count * 3, dtype=np.float32)
    cloth.vel_start.shape = (cloth.count, 3)
    
    # Normals
    cloth.v_normals = np.zeros(co.shape, dtype=np.float32)
    get_v_normals(ob, cloth.v_normals)

    # Wind
    cloth.wind = np.zeros(co.shape, dtype=np.float32)
    
    #noise---
    noise_zeros = np.zeros(cloth.count, dtype=np.float32)
    random = np.random.random(cloth.count)
    noise_zeros[:] = random
    cloth.noise = ((noise_zeros + -0.5) * ob.mc.noise * 0.1)[:, nax]
    
    #cloth.waiting = False
    #cloth.clicked = False # for the grab tool
    
    # -------------->>>

    # new self collisions:

    cloth.tridex = triangulate(ob)
    cloth.tridexer = np.arange(cloth.tridex.shape[0], dtype=np.int32)

    c_peat = np.repeat(np.arange(cloth.idxer.shape[0], dtype=np.int16), cloth.tridexer.shape[0])
    t_peat = np.tile(np.arange(cloth.tridexer.shape[0], dtype=np.int16), co.shape[0])

    # eliminate tris that contain the point:
    co_tidex = c_peat.reshape(cloth.idxer.shape[0], cloth.tridexer.shape[0])
    tri_tidex = cloth.tridex[t_peat.reshape(cloth.idxer.shape[0], cloth.tridexer.shape[0])]
    check = tri_tidex == co_tidex[:,:,nax]
    cull = ~np.any(check, axis=2)
    
    # duplicate of each tri for each vert and each vert for each tri
    cloth.c_peat = c_peat[cull.ravel()]
    cloth.t_peat = t_peat[cull.ravel()]

    
    self_col = ob.mc.self_collision
    if self_col:
        # collision======:
        # collision======:
        cloth.p_count = len(ob.data.polygons)
        #cloth.p_means = get_poly_centers(ob)
        cloth.p_means = simple_poly_centers(ob, key="modeling cloth source key")

        
        # could put in a check in case int 32 isn't big enough...
        cloth.cy_dists, cloth.point_mean_neighbors = generate_collision_data(ob, cloth.pindexer, cloth.p_means)
        cloth.cy_dists *= ob.mc.self_collision_cy_size
        
        nei = cloth.point_mean_neighbors.ravel() # eliminate neighbors for point in face check
        cloth.v_repeater = np.repeat(cloth.pindexer, cloth.p_count)[nei]
        cloth.p_repeater = np.tile(np.arange(cloth.p_count, dtype=np.int32),(cloth.count,))[nei]
        cloth.bool_repeater = np.ones(cloth.p_repeater.shape[0], dtype=np.bool)
        
        cloth.mean_idxer = np.arange(cloth.p_count, dtype=np.int32)
        cloth.mean_tidxer = np.tile(cloth.mean_idxer, (cloth.count, 1))
        
        # collision======:
        # collision======:

    
    bpy.ops.object.mode_set(mode=mode)
    print('INFO: Cloth data for', ob.name, 'is created!')
    return cloth


def run_handler(ob, cloth):
    #print(bpy.context.scene.frame_current)
    scene = bpy.context.scene
    extra_data = bpy.context.window_manager.modeling_cloth_data_set_extra
    col_data = bpy.context.window_manager.modeling_cloth_data_set_colliders
    #ob = cloth.ob

    # BIG TRY, UPDATE CLOTH IF ERROR HAPPENS
    if ob.mc.frame_update | ob.mc.scene_update:
    #try:
        #cloth = get_cloth_data(ob)

        #if ob.mode == 'EDIT':
        if ob.mode != 'OBJECT':
            ob.mc.waiting = True

        if ob.mc.waiting:    
            if ob.mode == 'OBJECT':
                # Update cloth if number of vertices changes
                #if ob.mc.force_update or len(ob.data.vertices) != cloth.co.shape[0]:
                #    create_cloth_data(ob)
                #    ob.mc.force_update = False
                create_cloth_data(ob)
                ob.mc.waiting = False

        if not ob.mc.waiting:
            #print('INFO: Beginning cloth handler loops...')
    
            eidx = cloth.eidx # world's most important variable

            ob.data.shape_keys.key_blocks['modeling cloth source key'].data.foreach_get('co', cloth.sco.ravel())
            sco = cloth.sco
            #sco.shape = (cloth.count, 3)
            #ob.data.shape_keys.key_blocks['modeling cloth key'].data.foreach_get('co', cloth.co.ravel())
            co = cloth.co
            #co.shape = (cloth.count, 3)
            svecs = sco[eidx[:, 1]] - sco[eidx[:, 0]]
            sdots = np.einsum('ij,ij->i', svecs, svecs)

            co[cloth.pindexer] += cloth.noise[cloth.pindexer]
            cloth.noise *= ob.mc.noise_decay

            # mix in vel before collisions and sewing
            co[cloth.pindexer] += cloth.vel[cloth.pindexer]

            cloth.vel_start[:] = co
            force = ob.mc.spring_force
            mix = cloth.mix * force

            #if cloth.clicked: # for the grab tool
                #grab = np.array([extra_data['stored_vidx'][v] + extra_data['move'] for v in range(len(extra_data['vidx']))])
                #grab_idx = extra_data['vidx']    
                    #loc = extra_data['stored_vidx'][v] + extra_data['move']

            pin_list = []
            if len(ob.mc.pins) > 0:
                pin_list, hook_list = check_and_get_pins_and_hooks(ob)
                hook_co = np.array([ob.matrix_world.inverted() * hook.matrix_world.to_translation() 
                    for hook in hook_list])

            for x in range(ob.mc.iterations):    

                # add pull
                vecs = co[eidx[:, 1]] - co[eidx[:, 0]]
                dots = np.einsum('ij,ij->i', vecs, vecs)
                div = np.nan_to_num(sdots / dots)
                swap = vecs * np.sqrt(div)[:, nax]
                move = vecs - swap
                # pull only test--->>>
                #move[div > 1] = 0
                push = ob.mc.push_springs
                if push == 0:
                    move[div > 1] = 0
                else:
                    move[div > 1] *= push
                # pull only test--->>>
                
                tiled_move = np.append(move, -move, axis=0)[cloth.unpinned] * mix # * mix for stability: force multiplied by 1/number of springs

                #print(np.unique(cloth.eidx_tiler))
                
                #tiled_move = np.append(move, -move, axis=0)[cloth.unpinned] * new_mix # * mix for stability: force multiplied by 1/number of springs
                #T = time.time()
                np.add.at(cloth.co, cloth.eidx_tiler, tiled_move)
                #print(x, (time.time()-T)*1000)

                ###if ob.mc.object_collision_detect:
                    ###if cloth.collide_list.shape[0] > 0:
                        ###cloth.co[cloth.col_idx] = cloth.collide_list
                
                #if len(cloth.pin_list) > 0:
                #    hook_co = np.array([ob.matrix_world.inverted() * i.matrix_world.to_translation() for i in cloth.hook_list])
                #    cloth.co[cloth.pin_list] = hook_co
                #if pin_count > 0:
                if pin_list:
                    cloth.co[pin_list] = hook_co
                
                # grab inside spring iterations
                if ob.mc.clicked: # for the grab tool
                    cloth.co[extra_data['vidx']] = np.array(extra_data['stored_vidx']) + np.array(+ extra_data['move'])   
            
            spring_dif = cloth.co - cloth.vel_start
            grav = ob.mc.gravity * (.01 / ob.mc.iterations)
            cloth.vel += revert_rotation(ob, np.array([0, 0, grav]))

            # refresh normals for inflate and wind
            get_v_normals(ob, cloth.v_normals)

            # wind:
            x = ob.mc.wind_x
            y = ob.mc.wind_y
            z = ob.mc.wind_z
            wind_vec = np.array([x,y,z])
            generate_wind(wind_vec, ob, cloth.v_normals, cloth.wind, cloth.vel)            

            # inflate
            inflate = ob.mc.inflate * .1
            if inflate != 0:
                cloth.v_normals *= inflate
                cloth.vel += cloth.v_normals

            #cloth.vel += spring_dif * 4                    
            # inextensible calc:
            ab_dot = np.einsum('ij, ij->i', cloth.vel, spring_dif)
            aa_dot = np.einsum('ij, ij->i', spring_dif, spring_dif)
            div = np.nan_to_num(ab_dot / aa_dot)
            cp = spring_dif * div[:, nax]
            cloth.vel -= np.nan_to_num(cp)
            cloth.vel += (spring_dif + cp)
            
            cloth.vel += spring_dif        

            # The amount of drag increases with speed. 
            # have to converto to a range between 0 and 1
            squared_move_dist = np.einsum("ij, ij->i", cloth.vel, cloth.vel)
            squared_move_dist += 1
            cloth.vel *= (1 / (squared_move_dist / ob.mc.velocity))[:, nax]
            
            # old self collisions
            self_col = ob.mc.self_collision

            if self_col:
                V3 = [] # because I'm multiplying the vel by this value and it doesn't exist unless there are collisions
                col_margin = ob.mc.self_collision_margin
                sq_margin = col_margin ** 2
                
                cloth.p_means = get_poly_centers(ob)
                
                #======== collision tree---
                # start with the greatest dimension(if it's flat on the z axis, it will return everything so start with an axis with the greatest dimensions)
                order = np.argsort(ob.dimensions) # last on first since it goes from smallest to largest
                axis_1 = cloth.co[:, order[2]]
                axis_2 = cloth.co[:, order[1]]
                axis_3 = cloth.co[:, order[0]]
                center_1 = cloth.p_means[:, order[2]]
                center_2 = cloth.p_means[:, order[1]]
                center_3 = cloth.p_means[:, order[0]]
                
                V = cloth.v_repeater # one set of verts for each face
                P = cloth.p_repeater # faces repeated in order to aling to v_repearter
                
                check_1 = np.abs(axis_1[V] - center_1[P]) < cloth.cy_dists[P]
                V1 = V[check_1]
                P1 = P[check_1]
                C1 = cloth.cy_dists[P1]
                
                check_2 = np.abs(axis_2[V1] - center_2[P1]) < C1
                
                V2 = V1[check_2]
                P2 = P1[check_2]
                C2 = C1[P2]            

                check_3 = np.abs(axis_3[V2] - center_3[P2]) < C2

                v_hits = V2[check_3]
                p_hits = P2[check_3]
                #======== collision tree end ---
                if p_hits.shape[0] > 0:        
                    # now do closest point edge with points on normals
                    normals = get_poly_normals(ob)[p_hits]
                    
                    base_vecs = cloth.co[v_hits] - cloth.p_means[p_hits]
                    d = np.einsum('ij,ij->i', base_vecs, normals) / np.einsum('ij,ij->i', normals, normals)        
                    cp = normals * d[:, nax]
                    
                    # now measure the distance along the normal to see if it's in the cylinder
                    
                    cp_dot = np.einsum('ij,ij->i', cp, cp)
                    in_margin = cp_dot < sq_margin
                    
                    if in_margin.shape[0] > 0:
                        V3 = v_hits[in_margin]
                        #P3 = p_hits[in_margin]
                        cp3 = cp[in_margin]
                        cpd3 = cp_dot[in_margin]
                        
                        d1 = sq_margin
                        d2 = cpd3
                        div = d1/d2
                        surface = cp3 * np.sqrt(div)[:, nax]

                        force = np.nan_to_num(surface - cp3)
                        force *= ob.mc.self_collision_force
                        
                        cloth.co[V3] += force

                        cloth.vel[V3] *= .2                   
                        #np.add.at(cloth.co, V3, force * .5)
                        #np.multiply.at(cloth.vel, V3, 0.2)
                        
                        # could get some speed help by iterating over a dict maybe
                        #if False:    
                        #if True:    
                            #for i in range(len(P3)):
                                #cloth.co[cloth.v_per_p[P3[i]]] -= force[i]
                                #cloth.vel[cloth.v_per_p[P3[i]]] -= force[i]
                                #cloth.vel[cloth.v_per_p[P3[i]]] *= 0.2
                                #cloth.vel[cloth.v_per_p[P3[i]]] += cloth.vel[V3[i]]
                                #need to transfer the velocity back and forth between hit faces.

            #collision=====================================


            if ob.mc.sew != 0:
                if len(cloth.sew_edges) > 0:
                    sew_edges = cloth.sew_edges
                    rs = co[sew_edges[:,1]]
                    ls = co[sew_edges[:,0]]
                    sew_vecs = (rs - ls) * 0.5 * ob.mc.sew
                    co[sew_edges[:,1]] -= sew_vecs
                    co[sew_edges[:,0]] += sew_vecs

            
            # floor ---
            if ob.mc.floor:    
                floored = cloth.co[:,2] < 0        
                cloth.vel[:,2][floored] *= -1
                cloth.vel[floored] *= .1
                cloth.co[:, 2][floored] = 0
            # floor ---            
            

            #co[cloth.pindexer] += cloth.vel[cloth.pindexer]
            #cloth.co = co
            
            # objects ---
            #T = time.time()
            if ob.mc.object_collision_detect:
                #print('INFO: Doing collider loops...')
                cull_ids = []
                for i, cp in enumerate(scene.mc.collider_pointers):
                    # Delete pointers if object is deleted
                    if not cp.object or cp.object.users < 2: 
                        cull_ids.append(i)
                        continue
                    #col = get_collider_data(cp.object)
                    if cp.object == ob:    
                        # Does nothing because it will cause error if using modifier
                        pass
                        #if not AUTO_UPDATE_WHEN_ERROR_HAPPENS:
                        #    self_collide(ob)
                        #else:
                        #    try:
                        #       self_collide(ob)
                        #    except:
                        #        print(sys.exc_info())
                        #        print('ERROR: Collider error! Updating to new collider data!')
                        #        col = create_collider_data(ob)
                        #        self_collide(ob)
                    else:    
                        if not AUTO_UPDATE_WHEN_ERROR_HAPPENS:
                            object_collide(ob, cp.object)
                        else:
                            try:
                               object_collide(ob, cp.object)
                            except:
                                print(sys.exc_info())
                                print('ERROR: Collider error! Updating to new collider data!')
                                col = create_collider_data(cp.object)
                                object_collide(ob, cp.object)

                # Delete pointers and data for deleted colliders
                for i in reversed(cull_ids):
                    cp = scene.mc.collider_pointers[i]

                    #cull_keys = []
                    #for key, col in col_data.items():
                    #    if col.ob == cp.object:
                    #        cull_keys.append(key)

                    #for key in cull_keys:
                    #    del(col_data[key])

                    scene.mc.collider_pointers.remove(i)
            #print(time.time()-T, "the whole enchalada")
            #print(x)
            # objects ---

            #if len(cloth.pin_list) > 0:
            #    cloth.co[cloth.pin_list] = hook_co
            #    cloth.vel[cloth.pin_list] = 0
            if pin_list:
                cloth.co[pin_list] = hook_co
                cloth.vel[pin_list] = 0

            if ob.mc.clicked: # for the grab tool
                cloth.co[extra_data['vidx']] = np.array(extra_data['stored_vidx']) + np.array(+ extra_data['move'])
            
            ob.data.shape_keys.key_blocks['modeling cloth key'].data.foreach_set('co', cloth.co.ravel())
            ob.data.shape_keys.key_blocks['modeling cloth key'].mute = True
            ob.data.shape_keys.key_blocks['modeling cloth key'].mute = False

    #except:
    #    print(sys.exc_info())
    #    print('ERROR: Cloth error! Updating cloth!')
    #    create_cloth_data(ob)
    #    #if ob.mc.object_collision_detect:
    #    #    for i, cp in enumerate(scene.mc.collider_pointers):
    #    #        create_collider_data(cp.object)

# +++++++++++++ object collisions ++++++++++++++
def bounds_check(co1, co2, fudge):
    """Returns True if object bounding boxes intersect.
    Have to add the fudge factor for collision margins"""
    check = False
    co1_max = None # will never return None if check is true
    co1_min = np.min(co1, axis=0)
    co2_max = np.max(co2, axis=0)

    if np.all(co2_max + fudge > co1_min):
        co1_max = np.max(co1, axis=0)
        co2_min = np.min(co2, axis=0)        
        
        if np.all(co1_max > co2_min - fudge):
            check = True

    return check, co1_min, co1_max # might as well reuse the checks


def triangle_bounds_check(co, tri_co, co_min, co_max, idxer, fudge):
    """Returns a bool aray indexing the triangles that
    intersect the bounds of the object"""

    # min check cull step 1
    tri_min = np.min(tri_co, axis=1) - fudge
    check_min = co_max > tri_min
    in_min = np.all(check_min, axis=1)
    
    # max check cull step 2
    idx = idxer[in_min]
    tri_max = np.max(tri_co[in_min], axis=1) + fudge
    check_max = tri_max > co_min
    in_max = np.all(check_max, axis=1)
    in_min[idx[~in_max]] = False
    
    return in_min, tri_min[in_min], tri_max[in_max] # can reuse the min and max


def tri_back_check(co, tri_min, tri_max, idxer, fudge):
    """Returns a bool aray indexing the vertices that
    intersect the bounds of the culled triangles"""

    # min check cull step 1
    tb_min = np.min(tri_min, axis=0) - fudge
    check_min = co > tb_min
    in_min = np.all(check_min, axis=1)
    idx = idxer[in_min]
    
    # max check cull step 2
    tb_max = np.max(tri_max, axis=0) + fudge
    check_max = co[in_min] < tb_max
    in_max = np.all(check_max, axis=1)        
    in_min[idx[~in_max]] = False    
    
    return in_min 
    

def v_per_tri(co, tri_min, tri_max, idxer, tridexer, c_peat=None, t_peat=None):
    """Checks each point against the bounding box of each triangle"""
    
    if c_peat is None:
        c_peat = np.repeat(np.arange(idxer.shape[0], dtype=np.int16), tridexer.shape[0])
        t_peat = np.tile(np.arange(tridexer.shape[0], dtype=np.int16), co.shape[0])
    
    # X
    # Step 1 check x_min (because we're N squared here we break it into steps)
    co_x = co[:, 0]
    check_x_min = co_x[c_peat] > tri_min[:, 0][t_peat]
    c_peat = c_peat[check_x_min]
    if c_peat.shape[0] == 0:
        return
    t_peat = t_peat[check_x_min]

    # Step 2 check x max
    check_x_max = co_x[c_peat] < tri_max[:, 0][t_peat]
    c_peat = c_peat[check_x_max]
    if c_peat.shape[0] == 0:
        return
    t_peat = t_peat[check_x_max]
    
    # Y
    # Step 3 check y min    
    co_y = co[:, 1]
    check_y_min = co_y[c_peat] > tri_min[:, 1][t_peat]
    c_peat = c_peat[check_y_min]
    if c_peat.shape[0] == 0:
        return
    t_peat = t_peat[check_y_min]

    # Step 4 check y max
    check_y_max = co_y[c_peat] < tri_max[:, 1][t_peat]
    c_peat = c_peat[check_y_max]
    if c_peat.shape[0] == 0:
        return
    t_peat = t_peat[check_y_max]
        
    # Z
    # Step 5 check z min    
    co_z = co[:, 2]
    check_z_min = co_z[c_peat] > tri_min[:, 2][t_peat]
    c_peat = c_peat[check_z_min]
    if c_peat.shape[0] == 0:
        return
    t_peat = t_peat[check_z_min]

    # Step 6 check y max
    check_z_max = co_z[c_peat] < tri_max[:, 2][t_peat]
    c_peat = c_peat[check_z_max]
    if c_peat.shape[0] == 0:
        return
    t_peat = t_peat[check_z_max]    

    return idxer[c_peat], t_peat


def inside_triangles(tri_vecs, v2, co, tri_co_2, cidx, tidx, nor, ori, in_margin):
    idxer = np.arange(in_margin.shape[0], dtype=np.int32)[in_margin]
    
    r_co = co[cidx[in_margin]]    
    r_tri = tri_co_2[tidx[in_margin]]
    
    v0 = tri_vecs[:,0]
    v1 = tri_vecs[:,1]
    
    d00_d11 = np.einsum('ijk,ijk->ij', tri_vecs, tri_vecs)
    d00 = d00_d11[:,0]
    d11 = d00_d11[:,1]
    d01 = np.einsum('ij,ij->i', v0, v1)
    d02 = np.einsum('ij,ij->i', v0, v2)
    d12 = np.einsum('ij,ij->i', v1, v2)

    div = 1 / (d00 * d11 - d01 * d01)
    u = (d11 * d02 - d01 * d12) * div
    v = (d00 * d12 - d01 * d02) * div
    check = (u > 0) & (v > 0) & (u + v < 1)
    in_margin[idxer] = check


def object_collide(cloth_ob, col_ob):
    #print('INFO: Beginning collision between', cloth_ob.name, 'and', col_ob.name)
    ###hits = False
    ###cloth.col_idx = np.empty(0, dtype=np.int32)
    ###cloth.collide_list = np.empty(0, dtype=np.int32)
    # get transforms in world space:
    cloth = get_cloth_data(cloth_ob)
    col = get_collider_data(col_ob)

    proxy_in_place(col_ob, col)
    apply_in_place(cloth_ob, cloth.co, cloth)
    
    inner_margin = col.ob.mc.object_collision_inner_margin
    outer_margin = col.ob.mc.object_collision_outer_margin
    fudge = max(inner_margin, outer_margin)
    #fudge = outer_margin# * 2
    
    # check object bounds: (need inner and out margins to adjust box size)
    box_check, co1_min, co1_max = bounds_check(cloth.co, col.co, fudge)

    # check for triangles inside the cloth bounds
    if box_check:
        proxy_v_normals_in_place(col_ob, col)
        tri_co = col.co[col.tridex]

        # tri vel
        tri_vo = col.vel[col.tridex]
        tris_in, tri_min, tri_max = triangle_bounds_check(cloth.co, tri_co, co1_min, co1_max, col.tridexer, fudge)

        # check for verts in the bounds around the culled triangles
        if np.any(tris_in):    
            tri_co_2 = tri_co[tris_in]
            back_check = tri_back_check(cloth.co, tri_min, tri_max, cloth.idxer, fudge)
    
            # begin every vertex co against every tri
            if np.any(back_check):
                v_tris = v_per_tri(cloth.co[back_check], tri_min, tri_max, cloth.idxer[back_check], col.tridexer[tris_in])
                
                if v_tris is not None:
                    # update the normals. cross_vecs used by barycentric tri check
                    # move the surface along the vertex normals by the outer margin distance
                    marginalized = (col.co + col.v_normals * outer_margin)[col.tridex]
                    tri_normals_in_place(col_ob, col, marginalized)
                    # add normals to make extruded tris
                    norms_2 = col.normals[tris_in]
                    u_norms = norms_2 / np.sqrt(np.einsum('ij, ij->i', norms_2, norms_2))[:, nax] 
                                        
                    cidx, tidx = v_tris
                    ori = col.origins[tris_in][tidx]
                    nor = u_norms[tidx]
                    vec2 = cloth.co[cidx] - ori
                    
                    d = np.einsum('ij, ij->i', nor, vec2) # nor is unit norms
                    in_margin = (d > -(inner_margin + outer_margin)) & (d < 0)#outer_margin) (we have offset outer margin)
                    
                    # <<<--- Inside triangle check --->>>
                    # will overwrite in_margin:
                    cross_2 = col.cross_vecs[tris_in][tidx][in_margin]
                    inside_triangles(cross_2, vec2[in_margin], cloth.co, marginalized[tris_in], cidx, tidx, nor, ori, in_margin)
                    
                    if np.any(in_margin):
                        # collision response --------------------------->>>
                        # when we only compute the velocity for points that hit,
                        # we get weird velocity vectors when moving in and out
                        # of collision space
                        tri_vo = tri_vo[tris_in]
                        tri_vel1 = np.mean(tri_co_2[tidx[in_margin]], axis=1)
                        tri_vel2 = np.mean(tri_vo[tidx[in_margin]], axis=1)
                        tvel = tri_vel1 - tri_vel2
                        tri_vo[:] = 0
                        
                        
                        
                        col_idx = cidx[in_margin] 
                        #u, ind = np.unique(col_idx, True)
                        cloth.co[col_idx] -= nor[in_margin] * (d[in_margin])[:, nax]
                        #cloth.co[u] -= nor[in_margin][ind] * (d[in_margin][ind])[:, nax]
                        cloth.vel[col_idx] = tvel
                        #cloth.vel[col_idx] = 0
                        #cloth.vel[u] = tvel[ind]
                        # when iterating springs, we keep putting the collided points back
                        ###cloth.col_idx = col_idx
                        ###cloth.collide_list = cloth.co[col_idx]
                        ###hits = True                    
                # could use the mean of the original trianlge to determine which face
                #   to collide with when there are multiples. So closest mean gets used.
    col.vel[:] = col.co    
    revert_in_place(cloth_ob, cloth.co)
    ###if hits:
        ###cloth.collide_list = cloth.co[col_idx]

# update functions --------------------->>>    



def grid_sample(ob, tri_min, tri_max, box_count=10, offset=0.00001):
    """divide mesh into grid and sample from each segment.
    offset prevents boxes from excluding any verts"""

    # I have to eliminate tris containing the vertex
    # I have boxes containing verts
    # I have min and max corners of each tri
    # I have to check every tri whose bounding box intersects the box with the vert
    # If a vert is in a box, all the tris containing it are in that box


    #ob = cloth.ob
    cloth = get_cloth_data(ob)
    co = cloth.co
    #co = get_co(ob)
    
    # get bounding box corners
    min = np.min(co, axis=0)
    max = np.max(co, axis=0)
    
    # box count is based on largest dimension
    dimensions = max - min
    largest_dimension = np.max(dimensions)
    box_size = largest_dimension / box_count
    
    # get box count for each axis
    xyz_count = dimensions // box_size + 1 #(have to add one so we don't end up with zero boxes when the mesh is flat)
    
    # dynamic number of boxes on each axis:
    box_dimensions = dimensions / xyz_count # each box is this size
    
    line_end = max - box_dimensions # we back up one from the last value
    
    x_line = np.linspace(min[0], line_end[0], num=xyz_count[0], dtype=np.float32)
    y_line = np.linspace(min[1], line_end[1], num=xyz_count[1], dtype=np.float32)
    z_line = np.linspace(min[2], line_end[2], num=xyz_count[2], dtype=np.float32)
    
    
    idxer = np.arange(co.shape[0])
    
    # get x bools
    x_grid = co[:, 0] - x_line[:,nax]
    x_bools = (x_grid + offset > 0) & (x_grid - offset < box_dimensions[0])
    cull_x_bools = x_bools[np.any(x_bools, axis=1)] # eliminate grid sections with nothing
    xb = cull_x_bools

    x_idx = np.tile(idxer, (xyz_count[0], 1))
    
    samples = []
    
    for boo in xb:
        xidx = idxer[boo]
        y_grid = co[boo][:, 1] - y_line[:,nax]
        y_bools = (y_grid + offset > 0) & (y_grid - offset < box_dimensions[1])
        cull_y_bools = y_bools[np.any(y_bools, axis=1)] # eliminate grid sections with nothing
        yb = cull_y_bools
        #print(yb[0].shape, "++++++what's my shape here?")
        #print(yb[0][yb[0]].shape, "what's my shape here?")
        for yboo in yb:
            yidx = xidx[yboo]
            z_grid = co[yidx][:, 2] - z_line[:,nax]
            z_bools = (z_grid + offset > 0) & (z_grid - offset < box_dimensions[2])
            cull_z_bools = z_bools[np.any(z_bools, axis=1)] # eliminate grid sections with nothing
            zb = cull_z_bools        
            for zboo in zb:
                #samples.append(yidx[zboo][0])
            
                # !!! to use this for collisions !!!:
                #if False:    
                samples.extend([yidx[zboo].tolist()])

    
    #print(samples[:10], "what are these?")
    for i in samples[40]:
        ob.data.vertices[i].select=True
    return np.unique(samples) # if offset is zero we don't need unique... return samples


def self_collide(ob):
    #print('INFO: Beginning self collision of', ob.name)
    #return
    # get transforms in world space:
    cloth = get_cloth_data(ob)
    col = get_collider_data(ob)

    fudge = col.ob.mc.object_collision_outer_margin

    #proxy_v_normals_in_place(col, False)
    tri_co = cloth.co[cloth.tridex]

    # tri vel
    tri_vo = cloth.vel[cloth.tridex]


    tri_min = np.min(tri_co, axis=1) - fudge
    tri_max = np.max(tri_co, axis=1) + fudge

    #T = time.time()
    #
    #v_tris = v_per_tri(cloth.co, tri_min, tri_max, cloth.idxer, cloth.tridexer, cloth.c_peat, cloth.t_peat)
    #x = grid_sample(ob, tri_min, tri_max, box_count=10, offset=0.00001)
    #print(time.time()-T, "time inside")
    #print(cloth.t_peat)
    #print(cloth.c_peat)
    
    
    return
    if v_tris is not None:
        # update the normals. cross_vecs used by barycentric tri check
        # move the surface along the vertex normals by the outer margin distance
        tri_normals_in_place(ob, col, tri_co)
        # add normals to make extruded tris
        u_norms = norms_2 / np.sqrt(np.einsum('ij, ij->i', norms_2, norms_2))[:, nax] 
                            
        cidx, tidx = v_tris
        ori = col.origins[tris_in][tidx]
        nor = u_norms[tidx]
        vec2 = cloth.co[cidx] - ori
        
        d = np.einsum('ij, ij->i', nor, vec2) # nor is unit norms
        in_margin = (d > -inner_margin) & (d < 0)#< outer_margin)
        
        # <<<--- Inside triangle check --->>>
        # will overwrite in_margin:
        cross_2 = col.cross_vecs[tris_in][tidx][in_margin]
        inside_triangles(cross_2, vec2[in_margin], cloth.co, marginalized[tris_in], cidx, tidx, nor, ori, in_margin)
        
        
        if np.any(in_margin):
            # collision response --------------------------->>>
            tri_vo = tri_vo[tris_in]
            tri_vel1 = np.mean(tri_co_2[tidx[in_margin]], axis=1)
            tri_vel2 = np.mean(tri_vo[tidx[in_margin]], axis=1)
            tvel = tri_vel1 - tri_vel2
            
            
            col_idx = cidx[in_margin] 
            #u, ind = np.unique(col_idx, True)
            cloth.co[col_idx] -= nor[in_margin] * (d[in_margin])[:, nax]
            #cloth.co[u] -= nor[in_margin][ind] * (d[in_margin][ind])[:, nax]
            cloth.vel[col_idx] = tvel
            #cloth.vel[col_idx] = 0
            #cloth.vel[u] = tvel[ind]
            col.vel[:] = col.co                    

            # when iterating springs, we keep putting the collided points back
            ###cloth.col_idx = col_idx
            ###cloth.collide_list = cloth.co[col_idx]
            ###hits = True                    
    # could use the mean of the original trianlge to determine which face
    #   to collide with when there are multiples. So closest mean gets used.
    
    revert_in_place(ob, cloth.co)
    ###if hits:
        ###cloth.collide_list = cloth.co[col_idx]


class ColliderData:
    def __init__(self):
        pass
        #self.co = None
        #self.cross_vecs = None
        #self.normals = None
        #self.origins = None
        #self.tridex = None
        #self.tridexer = None
        #self.v_normals = None
        #self.vel = None

        #self.ob = None


def get_collider_data(ob):
    #print('INFO: Getting collider data of', ob.name, '...')
    col_data = bpy.context.window_manager.modeling_cloth_data_set_colliders
    #try:
    #    col = col_data[ob.name]
    #    print('Her yaa go!', col.ob)
    #    return col
    #except:
    #    print(sys.exc_info())
    #    # Search for possible name changes
    #    for ob_name, c in col_data.items():
    #        if c.ob == ob:

    #            # Rename the key
    #            col_data[ob.name] = col_data.pop(ob_name)
    #            return col_data[ob.name]

    ## If collider still not found
    #return create_collider_data(ob)

    col = None
    for key, c in col_data.items():
    #for c in col_data:
        if c.ob == ob:
            col = c

    if not col:
        col = create_collider_data(ob)

    #print('Her yaa go!', col.ob)

    return col

def create_collider_data(ob):
    # maybe fixed? !!! bug where first frame of collide uses empty data. Stuff goes flying.
    #print('INFO: Creating collider data of', ob.name, '...')
    col_data = bpy.context.window_manager.modeling_cloth_data_set_colliders

    # Try to get the collider data first
    #try:
    #    col =  col_data[ob.name]
    #except:
    #    # Search for possible name changes
    #    col = None
    #    for ob_name, c in col_data.items():
    #        if c.ob == ob:

    #            # Rename the key
    #            col_data[ob.name] = col_data.pop(ob_name)
    #            col = col_data[ob.name]
    #            break

    #    # If collider still not found
    #    if not col:
    #        col = ColliderData()
    #        col_data[ob.name] = col
    #        col.ob = ob

    col = ColliderData()
    col_data[ob.name] = col
    col.ob = ob

    col.co = np.copy(get_proxy_co(ob, None))
    proxy_in_place(ob, col)
    col.v_normals = proxy_v_normals(ob)
    col.vel = np.copy(col.co)
    col.tridex = triangulate(ob)
    col.tridexer = np.arange(col.tridex.shape[0], dtype=np.int32)
    # cross_vecs used later by barycentric tri check
    proxy_v_normals_in_place(ob, col)
    marginalized = np.array(col.co + col.v_normals * ob.mc.object_collision_outer_margin, dtype=np.float32)
    col.cross_vecs, col.origins, col.normals = get_tri_normals(marginalized[col.tridex])    

    col.cross_vecs.dtype = np.float32
    col.origins.dtype = np.float32
    #col.normals.dtype = np.float32

    print('INFO: Collider data for', ob.name, 'is created!')

    return col


# Self collision object
def create_self_collider(ob):
    # maybe fixed? !!! bug where first frame of collide uses empty data. Stuff goes flying.
    col = ColliderData()
    col.ob = ob
    col.co = get_co(ob, None)
    proxy_in_place(ob, col)
    col.v_normals = proxy_v_normals(ob)
    col.vel = np.copy(col.co)
    col.tridex = triangulate(ob)
    col.tridexer = np.arange(col.tridex.shape[0], dtype=np.int32)
    # cross_vecs used later by barycentric tri check
    proxy_v_normals_in_place(ob, col)
    marginalized = np.array(col.co + col.v_normals * ob.mc.object_collision_outer_margin, dtype=np.float32)
    col.cross_vecs, col.origins, col.normals = get_tri_normals(marginalized[col.tridex])    

    col.cross_vecs.dtype = np.float32
    col.origins.dtype = np.float32
    #col.normals.dtype = np.float32

    return col


# collide object updater
def collision_object_update(self, context):
    """Updates the collider object"""    
    ob = self.id_data
    scene = context.scene
    #extra_data = context.window_manager.modeling_cloth_data_set_extra
    #col_data = context.window_manager.modeling_cloth_data_set_colliders
    #collide = ob.mc.object_collision
    #print(collide, "this is the collide state")
    # remove objects from dict if deleted
    #cull_list = []
    #if 'colliders' in extra_data:
    #    if extra_data['colliders'] is not None:   
    #        if not collide:
    #            print("not collide")
    #            if ob.name in extra_data['colliders']:
    #                print("name was here")
    #                del(extra_data['colliders'][ob.name])
    #        for i in extra_data['colliders']:
    #            remove = True
    #            if i in bpy.data.objects:
    #                if bpy.data.objects[i].type == "MESH":
    #                    if bpy.data.objects[i].mc.object_collision:
    #                        remove = False
    #            if remove:
    #                cull_list.append(i)
    #for i in cull_list:
    #    del(extra_data['colliders'][i])

    ## add class to dict if true.
    #if collide:    
    #    if 'colliders' not in extra_data:    
    #        extra_data['colliders'] = {}
    #    if extra_data['colliders'] is None:
    #        extra_data['colliders'] = {}
    #    extra_data['colliders'][ob.name] = create_collider_data(ob)

    if self.object_collision:
        cp = scene.mc.collider_pointers.add()
        cp.object = ob
        #create_collider_data(ob)
        #col_data[ob.name] = create_collider_data(ob)
    else:
        for i, cp in enumerate(scene.mc.collider_pointers):
            if cp.object == ob:
                scene.mc.collider_pointers.remove(i)
                break
    
# cloth object detect updater:
def cloth_object_update(self, context):
    """Updates the cloth object when detecting."""
    print("ran the detect updater. It did nothing.")


def manage_animation_handler(self, context):
    ob = self.id_data
    if ob.mc.frame_update:
        ob.mc.scene_update = False
        #create_cloth_data(ob)
    
        
def manage_continuous_handler(self, context):    
    ob = self.id_data
    if ob.mc.scene_update:
        ob.mc.frame_update = False
        #create_cloth_data(ob)
    

# =================  Handler  ======================

def handler_unified(scene, frame_update=False):
    data = bpy.context.window_manager.modeling_cloth_data_set
    cull_ids = []

    for i, cp in enumerate(scene.mc.cloth_pointers):
        ob = cp.object

        # Check if object still exists
        if not ob or ob.users < 2:
            cull_ids.append(i)

        elif ob.users == 2 and scene.mc.last_object == ob and not ob.mc.object_collision:
            scene.mc.last_object = None
            cull_ids.append(i)

        elif ob.users == 2 and scene.mc.last_object != ob and ob.mc.object_collision:
            for idx, p in enumerate(scene.mc.collider_pointers):
                if p.object == ob:
                    scene.mc.collider_pointers.remove(idx)
                    break
            cull_ids.append(i)

        elif ob.users == 3 and scene.mc.last_object == ob and ob.mc.object_collision:
            scene.mc.last_object = None
            for idx, p in enumerate(scene.mc.collider_pointers):
                if p.object == ob:
                    scene.mc.collider_pointers.remove(idx)
                    break
            cull_ids.append(i)

        ## End checking

        else:
            cloth = get_cloth_data(ob)

            # Frame update
            if frame_update and ob.mc.frame_update:
                run_handler(ob, cloth)
                if ob.mc.auto_reset:
                    if scene.frame_current <= 1:    
                        reset_shapes(ob)

            # Scene update
            elif not frame_update and ob.mc.scene_update:
                run_handler(ob, cloth)

    # Delete missing object pointer and cloth data
    for i in reversed(cull_ids):
        cp = scene.mc.cloth_pointers[i]
        ob = cp.object

        # Remove cloth data first
        cull_keys = []
        for key, cloth in data.items():
            if ob == cloth.ob:
            #if ob == key:
                cull_keys.append(key)

        for key in cull_keys:
            del(data[key])

        # Remove pointers
        scene.mc.cloth_pointers.remove(i)

@persistent
def handler_frame(scene):
    handler_unified(scene, frame_update=True)


@persistent
def handler_scene(scene):
    handler_unified(scene, frame_update=False)

def get_cloth_data(ob):
    data = bpy.context.window_manager.modeling_cloth_data_set
    try:
        return data[ob.name]
    except:
        print(sys.exc_info())
        for ob_name, c in data.items():
            if c.ob == ob:

                # Rename the key
                data[ob.name] = data.pop(ob_name)
                return data[ob.name]

    ## If cloth still not found
    return create_cloth_data(ob)

def enable_cloth(self, context):
    ob = self.id_data
    scene = context.scene
    #data = context.window_manager.modeling_cloth_data_set
    extra_data = context.window_manager.modeling_cloth_data_set_extra
    scene.mc.last_object = ob
    
    # object collisions
    #colliders = [i for i in bpy.data.objects if i.mc.object_collision]
    #if len(colliders) == 0:    
    #    extra_data['colliders'] = None    
    
    # iterate through dict: for i, j in d.items()
    if ob.mc.enable:
        # New cloth on scene data
        cl = scene.mc.cloth_pointers.add()
        cl.object = ob
        
        #cloth = create_cloth_data(ob) # generate an instance of the class
        #data[cloth.ob.name] = cloth  # store class in dictionary using the object name as a key
        #cloth = create_cloth()
    else:
        for i, cl in enumerate(scene.mc.cloth_pointers):
            if cl.object == ob:
                scene.mc.cloth_pointers.remove(i)
    
    #cull_keys = [] # can't delete dict items while iterating
    #for key, value in data.items():
    #    if not value.ob.mc.enable:
    #        cull_keys.append(key) # store keys to delete

    #for key in cull_keys:
    #    del data[key]
    
#    # could keep the handler unless there are no modeling cloth objects active
#    
#    if handler_frame in bpy.app.handlers.frame_change_post:
#        bpy.app.handlers.frame_change_post.remove(handler_frame)
#    
#    if len(data) > 0:
#        bpy.app.handlers.frame_change_post.append(handler_frame)

def visible_objects_and_duplis(context):
    """Loop over (object, matrix) pairs (mesh only)"""

    for obj in context.visible_objects:
        if obj.type == 'MESH':
            if obj.mc.enable:    
                yield (obj, obj.matrix_world.copy())
                   
def obj_ray_cast(obj, matrix, ray_origin, ray_target):
    """Wrapper for ray casting that moves the ray into object space"""

    # get the ray relative to the object
    matrix_inv = matrix.inverted()
    ray_origin_obj = matrix_inv * ray_origin
    ray_target_obj = matrix_inv * ray_target
    ray_direction_obj = ray_target_obj - ray_origin_obj

    # cast the ray
    success, location, normal, face_index = obj.ray_cast(ray_origin_obj, ray_direction_obj)

    if face_index > len(obj.data.polygons):
        return None, None, None
    elif success:
        return location, normal, face_index
    else:
        return None, None, None


class ModelingClothPin(bpy.types.Operator):
    """Modal ray cast for placing pins"""
    bl_idname = "view3d.modeling_cloth_pin"
    bl_label = "Modeling Cloth Pin"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        data = context.window_manager.modeling_cloth_data_set
        return context.space_data.type == 'VIEW_3D' and any(data)

    def __init__(self):
        self.obj = None
        self.latest_hit = None
        self.closest = None

    def invoke(self, context, event):
        bpy.ops.object.select_all(action='DESELECT')    
        context.scene.mc.pin_alert = True

        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def raycast(self, context, event):
        # get the context arguments
        scene = context.scene
        region = context.region
        rv3d = context.region_data
        coord = event.mouse_region_x, event.mouse_region_y
    
        # get the ray from the viewport and mouse
        view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
    
        ray_target = ray_origin + view_vector
        
        guide = create_guide()
    
        # cast rays and find the closest object
        best_length_squared = -1.0
        best_obj = None
        best_matrix = None
        best_face_index = -1
        for obj, matrix in visible_objects_and_duplis(context):
            hit, normal, face_index = obj_ray_cast(obj, matrix, ray_origin, ray_target)
            if hit:
                hit_world = matrix * hit
                length_squared = (hit_world - ray_origin).length_squared
                if not best_obj or length_squared < best_length_squared:
                    best_length_squared = length_squared
                    best_obj = obj
                    best_face_index = face_index
                    best_matrix = matrix

        if best_obj:
            verts = np.array([best_matrix * best_obj.data.shape_keys.key_blocks['modeling cloth key'].data[v].co 
                for v in best_obj.data.polygons[best_face_index].vertices])
            vecs = verts - np.array(hit_world)
            vidx = [v for v in best_obj.data.polygons[best_face_index].vertices]
            self.closest = vidx[np.argmin(np.einsum('ij,ij->i', vecs, vecs))]
            loc = best_matrix * best_obj.data.shape_keys.key_blocks['modeling cloth key'].data[self.closest].co
            self.latest_hit = guide.location = loc
            self.obj = best_obj
        
    def modal(self, context, event):
        bpy.context.window.cursor_set("CROSSHAIR")

        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'NUMPAD_0',
        'NUMPAD_PERIOD','NUMPAD_1', 'NUMPAD_2', 'NUMPAD_3', 'NUMPAD_4',
         'NUMPAD_5', 'NUMPAD_6', 'NUMPAD_7', 'NUMPAD_8', 'NUMPAD_9'}:
            # allow navigation
            return {'PASS_THROUGH'}

        elif event.type == 'MOUSEMOVE':
            self.raycast(context, event)

        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if self.latest_hit and self.obj:
                e = bpy.data.objects.new('modeling_cloth_pin', None)
                bpy.context.scene.objects.link(e)
                e.location = self.latest_hit
                e.show_x_ray = True
                e.select = True
                e.empty_draw_size = .1
                #cloth = get_cloth_data(self.obj)
                pin = self.obj.mc.pins.add()
                pin.vertex_id = self.closest
                pin.hook = e
                #cloth.pin_list.append(self.closest)
                #cloth.hook_list.append(e)
                self.latest_hit = None
                self.obj = None
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            delete_guide()
            cloths = [i for i in bpy.data.objects if i.mc.enable] # so we can select an empty and keep the settings menu up
            context.scene.mc.pin_alert = False
            if len(cloths) > 0:                                        #
                ob = context.scene.mc.last_object
                bpy.context.scene.objects.active = ob
            bpy.context.window.cursor_set("DEFAULT")
            return {'FINISHED'}
            
        return {'RUNNING_MODAL'}


# drag===================================
# drag===================================
#[‘DEFAULT’, ‘NONE’, ‘WAIT’, ‘CROSSHAIR’, ‘MOVE_X’, ‘MOVE_Y’, ‘KNIFE’, ‘TEXT’, ‘PAINT_BRUSH’, ‘HAND’, ‘SCROLL_X’, ‘SCROLL_Y’, ‘SCROLL_XY’, ‘EYEDROPPER’]

# dragger===
class ModelingClothDrag(bpy.types.Operator):
    """Modal ray cast for dragging"""
    bl_idname = "view3d.modeling_cloth_drag"
    bl_label = "Modeling Cloth Drag"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        data = context.window_manager.modeling_cloth_data_set
        return context.space_data.type == 'VIEW_3D' and any(data)

    def __init__(self):
        self.clicked = False
        self.stored_mouse = None
        self.matrix = None

    def invoke(self, context, event):
        scene = context.scene
        #data = context.window_manager.modeling_cloth_data_set
        extra_data = context.window_manager.modeling_cloth_data_set_extra
        scene.mc.drag_alert = True
        bpy.ops.object.select_all(action='DESELECT')    

        extra_data['vidx'] = None # Vertex ids of dragged face
        extra_data['stored_vidx'] = None # Vertex coordinates
        extra_data['move'] = None # Direction of drag

        #for key, cloth in data.items():
        #    cloth.clicked = False
        for cp in scene.mc.cloth_pointers:
            if cp.object:
                cp.object.mc.clicked = False
            
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def main_drag(self, context, event):
        #print(self.clicked)
        # get the context arguments
        scene = context.scene
        extra_data = context.window_manager.modeling_cloth_data_set_extra
        region = context.region
        rv3d = context.region_data
        coord = event.mouse_region_x, event.mouse_region_y
    
        # get the ray from the viewport and mouse
        view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
    
        ray_target = ray_origin + view_vector

        if self.clicked:
    
            # cast rays and find the closest object
            best_length_squared = -1.0
            best_obj = None
            best_face_index = -1
            best_matrix = None
    
            for obj, matrix in visible_objects_and_duplis(context):
                hit, normal, face_index = obj_ray_cast(obj, matrix, ray_origin, ray_target)
                if hit:
                    hit_world = matrix * hit
                    length_squared = (hit_world - ray_origin).length_squared
    
                    if not best_obj or length_squared < best_length_squared:
                        best_length_squared = length_squared
                        best_obj = obj
                        best_face_index = face_index
                        best_matrix = matrix

            if best_obj:
                #data[best_obj.name].clicked = True
                #cloth = get_cloth_data(best_obj)
                #cloth.clicked = True
                best_obj.mc.clicked = True
                vidx = [v for v in best_obj.data.polygons[best_face_index].vertices]
                vert = best_obj.data.shape_keys.key_blocks['modeling cloth key'].data
                extra_data['vidx'] = vidx
                extra_data['stored_vidx'] = np.array([vert[v].co for v in extra_data['vidx']])
                self.stored_mouse = np.copy(ray_target)
                self.matrix = best_matrix.inverted()
                self.clicked = False
                        
        if self.stored_mouse is not None:
            move = np.array(ray_target) - self.stored_mouse
            extra_data['move'] = (move @ np.array(self.matrix)[:3, :3].T)
                   
    def modal(self, context, event):
        scene = context.scene
        #data = context.window_manager.modeling_cloth_data_set
        extra_data = context.window_manager.modeling_cloth_data_set_extra
        bpy.context.window.cursor_set("HAND")

        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            # allow navigation
            return {'PASS_THROUGH'}

        elif event.type == 'MOUSEMOVE':
            #pos = queryMousePosition()            
            self.main_drag(context, event)

        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            # when I click, If I have a hit, store the hit on press
            self.clicked = True
            extra_data['vidx'] = []
            
        elif event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            self.clicked = False
            self.stored_mouse = None
            extra_data['vidx'] = None
            #for key, cloth in data.items():
            #    cloth.clicked = False
            for cp in scene.mc.cloth_pointers:
                if cp.object:
                    cp.object.mc.clicked = False
            
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            self.clicked = False
            self.stored_mouse = None
            bpy.context.window.cursor_set("DEFAULT")
            scene.mc.drag_alert = False
            return {'FINISHED'}

        return {'RUNNING_MODAL'}


# drag===================================End
# drag===================================End



class DeletePins(bpy.types.Operator):
    """Delete modeling cloth pins and clear pin list for current object"""
    bl_idname = "object.delete_modeling_cloth_pins"
    bl_label = "Delete Modeling Cloth Pins"
    bl_options = {'REGISTER', 'UNDO'}    
    def execute(self, context):

        ob = get_last_object() # returns tuple with list and last cloth objects or None
        if not ob: return {'CANCELLED'}

        for i, pin in reversed(list(enumerate(ob[1].mc.pins))):
            bpy.data.objects.remove(pin.hook)
            ob[1].mc.pins.remove(i)

        bpy.context.scene.objects.active = ob[1]
        return {'FINISHED'}


class SelectPins(bpy.types.Operator):
    """Select modeling cloth pins for current object"""
    bl_idname = "object.select_modeling_cloth_pins"
    bl_label = "Select Modeling Cloth Pins"
    bl_options = {'REGISTER', 'UNDO'}    
    def execute(self, context):
        ob = get_last_object() # returns list and last cloth objects or None
        if not ob: return {'CANCELLED'}
        bpy.ops.object.select_all(action='DESELECT')
        for pin in ob[1].mc.pins:
            pin.hook.select = True

        return {'FINISHED'}


class PinSelected(bpy.types.Operator):
    """Add pins to verts selected in edit mode"""
    bl_idname = "object.modeling_cloth_pin_selected"
    bl_label = "Modeling Cloth Pin Selected"
    bl_options = {'REGISTER', 'UNDO'}    
    def execute(self, context):
        ob = bpy.context.object
        bpy.ops.object.mode_set(mode='OBJECT')
        sel = [i.index for i in ob.data.vertices if i.select]
                
        matrix = ob.matrix_world.copy()
        for v in sel:    
            e = bpy.data.objects.new('modeling_cloth_pin', None)
            bpy.context.scene.objects.link(e)
            if ob.active_shape_key is None:    
                closest = matrix * ob.data.vertices[v].co# * matrix
            else:
                closest = matrix * ob.active_shape_key.data[v].co# * matrix
            e.location = closest #* matrix
            e.show_x_ray = True
            e.select = True
            e.empty_draw_size = .1
            pin = ob.mc.pins.add()
            pin.vertex_id = v
            pin.hook = e
            ob.select = False
        bpy.ops.object.mode_set(mode='EDIT')       
        
        return {'FINISHED'}


#class UpdataPinWeights(bpy.types.Operator):
#    """Update Pin Weights"""
#    bl_idname = "object.modeling_cloth_update_pin_group"
#    bl_label = "Modeling Cloth Update Pin Weights"
#    bl_options = {'REGISTER', 'UNDO'}        
#    def execute(self, context):
#        update_pin_group()
#        return {'FINISHED'}


class GrowSource(bpy.types.Operator):
    """Grow Source Shape"""
    bl_idname = "object.modeling_cloth_grow"
    bl_label = "Modeling Cloth Grow"
    bl_options = {'REGISTER', 'UNDO'}        
    def execute(self, context):
        scale_source(1.02)
        return {'FINISHED'}


class ShrinkSource(bpy.types.Operator):
    """Shrink Source Shape"""
    bl_idname = "object.modeling_cloth_shrink"
    bl_label = "Modeling Cloth Shrink"
    bl_options = {'REGISTER', 'UNDO'}        
    def execute(self, context):
        scale_source(0.98)
        return {'FINISHED'}


class ResetShapes(bpy.types.Operator):
    """Reset Shapes"""
    bl_idname = "object.modeling_cloth_reset"
    bl_label = "Modeling Cloth Reset"
    bl_options = {'REGISTER', 'UNDO'}        
    def execute(self, context):
        reset_shapes()
        return {'FINISHED'}


class AddVirtualSprings(bpy.types.Operator):
    """Add Virtual Springs Between All Selected Vertices"""
    bl_idname = "object.modeling_cloth_add_virtual_spring"
    bl_label = "Modeling Cloth Add Virtual Spring"
    bl_options = {'REGISTER', 'UNDO'}        
    def execute(self, context):
        add_remove_virtual_springs()
        return {'FINISHED'}


class RemoveVirtualSprings(bpy.types.Operator):
    """Remove Virtual Springs Between All Selected Vertices"""
    bl_idname = "object.modeling_cloth_remove_virtual_spring"
    bl_label = "Modeling Cloth Remove Virtual Spring"
    bl_options = {'REGISTER', 'UNDO'}        
    def execute(self, context):
        add_remove_virtual_springs(remove=True)
        return {'FINISHED'}

# =================  Save & Load  ======================

# TODO:
# - Prevent handler to loop through all objects if data is empty V (already not happening)
# - Make sure to save collider class
# - Add option to not save dump
# - Try to use CollectionProperty to track cloth objects
# - Make all handlers persistent (need CollectionProperty implementation done)

#DATA_DUMP_TEXT_NAME = '__modeling_cloth_data_dump_'
#EXTRA_DATA_DUMP_TEXT_NAME = '__modeling_cloth_extra_data_dump_'

# Blender type cannot natively pickled, so encode them in certain way
#def encode_blender_type(key, val):
#    if type(val) == bpy.types.Object:
#        key = key + '___bpy.types.Object'
#        val = val.name
#    # Mathutil vector and matrix cannot be pickled
#    elif type(val) == Vector:
#        key = key + '___Vector'
#        val = val.to_tuple()
#    elif type(val) == Matrix:
#        mat_copy = []
#        for e in val:
#            mat_copy.append(e.to_tuple())
#        key = key + '___Matrix' 
#        val = mat_copy
#
#    return key, val
#
#def decode_blender_type(key, val):
#    if '___' in key:
#        splits = key.split('___')
#        key = splits[0]
#        val_type = splits[1]
#
#        if val_type == 'bpy.types.Object':
#            val = bpy.data.objects.get(val)
#        elif val_type == 'Vector':
#            val = Vector(val)
#        elif val_type == 'Matrix':
#            val = Matrix(val)
#
#        elif val_type == dict:
#            for k, v in val.items():
#                if type(v) == ColliderData:
#                    pass
#
#    return key, val
#
#def create_data_copy():
#    data = bpy.context.window_manager.modeling_cloth_data_set
#
#    data_copy = {}
#    for ob_name, cloth in data.items():
#        # Check if object still exists
#        ob = bpy.data.objects.get(ob_name)
#        if not ob: continue
#        
#        # Create cloth object copy
#        cloth_copy = {}
#        for attr in dir(cloth):
#            if attr.startswith('__'): continue
#            val = getattr(cloth, attr)
#            key, val = encode_blender_type(attr, val)
#            cloth_copy[key] = val
#        data_copy[ob_name] = cloth_copy
#
#    return data_copy
#
#def create_extra_data_copy():
#    extra_data = bpy.context.window_manager.modeling_cloth_data_set_extra
#    extra_data_copy = {}
#
#    for key, val in extra_data.items():
#        key, val = encode_blender_type(key, val)
#        extra_data_copy[key] = val
#
#    return extra_data_copy
#
#@persistent
#def save_cloth_data_set(scene):
#    #if len(data.items()) == 0: 
#    #    return
#
#    # Create a data copy because it can't be pickled directly
#    data_copy = create_data_copy()
#    extra_data_copy = create_extra_data_copy()
#
#    if not data_copy: return
#
#    # Dump data to bytes
#    data_dumped = pickle.dumps(data_copy)
#    extra_data_dumped = pickle.dumps(extra_data_copy)
#
#    # Encode them to string
#    data_dumped_str = codecs.encode(data_dumped, "base64").decode()
#    extra_data_dumped_str = codecs.encode(extra_data_dumped, "base64").decode()
#
#    # Get or create text
#    data_text = bpy.data.texts.get(DATA_DUMP_TEXT_NAME)
#    if not data_text: 
#        data_text = bpy.data.texts.new(DATA_DUMP_TEXT_NAME)
#    else: data_text.clear()
#
#    extra_data_text = bpy.data.texts.get(EXTRA_DATA_DUMP_TEXT_NAME)
#    if not extra_data_text: 
#        extra_data_text = bpy.data.texts.new(EXTRA_DATA_DUMP_TEXT_NAME)
#    else: extra_data_text.clear()
#
#    # Write it
#    data_text.write(data_dumped_str)
#    extra_data_text.write(extra_data_dumped_str)

#@persistent
#def load_cloth_data_set(scene):
#    # Reset datasets
#    data = bpy.types.Scene.modeling_cloth_data_set = {} 
#    extra_data = bpy.types.Scene.modeling_cloth_data_set_extra = {} 
#
#    data_text = bpy.data.texts.get(DATA_DUMP_TEXT_NAME)
#    extra_data_text = bpy.data.texts.get(EXTRA_DATA_DUMP_TEXT_NAME)
#
#    if data_text:
#
#        data_dumped = codecs.decode(data_text.as_string().encode(), "base64")
#        data_loaded = pickle.loads(data_dumped)
#        for ob_name, cloth_copy in data_loaded.items():
#            cloth = data[ob_name] = Cloth()
#            for key, val in cloth_copy.items():
#                key, val = decode_blender_type(key, val)
#                setattr(cloth, key, val)
#
#        # Delete text
#        bpy.data.texts.remove(data_text)
#
#    if extra_data_text:
#        extra_data_dumped = codecs.decode(extra_data_text.as_string().encode(), "base64")
#        extra_data_loaded = pickle.loads(extra_data_dumped)
#        for key, val in extra_data_loaded.items():
#            key, val = decode_blender_type(key, val)
#            extra_data[key] = val
#
#        # Delete text
#        bpy.data.texts.remove(extra_data_text)
#
#    # Add handlers
#    #for ob in bpy.data.objects:
#    #    if ob.mc.frame_update:
#    #        if not handler_frame in bpy.app.handlers.frame_change_post:
#    #            bpy.app.handlers.frame_change_post.append(handler_frame)
#    #    elif ob.mc.scene_update:
#    #        if not handler_scene in bpy.app.handlers.frame_change_post:
#    #            bpy.app.handlers.frame_change_post.append(handler_scene)

@persistent
def refresh_cloth_data(scene):
    scene = bpy.context.scene
    for cp in scene.mc.cloth_pointers:
        if cp.object:
            create_cloth_data(cp.object)

    for cp in scene.mc.collider_pointers:
        if cp.object:
            create_collider_data(cp.object)

class ModelingClothObject(bpy.types.PropertyGroup):
    object = PointerProperty(type=bpy.types.Object)

class ModelingClothCollider(bpy.types.PropertyGroup):
    object = PointerProperty(type=bpy.types.Object)

class ModelingClothGlobals(bpy.types.PropertyGroup):

    cloth_pointers = CollectionProperty(
            name="Modeling Cloth Objects", 
            description = 'List of cloth objects for quick pointers',
            type=ModelingClothObject)

    collider_pointers = CollectionProperty(
            name="Modeling Cloth Colliders", 
            description = 'List of collider objects for quick pointers',
            type=ModelingClothCollider)

    #extras = CollectionProperty(name="Modeling Cloth Data", type=)
    drag_alert = BoolProperty(default=False)
    pin_alert = BoolProperty(default=False)
    last_object = PointerProperty(type=bpy.types.Object)

class ModelingClothPin(bpy.types.PropertyGroup):
    vertex_id = IntProperty(default=-1)
    hook = PointerProperty(type=bpy.types.Object)

class ModelingClothVirtualSpring(bpy.types.PropertyGroup):
    vertex_id_1 = IntProperty(default=-1)
    vertex_id_2 = IntProperty(default=-1)

class ModelingClothObjectProps(bpy.types.PropertyGroup):

    enable = BoolProperty(name="Enable Modeling Cloth", 
        description="For toggling modeling cloth", 
        default=False, update=enable_cloth)

    floor = BoolProperty(name="Modeling Cloth Floor", 
        description="Stop at floor", 
        default=False)

    # handler type ----->>>        
    scene_update = BoolProperty(name="Modeling Cloth Continuous Update", 
        description="Choose continuous update", 
        default=False, update=manage_continuous_handler)        

    frame_update = BoolProperty(name="Modeling Cloth Handler Animation Update", 
        description="Choose animation update", 
        default=False, update=manage_animation_handler)
        
    auto_reset = BoolProperty(name="Modeling Cloth Reset at Frame 1", 
        description="Automatically reset if the current frame number is 1 or less", 
        default=False)#, update=manage_handlers)        
    # ------------------>>>

    noise = FloatProperty(name="Modeling Cloth Noise", 
        description="Set the noise strength", 
        default=0.001, precision=4, min=0, max=1, update=refresh_noise)

    noise_decay = FloatProperty(name="Modeling Cloth Noise Decay", 
        description="Multiply the noise by this value each iteration", 
        default=0.99, precision=4, min=0, max=1)#, update=refresh_noise_decay)

    # spring forces ------------>>>
    spring_force = FloatProperty(name="Modeling Cloth Spring Force", 
        description="Set the spring force", 
        default=1, precision=4, min=0, max=1.5)#, update=refresh_noise)

    push_springs = FloatProperty(name="Modeling Cloth Spring Force", 
        description="Set the spring force", 
        default=0.2, precision=4, min=0, max=1.5)#, update=refresh_noise)
    # -------------------------->>>

    gravity = FloatProperty(name="Modeling Cloth Gravity", 
        description="Modeling cloth gravity", 
        default=0.0, precision=4, min= -10, max=10)#, update=refresh_noise_decay)

    iterations = IntProperty(name="Iterations", 
        description="How stiff the cloth is", 
        default=2, min=1, max=500)#, update=refresh_noise_decay)

    velocity = FloatProperty(name="Velocity", 
        description="Cloth keeps moving", 
        default=.9, min= -1.1, max=1.1, soft_min= -1, soft_max=1)#, update=refresh_noise_decay)

    # Wind. Note, wind should be measured agains normal and be at zero when normals are at zero. Squared should work
    wind_x = FloatProperty(name="Wind X", 
        description="Not the window cleaner", 
        default=0, min= -1, max=1, soft_min= -10, soft_max=10)#, update=refresh_noise_decay)

    wind_y = FloatProperty(name="Wind Y", 
        description="Because wind is cool", 
        default=0, min= -1, max=1, soft_min= -10, soft_max=10)#, update=refresh_noise_decay)

    wind_z = FloatProperty(name="Wind Z", 
        description="It's windzee outzide", 
        default=0, min= -1, max=1, soft_min= -10, soft_max=10)#, update=refresh_noise_decay)

    turbulence = FloatProperty(name="Wind Turbulence", 
        description="Add Randomness to wind", 
        default=0, min=0, max=1, soft_min= -10, soft_max=10)#, update=refresh_noise_decay)

    # self collision ----->>>
    self_collision = BoolProperty(name="Modeling Cloth Self Collsion", 
        description="Toggle self collision", 
        default=False, update=collision_data_update)

    self_collision_force = FloatProperty(name="recovery force", 
        description="Self colide faces repel", 
        default=.17, precision=4, min= -1.1, max=1.1, soft_min= 0, soft_max=1)

    self_collision_margin = FloatProperty(name="Margin", 
        description="Self colide faces margin", 
        default=.08, precision=4, min= -1, max=1, soft_min= 0, soft_max=1)

    self_collision_cy_size = FloatProperty(name="Cylinder size", 
        description="Self colide faces cylinder size", 
        default=1, precision=4, min= 0, max=4, soft_min= 0, soft_max=1.5)
    # ---------------------->>>

    # extras ------->>>
    inflate = FloatProperty(name="inflate", 
        description="add force to vertex normals", 
        default=0, precision=4, min= -10, max=10, soft_min= -1, soft_max=1)

    sew = FloatProperty(name="sew", 
        description="add force to vertex normals", 
        default=0, precision=4, min= -10, max=10, soft_min= -1, soft_max=1)
    # -------------->>>

    # external collisions ------->>>
    object_collision = BoolProperty(name="Modeling Cloth Self Collsion", 
        description="Detect and collide with this object", 
        default=False, update=collision_object_update)
    
    object_collision_detect = BoolProperty(name="Modeling Cloth Self Collsion", 
        description="Detect collision objects", 
        default=False, update=cloth_object_update)    

    object_collision_outer_margin = FloatProperty(name="Modeling Cloth Outer Margin", 
        description="Collision margin on positive normal side of face", 
        default=0.04, precision=4, min=0, max=100, soft_min=0, soft_max=1000)
        
    object_collision_inner_margin = FloatProperty(name="Modeling Cloth Inner Margin", 
        description="Collision margin on negative normal side of face", 
        default=0.1, precision=4, min=0, max=100, soft_min=0, soft_max=1000)        

    # Not for manual editing ----->>>        
    
    waiting = BoolProperty(name='Pause Cloth Update',
            default=False)

    clicked = BoolProperty(name='Click for drag event',
            default=False)

    #force_update = BoolProperty(name='Force cloth update on loop',
    #        default = False)

    pins = CollectionProperty(name="Modeling Cloth Pins", 
            type=ModelingClothPin)

    virtual_springs = CollectionProperty(name="Modeling Cloth Virtual Springs", 
            type=ModelingClothVirtualSpring)

    # ---------------------------->>>
    

def create_properties():            

    bpy.types.Scene.mc = PointerProperty(type=ModelingClothGlobals)
    bpy.types.Object.mc = PointerProperty(type=ModelingClothObjectProps)

    # property dictionaries
    bpy.types.WindowManager.modeling_cloth_data_set = {} 
    bpy.types.WindowManager.modeling_cloth_data_set_colliders = {}
    bpy.types.WindowManager.modeling_cloth_data_set_extra = {} 
    
        
def remove_properties():            
    '''Drives to the grocery store and buys a sandwich'''
    # No need to remove properties because yolo


class ModelingClothPanel(bpy.types.Panel):
    """Modeling Cloth Panel"""
    bl_label = "Modeling Cloth Panel"
    bl_idname = "Modeling Cloth"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'
    bl_category = "Extended Tools"
    #gt_show = True
    
    def draw(self, context):
        scene = context.scene
        status = False
        layout = self.layout
        col = layout.column(align=True)
        col.label(text="Modeling Cloth")
        ob = bpy.context.object
        cloths = [i for i in bpy.data.objects if i.mc.enable] # so we can select an empty and keep the settings menu up
        if len(cloths) > 0:
            status = scene.mc.pin_alert
            if ob is not None:
                if ob.type != 'MESH' or status:
                    ob = scene.mc.last_object

        if ob is not None:
            if ob.type == 'MESH':
                col.prop(ob.mc ,"enable", text="Modeling Cloth", icon='SURFACE_DATA')               
                #col = layout.column(align=True)
                
                if ob.mc.enable:
                    col.label('Active: ' + ob.name)
                    col = layout.column(align=True)
                    col.scale_y = 2.0

                    col = layout.column(align=True)
                    col.scale_y = 1.4
                    col.prop(ob.mc, "frame_update", text="Animation Update", icon="TRIA_RIGHT")
                    if ob.mc.frame_update:    
                        col.prop(ob.mc, "auto_reset", text="Frame 1 Reset")
                    col.prop(ob.mc, "scene_update", text="Continuous Update", icon="TIME")
                    col = layout.column(align=True)
                    col.scale_y = 2.0
                    col.operator("object.modeling_cloth_reset", text="Reset")
                    col.alert = scene.mc.drag_alert
                    col.operator("view3d.modeling_cloth_drag", text="Grab")
                    col = layout.column(align=True)
                        
                    col.prop(ob.mc ,"iterations", text="Iterations")#, icon='OUTLINER_OB_LATTICE')               
                    col.prop(ob.mc ,"spring_force", text="Stiffness")#, icon='OUTLINER_OB_LATTICE')               
                    col.prop(ob.mc ,"push_springs", text="Push Springs")#, icon='OUTLINER_OB_LATTICE')               
                    col.prop(ob.mc ,"noise", text="Noise")#, icon='PLAY')               
                    col.prop(ob.mc ,"noise_decay", text="Decay Noise")#, icon='PLAY')               
                    col.prop(ob.mc ,"gravity", text="Gravity")#, icon='PLAY')        
                    col.prop(ob.mc ,"inflate", text="Inflate")#, icon='PLAY')        
                    col.prop(ob.mc ,"sew", text="Sew Force")#, icon='PLAY')        
                    col.prop(ob.mc ,"velocity", text="Velocity")#, icon='PLAY')        
                    col = layout.column(align=True)
                    col.label("Wind")                
                    col.prop(ob.mc ,"wind_x", text="Wind X")#, icon='PLAY')        
                    col.prop(ob.mc ,"wind_y", text="Wind Y")#, icon='PLAY')        
                    col.prop(ob.mc ,"wind_z", text="Wind Z")#, icon='PLAY')        
                    col.prop(ob.mc ,"turbulence", text="Turbulence")#, icon='PLAY')        
                    col.prop(ob.mc ,"floor", text="Floor")#, icon='PLAY')        
                    col = layout.column(align=True)
                    col.scale_y = 1.5
                    col.alert = status
                    if ob.mc.enable:    
                        if ob.mode == 'EDIT':
                            col.operator("object.modeling_cloth_pin_selected", text="Pin Selected")
                            col = layout.column(align=True)
                            col.operator("object.modeling_cloth_add_virtual_spring", text="Add Virtual Springs")
                            col.operator("object.modeling_cloth_remove_virtual_spring", text="Remove Selected")
                        else:
                            col.operator("view3d.modeling_cloth_pin", text="Create Pins")
                        col = layout.column(align=True)
                        col.operator("object.select_modeling_cloth_pins", text="Select Pins")
                        col.operator("object.delete_modeling_cloth_pins", text="Delete Pins")
                        col.operator("object.modeling_cloth_grow", text="Grow Source")
                        col.operator("object.modeling_cloth_shrink", text="Shrink Source")
                        col = layout.column(align=True)
                        col.prop(ob.mc ,"self_collision", text="Self Collision")#, icon='PLAY')        
                        col.prop(ob.mc ,"self_collision_force", text="Repel")#, icon='PLAY')        
                        col.prop(ob.mc ,"self_collision_margin", text="Margin")#, icon='PLAY')        
                        col.prop(ob.mc ,"self_collision_cy_size", text="Cylinder Size")#, icon='PLAY')        

                # object collisions
                col = layout.column(align=True)
                col.label("Collisions")
                if ob.mc.enable:    
                    col.prop(ob.mc ,"object_collision_detect", text="Object Collisions", icon="PHYSICS")

                col.prop(ob.mc ,"object_collision", text="Collider", icon="STYLUS_PRESSURE")
                if ob.mc.object_collision:    
                    col.prop(ob.mc ,"object_collision_outer_margin", text="Outer Margin", icon="FORCE_FORCE")
                    col.prop(ob.mc ,"object_collision_inner_margin", text="Inner Margin", icon="STICKY_UVS_LOC")
                
                col.label("Collide List:")
                #colliders = [i.name for i in bpy.data.objects if i.mc.object_collision]
                #for i in colliders:
                #    col.label(i)
                for cp in scene.mc.collider_pointers:
                    if cp.object:
                        col.label(cp.object.name)
                    
                # =============================
                col = layout.column(align=True)
                col.label('Collision Series')
                col.operator("object.modeling_cloth_collision_series", text="Paperback")
                col.operator("object.modeling_cloth_collision_series_kindle", text="Kindle")
                col.operator("object.modeling_cloth_donate", text="Donate")


class CollisionSeries(bpy.types.Operator):
    """Support my addons by checking out my awesome sci fi books"""
    bl_idname = "object.modeling_cloth_collision_series"
    bl_label = "Modeling Cloth Collision Series"
        
    def execute(self, context):
        collision_series()
        return {'FINISHED'}


class CollisionSeriesKindle(bpy.types.Operator):
    """Support my addons by checking out my awesome sci fi books"""
    bl_idname = "object.modeling_cloth_collision_series_kindle"
    bl_label = "Modeling Cloth Collision Series Kindle"
        
    def execute(self, context):
        collision_series(False)
        return {'FINISHED'}


class Donate(bpy.types.Operator):
    """Support my addons by donating"""
    bl_idname = "object.modeling_cloth_donate"
    bl_label = "Modeling Cloth Donate"

        
    def execute(self, context):
        collision_series(False, False)
        self.report({'INFO'}, 'Paypal, The3dAdvantage@gmail.com')
        return {'FINISHED'}


def collision_series(paperback=True, kindle=True):
    import webbrowser
    import imp
    if paperback:    
        webbrowser.open("https://www.createspace.com/6043857")
        imp.reload(webbrowser)
        webbrowser.open("https://www.createspace.com/7164863")
        return
    if kindle:
        webbrowser.open("https://www.amazon.com/Resolve-Immortal-Flesh-Collision-Book-ebook/dp/B01CO3MBVQ")
        imp.reload(webbrowser)
        webbrowser.open("https://www.amazon.com/Formulacrum-Collision-Book-Rich-Colburn-ebook/dp/B0711P744G")
        return
    webbrowser.open("https://www.paypal.com/donate/?token=G1UymFn4CP8lSFn1r63jf_XOHAuSBfQJWFj9xjW9kWCScqkfYUCdTzP-ywiHIxHxYe7uJW&country.x=US&locale.x=US")

# ============================================================================================    


def register():
    create_properties()

    # Register all classes if this file loaded individually
    if __name__ in {'__main__', 'ModelingCloth'}:
        bpy.utils.register_module(__name__)

    # Main handlers
    bpy.app.handlers.frame_change_post.append(handler_frame)
    bpy.app.handlers.scene_update_post.append(handler_scene)

    # Add save & load handlers
    #bpy.app.handlers.save_pre.append(save_cloth_data_set)
    #bpy.app.handlers.load_post.append(load_cloth_data_set)
    bpy.app.handlers.load_post.append(refresh_cloth_data)


def unregister():
    # Remove save & load handlers
    #bpy.app.handlers.save_pre.remove(save_cloth_data_set)
    #bpy.app.handlers.load_post.remove(load_cloth_data_set)
    bpy.app.handlers.load_post.remove(refresh_cloth_data)

    # Remove main handlers
    bpy.app.handlers.frame_change_post.remove(handler_frame)
    bpy.app.handlers.scene_update_post.remove(handler_scene)

    remove_properties()

    # Unregister all classes if this file loaded individually
    if __name__ in {'__main__', 'ModelingCloth'}:
        bpy.utils.unregister_module(__name__)
    
if __name__ == "__main__":
    register()

    # testing!!!!!!!!!!!!!!!!
    #generate_collision_data(bpy.context.object)
    # testing!!!!!!!!!!!!!!!!


    
    for i in bpy.data.objects:
        i.mc.enable = False
        i.mc.object_collision = False
        
    for i in bpy.app.handlers.frame_change_post:
        if i.__name__ == 'handler_frame':
            bpy.app.handlers.frame_change_post.remove(i)
            
    for i in bpy.app.handlers.scene_update_post:
        if i.__name__ == 'handler_scene':
            bpy.app.handlers.scene_update_post.remove(i)            
