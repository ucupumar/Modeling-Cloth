# You are at the top. If you attempt to go any higher
#   you will go beyond the known limits of the code
#   universe where there are most certainly monsters

# might be able to get a speedup where I'm appending move and -move

# to do:
#  use point raycaster to make a cloth_wrap option
#  self colisions
    # maybe do dynamic margins for when cloth is moving fast
#  object collisions
    # collisions need to properly exclude pinned and vertex pinned
#  add bending springs
#  add curl by shortening bending springs on one axis or diagonal
#  independantly scale bending springs and structural to create buckling
#  option to cache animation?
#  Custom Source shape option for animated shapes


# collisions:
# Only need to check one of the edges for groups connected to a vertex    
# for edge to face intersections...
# figure out where the edge hit the face
# figure out which end of the edge is inside the face
# move along the face normal to the surface for the point inside.
# if I reflect by flipping the vel around the face normal
#   if it collides on the bounce it will get caught on the next iteration

# Sewing
# Could create super sewing that doesn't use edges but uses scalars along the edge to place virtual points
#   sort of a barycentric virtual spring. Could even use it to sew to faces if I can think of a ui for where on the face.


'''??? Would it make sense to do self collisions with virtual edges ???'''
'''??? Could do dynamic collision margins for stuff moving fast ???'''




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
import time, sys

#enable_numexpr = True
enable_numexpr = False
if enable_numexpr:
    import numexpr as ne

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


def get_proxy_co(ob, arr, me):
    """Returns vertex coords with modifier effects as N x 3"""
    if arr is None:
        arr = np.zeros(len(me.vertices) * 3, dtype=np.float32)
        arr.shape = (arr.shape[0] //3, 3)    
    c = arr.shape[0]
    me.vertices.foreach_get('co', arr.ravel())
    arr.shape = (c, 3)
    return arr


def triangulate(ob, me):
    """Requires a mesh. Returns an index array for viewing co as triangles"""
    obm = bmesh.new()
    obm.from_mesh(me)        
    bmesh.ops.triangulate(obm, faces=obm.faces)
    #obm.to_mesh(me)        
    count = len(obm.faces)    
    #tri_idx = np.zeros(count * 3, dtype=np.int32)        
    #me.polygons.foreach_get('vertices', tri_idx)
    tri_idx = np.array([[v.index for v in f.verts] for f in obm.faces])
    
    obm.free()
    
    return tri_idx#.reshape(count, 3)


def tri_normals_in_place(col, tri_co):    
    """Takes N x 3 x 3 set of 3d triangles and 
    returns non-unit normals and origins"""
    col.origins = tri_co[:,0]
    col.cross_vecs = tri_co[:,1:] - col.origins[:, nax]
    col.normals = np.cross(col.cross_vecs[:,0], col.cross_vecs[:,1])
    col.nor_dots = np.einsum("ij, ij->i", col.normals, col.normals)
    col.normals /= np.sqrt(col.nor_dots)[:, nax]


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


def proxy_in_place(col, me):
    """Overwrite vert coords with modifiers in world space"""
    me.vertices.foreach_get('co', col.co.ravel())
    col.co = apply_transforms(col.ob, col.co)


def apply_rotation(col):
    """When applying vectors such as normals we only need
    to rotate"""
    m = np.array(col.ob.matrix_world)
    mat = m[:3, :3].T
    col.v_normals = col.v_normals @ mat
    

def proxy_v_normals_in_place(col, world=True, me=None):
    """Overwrite vert coords with modifiers in world space"""
    me.vertices.foreach_get('normal', col.v_normals.ravel())
    if world:    
        apply_rotation(col)


def proxy_v_normals(ob, me):
    """Overwrite vert coords with modifiers in world space"""
    arr = np.zeros(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get('normal', arr)
    arr.shape = (arr.shape[0] //3, 3)
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
    cloths = [i for i in bpy.data.objects if i.mclo.enable] # so we can select an empty and keep the settings menu up
    if bpy.context.object.mclo.enable:
        return cloths, bpy.context.object
    
    if len(cloths) > 0:
        ob = bpy.context.scene.mclo.last_object
        return cloths, ob
    return None, None


def get_poly_centers(ob, type=np.float32, mesh=None):
    mod = False
    m_count = len(ob.modifiers)
    if m_count > 0:
        show = np.zeros(m_count, dtype=np.bool)
        ren_set = np.copy(show)
        ob.modifiers.foreach_get('show_render', show)
        ob.modifiers.foreach_set('show_render', ren_set)
        mod = True
    p_count = len(mesh.polygons)
    center = np.zeros(p_count * 3, dtype=type)
    mesh.polygons.foreach_get('center', center)
    center.shape = (p_count, 3)
    if mod:
        ob.modifiers.foreach_set('show_render', show)

    return center


def simple_poly_centers(ob, key=None):
    if key is not None:
        s_key = ob.data.shape_keys.key_blocks[key].data
        return np.squeeze([[np.mean([ob.data.vertices[i].co for i in p.vertices], axis=0)] for p in ob.data.polygons])


def get_poly_normals(ob, type=np.float32, mesh=None):
    mod = False
    m_count = len(ob.modifiers)
    if m_count > 0:
        show = np.zeros(m_count, dtype=np.bool)
        ren_set = np.copy(show)
        ob.modifiers.foreach_get('show_render', show)
        ob.modifiers.foreach_set('show_render', ren_set)
        mod = True
    p_count = len(mesh.polygons)
    normal = np.zeros(p_count * 3, dtype=type)
    mesh.polygons.foreach_get('normal', normal)
    normal.shape = (p_count, 3)
    if mod:
        ob.modifiers.foreach_set('show_render', show)

    return normal


def get_v_normals(ob, arr, mesh):
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
    #v_count = len(mesh.vertices)
    #normal = np.zeros(v_count * 3)#, dtype=type)
    mesh.vertices.foreach_get('normal', arr.ravel())
    #normal.shape = (v_count, 3)
    if mod:
        ob.modifiers.foreach_set('show_render', show)


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
    
    
    
    # so if I have a vertex with one or more sew edges attached
    # I need to get the mean location of all verts shared by those edges
    # every one of those verts needs to move towards the total mean
    
    
    # get linear edges
    e_count = len(obm.edges)
    eidx = np.zeros(e_count * 2, dtype=np.int32)
    e_bool = np.zeros(e_count, dtype=np.bool)
    e_bool[sew] = True
    ob.data.edges.foreach_get('vertices', eidx)
    eidx.shape = (e_count, 2)

    # get diagonal edges:
    diag_eidx = []
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
    lin_eidx = eidx[~e_bool]
    diag_eidx = np.array(diag_eidx)
    
    # deal with sew verts connected to more than one edge
    s_t_rav = sew_eidx.T.ravel()
    s_uni, s_inv, s_counts = np.unique(s_t_rav,return_inverse=True, return_counts=True)
    s_multi = s_counts > 1
    
    multi_groups = None
    if np.any(s_counts):
        multi_groups = []
        ls = sew_eidx[:,0]
        rs = sew_eidx[:,1]
        
        for i in s_uni[s_multi]:
            print(ls[rs==i])
            gr = np.array([i])
            gr = np.append(gr, ls[rs==i])
            gr = np.append(gr, rs[ls==i])
            multi_groups.append(gr)
        
    return lin_eidx, diag_eidx, sew_eidx, multi_groups


def add_remove_virtual_springs(remove=False):
    ob = get_last_object()[1]
    cloth = get_cloth_data(ob)
    obm = get_bmesh()
    obm.verts.ensure_lookup_table()
    count = len(obm.verts)
    idxer = np.arange(count, dtype=np.int32)
    sel = np.array([v.select for v in obm.verts])    
    selected = idxer[sel]

    virtual_springs = np.array([[vs.vertex_id_1, vs.vertex_id_2] for vs in ob.mclo.virtual_springs])
    if virtual_springs.shape[0] == 0:
        virtual_springs.shape = (0, 2)

    if remove:
        ls = virtual_springs[:, 0]
        
        in_sel = np.in1d(ls, idxer[sel])

        deleter = np.arange(ls.shape[0], dtype=np.int32)[in_sel]

        for i in reversed(deleter):
            ob.mclo.virtual_springs.remove(i)

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
            new_vs = ob.mclo.virtual_springs.add()
            new_vs.vertex_id_1 = i
            new_vs.vertex_id_2 = sv

    # gets appended to eidx in the cloth_init function after calling get connected polys in case geometry changes


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
        if ob.mclo.enable:
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
        if bpy.context.object.mclo.enable:
            ob = bpy.context.object
        else:    
            ob = bpy.context.scene.mclo.last_object

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
    keys['Basis'].data.foreach_get('co', co)
    #co = applied_key_co(ob, None, 'modeling cloth source key')
    keys['modeling cloth source key'].data.foreach_set('co', co)
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
    
    return mix
        

def collision_data_update(self, context):
    ob = self.id_data
    if ob.mclo.self_collision:    
        create_cloth_data(ob)


def refresh_noise(self, context):
    ob = self.id_data
    cloth = get_cloth_data(ob)
    if cloth:
        zeros = np.zeros(cloth.count, dtype=np.float32)
        random = np.random.random(cloth.count)
        zeros[:] = random
        cloth.noise = ((zeros + -0.5) * ob.mclo.noise * 0.1)[:, nax]


def generate_wind(wind_vec, ob, cloth):
    """Maintains a wind array and adds it to the cloth vel"""    

    tri_nor = cloth.normals # non-unit calculated by tri_normals_in_place() per each triangle
    w_vec = revert_rotation(ob, wind_vec)

    turb = ob.mclo.turbulence    
    if turb != 0: 
        w_vec += np.random.random(3).astype(np.float32) * turb * np.mean(w_vec) * 4

    # only blow on verts facing the wind
    perp = np.abs(tri_nor @ w_vec)
    cloth.wind += w_vec
    cloth.wind *= perp[:, nax][:, nax]
    
    # reshape for add.at
    shape = cloth.wind.shape
    cloth.wind.shape = (shape[0] * 3, 3)
    
    cloth.wind *= cloth.tri_mix
    np.add.at(cloth.vel, cloth.tridex.ravel(), cloth.wind)
    cloth.wind.shape = shape


def generate_inflate(ob, cloth):
    """Blow it up baby!"""    
    
    tri_nor = cloth.normals #* ob.mclo.inflate # non-unit calculated by tri_normals_in_place() per each triangle
    #tri_nor /= np.einsum("ij, ij->i", tri_nor, tri_nor)[:, nax]
    
    # reshape for add.at
    shape = cloth.inflate.shape
    
    cloth.inflate += tri_nor[:, nax] * ob.mclo.inflate# * cloth.tri_mix
    print(cloth.inflate.shape, "shape of cloth.inflate")
    print(cloth.mix.shape)
    
    
    cloth.inflate.shape = (shape[0] * 3, 3)
    cloth.inflate *= cloth.tri_mix


    np.add.at(cloth.vel, cloth.tridex.ravel(), cloth.inflate)
    cloth.inflate.shape = shape
    cloth.inflate *= 0


# sewing functions ---------------->>>
def create_sew_edges():

    bpy.ops.mesh.bridge_edge_loops()
    bpy.ops.mesh.delete(type='ONLY_FACE')
    return
    #highlight a sew edge
    #compare vertex counts
    #subdivide to match counts
    #distribute and smooth back into mesh
    #create sew lines
     

    



# sewing functions ---------------->>>

    
def check_and_get_pins_and_hooks(ob):
    scene = bpy.context.scene
    pins = []
    hooks = []
    cull_ids = []
    for i, pin in enumerate(ob.mclo.pins):
        # Check if hook object still exists
        if not pin.hook or (pin.hook and not scene.objects.get(pin.hook.name)):
            cull_ids.append(i)
        else:
            #vert = ob.data.vertices[pin.vertex_id]
            pins.append(pin.vertex_id)
            hooks.append(pin.hook)

    # Delete missing hooks pointers 
    for i in reversed(cull_ids):
        pin = ob.mclo.pins[i]
        if pin.hook:
            bpy.data.objects.remove(pin.hook)
        ob.mclo.pins.remove(i)

    return pins, hooks

        
class ClothData:
    pass


def create_cloth_data(ob):
    """Creates instance of cloth object with attributes needed for engine"""
    scene = bpy.context.scene
    data = scene.modeling_cloth_data_set

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

    # get proxy object
    #proxy = ob.to_mesh(bpy.context.scene, False, 'PREVIEW')
    # ----------------

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
    cloth.pin_bool = ~np.array([ob.vertex_groups['modeling_cloth_pin'].weight(i) for i in range(cloth.count)], dtype=np.bool)

    # unique edges------------>>>
    uni_edges = get_minimal_edges(ob)
    if len(uni_edges[1]) > 0:   
        cloth.eidx = np.append(uni_edges[0], uni_edges[1], axis=0)
    else:
        cloth.eidx = uni_edges[0]
    #cloth.eidx = uni_edges[0][0]

    if len(ob.mclo.virtual_springs) > 0:
        virtual_springs = np.array([[vs.vertex_id_1, vs.vertex_id_2] for vs in ob.mclo.virtual_springs])
        cloth.eidx = np.append(cloth.eidx, virtual_springs, axis=0)
    cloth.eidx_tiler = cloth.eidx.T.ravel()    

    mixology = get_spring_mix(ob, cloth.eidx)


    #eidx1 = np.copy(cloth.eidx)
    cloth.pindexer = np.arange(cloth.count, dtype=np.int32)[cloth.pin_bool]
    cloth.unpinned = np.in1d(cloth.eidx_tiler, cloth.pindexer)
    cloth.eidx_tiler = cloth.eidx_tiler[cloth.unpinned]    

    cloth.sew_edges = uni_edges[2]
    cloth.multi_sew = uni_edges[3]
    
    # unique edges------------>>>

    #cloth.pcount = pindexer.shape[0]

    cloth.sco = np.zeros(cloth.count * 3, dtype=np.float32)
    ob.data.shape_keys.key_blocks['modeling cloth source key'].data.foreach_get('co', cloth.sco)
    cloth.sco.shape = (cloth.count, 3)
    cloth.co = np.zeros(cloth.count * 3, dtype=np.float32)
    ob.data.shape_keys.key_blocks['modeling cloth key'].data.foreach_get('co', cloth.co)
    cloth.co.shape = (cloth.count, 3)    
    co = cloth.co
    cloth.vel = np.zeros(cloth.count * 3, dtype=np.float32)
    cloth.vel.shape = (cloth.count, 3)
    cloth.vel_start = np.zeros(cloth.count * 3, dtype=np.float32)
    cloth.vel_start.shape = (cloth.count, 3)
    cloth.self_col_vel = np.copy(co)
    
    cloth.v_normals = np.zeros(co.shape, dtype=np.float32)
    #get_v_normals(ob, cloth.v_normals, proxy)

    #noise---
    noise_zeros = np.zeros(cloth.count, dtype=np.float32)
    random = np.random.random(cloth.count).astype(np.float32)
    noise_zeros[:] = random
    cloth.noise = ((noise_zeros + -0.5) * ob.mclo.noise * 0.1)[:, nax]
    
    #cloth.waiting = False
    #cloth.clicked = False # for the grab tool
    
    # this helps with extra springs behaving as if they had more mass---->>>
    cloth.mix = mixology[cloth.unpinned][:, nax]
    # -------------->>>

    # new self collisions:
    cloth.tridex = triangulate(ob, ob.data)
    cloth.tridexer = np.arange(cloth.tridex.shape[0], dtype=np.int32)
    cloth.tri_co = cloth.co[cloth.tridex]
    tri_normals_in_place(cloth, cloth.tri_co) # non-unit normals
    # -------------->>>
    
    tri_uni, tri_inv, tri_counts = np.unique(cloth.tridex, return_inverse=True, return_counts=True)
    cloth.tri_mix = (1 / tri_counts[tri_inv])[:, nax]
    
    cloth.wind = np.zeros(cloth.tri_co.shape, dtype=np.float32)
    cloth.inflate = np.zeros(cloth.tri_co.shape, dtype=np.float32)

    bpy.ops.object.mode_set(mode=mode)

    # remove proxy
    #bpy.data.meshes.remove(proxy)

    print('INFO: Cloth data for', ob.name, 'is created!')
    return cloth


def run_handler(ob, cloth):
    T = time.time()
    scene = bpy.context.scene
    extra_data = scene.modeling_cloth_data_set_extra
    col_data = scene.modeling_cloth_data_set_colliders

    if not ob.mclo.waiting and ob.mode != 'OBJECT':
        ob.mclo.waiting = True

    if ob.mclo.waiting:    
        if ob.mode == 'OBJECT':
            create_cloth_data(ob)
            ob.mclo.waiting = False

    if not ob.mclo.waiting:
        eidx = cloth.eidx # world's most important variable
        ob.data.shape_keys.key_blocks['modeling cloth source key'].data.foreach_get('co', cloth.sco.ravel())
        
        sco = cloth.sco
        co = cloth.co

        svecs = sco[eidx[:, 1]] - sco[eidx[:, 0]]
        sdots = np.einsum('ij,ij->i', svecs, svecs)

        co[cloth.pindexer] += cloth.noise[cloth.pindexer]
        #co += cloth.noise
        cloth.noise *= ob.mclo.noise_decay

        # mix in vel before collisions and sewing
        co[cloth.pindexer] += cloth.vel[cloth.pindexer]
        cloth.vel_start[:] = co


        force = ob.mclo.spring_force
        mix = cloth.mix * force

        pin_list = []
        if len(ob.mclo.pins) > 0:
            pin_list, hook_list = check_and_get_pins_and_hooks(ob)
            hook_co = np.array([ob.matrix_world.inverted() * hook.matrix_world.to_translation() 
                for hook in hook_list])

        for x in range(ob.mclo.iterations):    
            # add pull
            vecs = co[eidx[:, 1]] - co[eidx[:, 0]]
            dots = np.einsum('ij,ij->i', vecs, vecs)
            div = np.nan_to_num(sdots / dots)
            swap = vecs * np.sqrt(div)[:, nax]
            move = vecs - swap

            # pull separate test--->>>
            push = ob.mclo.push_springs
            if push == 0:
                move[div > 1] = 0
            else:
                move[div > 1] *= push
            # pull only test--->>>

            tiled_move = np.append(move, -move, axis=0)[cloth.unpinned] * mix # * mix for stability: force multiplied by 1/number of springs

            np.add.at(cloth.co, cloth.eidx_tiler, tiled_move)

            if pin_list:
                cloth.co[pin_list] = hook_co
            
            # grab inside spring iterations
            if ob.mclo.clicked: # for the grab tool
                cloth.co[extra_data['vidx']] = np.array(extra_data['stored_vidx']) + np.array(+ extra_data['move'])   

        spring_dif = cloth.co - cloth.vel_start
        grav = ob.mclo.gravity * (.01 / ob.mclo.iterations)
        cloth.vel += revert_rotation(ob, np.array([0, 0, grav]))

        # refresh normals for inflate wind and self collisions
        cloth.tri_co = cloth.co[cloth.tridex]
        tri_normals_in_place(cloth, cloth.tri_co) # unit normals
        # non-unit normals might be better for inflate and wind because
        # their strength is affected by the area as it is should be

        #place after wind and inflate unless those are added to vel after collisions
        if False:    
            if wind | inflate:
                cloth.tri_co = cloth.co[cloth.tridex]
                tri_normals_in_place(cloth, cloth.tri_co)                

        # get proxy object
        #proxy = ob.to_mesh(bpy.context.scene, False, 'PREVIEW')
        #proxy = ob.data
        #get_v_normals(ob, cloth.v_normals, proxy)


        # wind:
        x = ob.mclo.wind_x
        y = ob.mclo.wind_y
        z = ob.mclo.wind_z
        wind_vec = np.array([x,y,z])
        check_wind = wind_vec != 0
        if np.any(check_wind):
            generate_wind(wind_vec, ob, cloth)            

        # inflate
        inflate = ob.mclo.inflate
        if inflate != 0:
            generate_inflate(ob, cloth)
            #cloth.v_normals *= inflate
            #cloth.vel += cloth.v_normals

        # inextensible calc:

        ab_dot = np.einsum('ij, ij->i', cloth.vel, spring_dif)
        aa_dot = np.einsum('ij, ij->i', spring_dif, spring_dif)
        div = np.nan_to_num(ab_dot / aa_dot)
        cp = spring_dif * div[:, nax]
        cloth.vel -= np.nan_to_num(cp)
        cloth.vel += (spring_dif + cp)
        
        # !!! need to test if this should be added again here!!!
        cloth.vel += spring_dif        
        # !!! need to test if this should be added again here!!!

        # The amount of drag increases with speed. 
        # have to converto to a range between 0 and 1
        squared_move_dist = np.einsum("ij, ij->i", cloth.vel, cloth.vel)
        squared_move_dist += 1
        cloth.vel *= (1 / (squared_move_dist / ob.mclo.velocity))[:, nax]

        
        if ob.mclo.sew != 0:
            if len(cloth.sew_edges) > 0:
                sew_edges = cloth.sew_edges
                rs = co[sew_edges[:,1]]
                ls = co[sew_edges[:,0]]
                sew_vecs = (rs - ls) * 0.5 * ob.mclo.sew
                co[sew_edges[:,1]] -= sew_vecs
                co[sew_edges[:,0]] += sew_vecs

                # for sew verts with more than one sew edge
                if cloth.multi_sew is not None:
                    for sg in cloth.multi_sew:
                        cosg = co[sg]
                        meanie = np.mean(cosg, axis=0)
                        sg_vecs = meanie - cosg
                        co[sg] += sg_vecs * ob.mclo.sew

        # !!!!!  need to try adding in the velocity before doing the collision stuff
        # !!!!! so vel would be added here after wind and inflate but before collision

        
        # floor ---
        if ob.mclo.floor:    
            floored = cloth.co[:,2] < 0        
            cloth.vel[:,2][floored] *= -1
            cloth.vel[floored] *= .1
            cloth.co[:, 2][floored] = 0
        # floor ---            
        

        # objects ---
        #T = time.time()
        if ob.mclo.object_collision_detect:
            cull_ids = []
            for i, cp in enumerate(scene.mclo.collider_pointers):
                # Check if object is still exists
                if not cp.ob or (cp.ob and not scene.objects.get(cp.ob.name)):
                    cull_ids.append(i)
                    continue

                if cp.ob == ob:    
                    self_collide(ob)
                else:    
                    object_collide(ob, cp.ob)

            # Remove collider missing object from pointer list
            for i in reversed(cull_ids):
                o = scene.mclo.collider_pointers[i].ob
                if o:
                    o.mclo.object_collision = False
                else:
                    scene.mclo.collider_pointers.remove(i)

        #print(time.time()-T, "the whole enchalada")
        # objects ---


        
        if pin_list:
            cloth.co[pin_list] = hook_co
            cloth.vel[pin_list] = 0

        if ob.mclo.clicked: # for the grab tool
            cloth.co[extra_data['vidx']] = np.array(extra_data['stored_vidx']) + np.array(+ extra_data['move'])


        ob.data.shape_keys.key_blocks['modeling cloth key'].data.foreach_set('co', cloth.co.ravel())

        ob.data.shape_keys.key_blocks['modeling cloth key'].mute = True
        ob.data.shape_keys.key_blocks['modeling cloth key'].mute = False

        # remove proxy
        #proxy.user_clear()
        #bpy.data.meshes.remove(proxy)
        #del(proxy)
    print(time.time()-T, "the entire handler time")

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


def triangle_bounds_check(tri_co, co_min, co_max, idxer, fudge):
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


# -------------------------------------------------------
# -------------------------------------------------------
def zxy_grid(co_y, tymin, tymax, subs, c, t, c_peat, t_peat):
    # create linespace grid between bottom and top of tri z
    #subs = 7
    t_min = np.min(tymin)
    t_max = np.max(tymax)
    divs = np.linspace(t_min, t_max, num=subs, dtype=np.float32)            
    
    # figure out which triangles and which co are in each section
    co_bools = (co_y > divs[:-1][:, nax]) & (co_y < divs[1:][:, nax])
    tri_bools = (tymin < divs[1:][:, nax]) & (tymax > divs[:-1][:, nax])

    for i, j in zip(co_bools, tri_bools):
        if (np.sum(i) > 0) & (np.sum(j) > 0):
            c3 = c[i]
            t3 = t[j]
        
            c_peat.append(np.repeat(c3, t3.shape[0]))
            t_peat.append(np.tile(t3, c3.shape[0]))


def zx_grid(co_x, txmin, txmax, subs, c, t, c_peat, t_peat, co_y, tymin, tymax):
    # create linespace grid between bottom and top of tri z
    #subs = 7
    t_min = np.min(txmin)
    t_max = np.max(txmax)
    divs = np.linspace(t_min, t_max, num=subs, dtype=np.float32)            
    
    # figure out which triangles and which co are in each section
    co_bools = (co_x > divs[:-1][:, nax]) & (co_x < divs[1:][:, nax])
    tri_bools = (txmin < divs[1:][:, nax]) & (txmax > divs[:-1][:, nax])

    for i, j in zip(co_bools, tri_bools):
        if (np.sum(i) > 0) & (np.sum(j) > 0):
            c2 = c[i]
            t2 = t[j]
            
            zxy_grid(co_y[i], tymin[j], tymax[j], subs, c2, t2, c_peat, t_peat)


def z_grid(co_z, tzmin, tzmax, subs, co_x, txmin, txmax, co_y, tymin, tymax):
    # create linespace grid between bottom and top of tri z
    #subs = 7
    t_min = np.min(tzmin)
    t_max = np.max(tzmax)
    divs = np.linspace(t_min, t_max, num=subs, dtype=np.float32)
            
    # figure out which triangles and which co are in each section
    co_bools = (co_z > divs[:-1][:, nax]) & (co_z < divs[1:][:, nax])
    tri_bools = (tzmin < divs[1:][:, nax]) & (tzmax > divs[:-1][:, nax])

    c_ranger = np.arange(co_bools.shape[1])
    t_ranger = np.arange(tri_bools.shape[1])

    c_peat = []
    t_peat = []

    for i, j in zip(co_bools, tri_bools):
        if (np.sum(i) > 0) & (np.sum(j) > 0):
            c = c_ranger[i]
            t = t_ranger[j]

            zx_grid(co_x[i], txmin[j], txmax[j], subs, c, t, c_peat, t_peat, co_y[i], tymin[j], tymax[j])
    
    if (len(c_peat) == 0) | (len(t_peat) == 0):
        return None, None
    
    return np.hstack(c_peat), np.hstack(t_peat)
# -------------------------------------------------------
# -------------------------------------------------------    

    
"""Combined with numexpr the first check min and max is faster
    Combined without numexpr is slower. It's better to separate min and max"""
def v_per_tri(co, tri_min, tri_max, idxer, tridexer, c_peat=None, t_peat=None):
    """Checks each point against the bounding box of each triangle"""

    co_x, co_y, co_z = co[:, 0], co[:, 1], co[:, 2]
    
    subs = 7
    #subs = bpy.data.objects['Plane.002'].mclo.grid_size
    
    c_peat, t_peat = z_grid(co_z, tri_min[:, 2], tri_max[:, 2], subs, co_x, tri_min[:, 0], tri_max[:, 0], co_y, tri_min[:, 1], tri_max[:, 1])
    if c_peat is None:
        return
    # X
    # Step 1 check x_min (because we're N squared here we break it into steps)
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
    #return c_peat, t_peat


def inside_triangles(tri_vecs, v2, co, tri_co_2, cidx, tidx, nor, ori, in_margin, offset=None):
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
    
    # !!! Watch out for this number. It could affect speed !!! 
    if offset:
        check = (u > -offset) & (v > -offset) & (u + v < offset + 1)
    else:
        check = (u > 0) & (v > 0) & (u + v < 1)
    in_margin[idxer] = check


def object_collide(cloth_ob, col_ob):
    cloth = get_cloth_data(cloth_ob)
    col = get_collider_data(col_ob)

    proxy = col_ob.to_mesh(bpy.context.scene, True, 'PREVIEW')
    # Recreate collider data if number of vertices is changing
    if col.co.shape[0] != len(proxy.vertices):
        col = create_collider_data(col_ob)
    proxy_in_place(col, proxy)
    apply_in_place(cloth_ob, cloth.co, cloth)
    
    inner_margin = col_ob.mclo.object_collision_inner_margin
    outer_margin = col_ob.mclo.object_collision_outer_margin
    fudge = max(inner_margin, outer_margin)

    # check object bounds: (need inner and out margins to adjust box size)
    box_check, co1_min, co1_max = bounds_check(cloth.co, col.co, fudge)
    # check for triangles inside the cloth bounds
    #anim = col_ob.mclo.collision_animated


    if box_check:

        proxy_v_normals_in_place(col, True, proxy)
        tri_co = col.co[col.tridex]
        tri_vo = col.vel[col.tridex]

        tris_in, tri_min, tri_max = triangle_bounds_check(tri_co, co1_min, co1_max, col.tridexer, fudge)#, object.ob.dimensions)

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
                    tri_normals_in_place(col, marginalized)

                    # add normals to make extruded tris
                    u_norms = col.normals[tris_in]
                    #u_norms = norms_2 / np.sqrt(np.einsum('ij, ij->i', norms_2, norms_2))[:, nax] 
                                        
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
                        #if anim:    
                        t_in = tidx[in_margin]
                        
                        tri_vo = tri_vo[tris_in]
                        tri_vel1 = np.mean(tri_co_2[t_in], axis=1)
                        tri_vel2 = np.mean(tri_vo[t_in], axis=1)
                        tvel = tri_vel1 - tri_vel2

                        col_idx = cidx[in_margin] 
                        cloth.co[col_idx] -= nor[in_margin] * (d[in_margin])[:, nax]
                        cloth.vel[col_idx] = tvel

    col.vel[:] = col.co    
    revert_in_place(cloth_ob, cloth.co)

    #temp_ob = bpy.data.objects.new('__TEMP', proxy)
    #for key in proxy.shape_keys.key_blocks:
    #    temp_ob.shape_key_remove(key)
            
    #bpy.data.objects.remove(temp_ob)
    bpy.data.meshes.remove(proxy)


# self collider =============================================
def self_collide(ob):
    cloth = get_cloth_data(ob)
    #col = get_collider_data(ob)

    margin = ob.mclo.object_collision_outer_margin
    fudge = margin

    tri_co = cloth.tri_co

    tri_min = np.min(tri_co, axis=1) - fudge
    tri_max = np.max(tri_co, axis=1) + fudge    

    # begin every vertex co against every tri
    v_tris = v_per_tri(cloth.co, tri_min, tri_max, cloth.idxer, cloth.tridexer)
    if v_tris is not None:
        cidx, tidx = v_tris

        u_norms = cloth.normals

        # don't check faces the verts are part of        
        check_neighbors = cidx[:, nax] == cloth.tridex[tidx]
        cull = np.any(check_neighbors, axis=1)
        cidx, tidx = cidx[~cull], tidx[~cull]
        
        ori = cloth.origins[tidx]
        nor = u_norms[tidx]
        vec2 = cloth.co[cidx] - ori
        
        d = np.einsum('ij, ij->i', nor, vec2) # nor is unit norms
        in_margin = (d > -margin) & (d < margin)
        # <<<--- Inside triangle check --->>>
        # will overwrite in_margin:
        cross_2 = cloth.cross_vecs[tidx][in_margin]
        inside_triangles(cross_2, vec2[in_margin], cloth.co, tri_co, cidx, tidx, nor, ori, in_margin, offset=0.0)
        
        if np.any(in_margin):
            # collision response --------------------------->>>
            t_in = tidx[in_margin]
            #tri_vel1 = np.mean(tri_co[t_in], axis=1)
            #tvel = np.mean(tri_vo[t_in], axis=1)
            #tvel = tri_vel1 - tri_vel2
            t_vel = np.mean(cloth.vel[cloth.tridex][t_in], axis=1)
            
            col_idx = cidx[in_margin] 
            d_in = d[in_margin]
    
            sign_margin = margin * np.sign(d_in) # which side of the face
            c_move = ((nor[in_margin] * d_in[:, nax]) - (nor[in_margin] * sign_margin[:, nax]))#) * -np.sign(d[in_margin])[:, nax]
            #c_move *= 1 / cloth.ob.modeling_cloth_grid_size
            #cloth.co[col_idx] -= ((nor[in_margin] * d_in[:, nax]) - (nor[in_margin] * sign_margin[:, nax]))#) * -np.sign(d[in_margin])[:, nax]
            cloth.co[col_idx] -= c_move #* .7
            #cloth.vel[col_idx] = 0
            cloth.vel[col_idx] = t_vel

    #col.vel[:] = col.co    
# self collider =============================================


# update functions --------------------->>>    
def tile_and_remove_neighbors(vidx, tidx, c_peat, t_peat):

    tshape = tidx.shape[0]
    vshape = vidx.shape[0]

    # eliminate tris that contain the point: 
    # check the speed difference of doing a reshape with ravel at the end
    co_tidex = c_peat.reshape(vshape, tshape)
    tri_tidex = tidx[t_peat.reshape(vshape, tshape)]
    check = tri_tidex == vidx[co_tidex][:,:,nax]
    cull = ~np.any(check, axis=2)

    # duplicate of each tri for each vert and each vert for each tri
    c_peat = c_peat[cull.ravel()]
    t_peat = t_peat[cull.ravel()]
    
    return c_peat, t_peat


class ColliderData:
    pass


class SelfColliderData:
    pass


def get_collider_data(ob):
    col_data = bpy.context.scene.modeling_cloth_data_set_colliders

    col = None
    for key, c in col_data.items():
        if c.ob == ob:
            col = c

    if not col:
        col = create_collider_data(ob)

    return col

def create_collider_data(ob):
    col_data = bpy.context.scene.modeling_cloth_data_set_colliders

    col = ColliderData()
    col_data[ob.name] = col
    col.ob = ob

    # get proxy
    proxy = ob.to_mesh(bpy.context.scene, True, 'PREVIEW')

    col.co = get_proxy_co(ob, None, proxy)
    col.idxer = np.arange(col.co.shape[0], dtype=np.int32)
    proxy_in_place(col, proxy)
    col.v_normals = proxy_v_normals(col.ob, proxy)
    col.vel = np.copy(col.co)
    col.tridex = triangulate(col.ob, proxy)
    col.tridexer = np.arange(col.tridex.shape[0], dtype=np.int32)
    # cross_vecs used later by barycentric tri check
    proxy_v_normals_in_place(col, True, proxy)
    marginalized = np.array(col.co + col.v_normals * ob.mclo.object_collision_outer_margin, dtype=np.float32)
    col.cross_vecs, col.origins, col.normals = get_tri_normals(marginalized[col.tridex])    

    col.cross_vecs.dtype = np.float32
    col.origins.dtype = np.float32
    #col.normals.dtype = np.float32

    # remove proxy
    bpy.data.meshes.remove(proxy)

    print('INFO: Collider data for', ob.name, 'is created!')
    return col


# Self collision object
def create_self_collider(ob):
    # maybe fixed? !!! bug where first frame of collide uses empty data. Stuff goes flying.
    col = ColliderData()
    col.ob = ob
    col.co = get_co(ob, None)
    proxy_in_place(col)
    col.v_normals = proxy_v_normals(ob)
    col.vel = np.copy(col.co)
    col.tridex = triangulate(ob)
    col.tridexer = np.arange(col.tridex.shape[0], dtype=np.int32)
    # cross_vecs used later by barycentric tri check
    proxy_v_normals_in_place(col)
    marginalized = np.array(col.co + col.v_normals * ob.mclo.object_collision_outer_margin, dtype=np.float32)
    col.cross_vecs, col.origins, col.normals = get_tri_normals(marginalized[col.tridex])    

    col.cross_vecs.dtype = np.float32
    col.origins.dtype = np.float32
    #col.normals.dtype = np.float32

    return col


# collide object updater
def collision_object_update(self, context):
    """Updates the collider object"""    
    scene = context.scene
    col_data = scene.modeling_cloth_data_set_colliders
    ob = self.id_data

    if self.object_collision:
        cp = scene.mclo.collider_pointers.add()
        cp.ob = ob
    else:
        for i, cp in enumerate(scene.mclo.collider_pointers):
            if cp.ob == ob:

                # Remove collider data first
                cull_keys = []
                for key, col in col_data.items():
                    if col.ob == cp.ob:
                        cull_keys.append(key)
                for key in cull_keys:
                    del(col_data[key])

                scene.mclo.collider_pointers.remove(i)
                break
    

# cloth object detect updater:
def cloth_object_update(self, context):
    """Updates the cloth object when detecting."""
    print("ran the detect updater. It did nothing.")


def manage_animation_handler(self, context):
    ob = self.id_data
    if ob.mclo.frame_update:
        ob.mclo.scene_update = False
    

def manage_continuous_handler(self, context):    
    ob = self.id_data
    if ob.mclo.scene_update:
        ob.mclo.frame_update = False
    

# =================  Handler  ======================
@persistent
def handler_frame(scene):
    handler_unified(scene, frame_update=True)


@persistent
def handler_scene(scene):
    handler_unified(scene, frame_update=False)


def handler_unified(scene, frame_update=False):
    data = bpy.context.scene.modeling_cloth_data_set
    cull_ids = []

    for i, cp in enumerate(scene.mclo.cloth_pointers):
        ob = cp.ob

        # Check if object still exists
        if not ob or (ob and not scene.objects.get(ob.name)):
            if scene.mclo.last_object == ob:
                scene.mclo.last_object = None
            cull_ids.append(i)

        else:
            cloth = get_cloth_data(ob)

            # Frame update
            if frame_update and ob.mclo.frame_update:
                run_handler(ob, cloth)
                if ob.mclo.auto_reset:
                    if scene.frame_current <= 1:    
                        reset_shapes(ob)

            # Scene update
            elif not frame_update and ob.mclo.scene_update:
                run_handler(ob, cloth)

    # Remove missing object from cloth pointer
    for i in reversed(cull_ids):
        ob = scene.mclo.cloth_pointers[i].ob
        if ob:
            ob.mclo.enable = False
        else:
            scene.mclo.cloth_pointers.remove(i)


def get_cloth_data(ob):
    data = bpy.context.scene.modeling_cloth_data_set
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
    data = scene.modeling_cloth_data_set
    extra_data = scene.modeling_cloth_data_set_extra
    scene.mclo.last_object = ob
    
    if ob.mclo.enable:
        # New cloth on scene data
        cp = scene.mclo.cloth_pointers.add()
        cp.ob = ob
        create_cloth_data(ob)
    else:
        for i, cp in enumerate(scene.mclo.cloth_pointers):
            if cp.ob == ob:

                # Remove cloth data first
                cull_keys = []
                for key, cloth in data.items():
                    if ob == cloth.ob:
                        cull_keys.append(key)

                for key in cull_keys:
                    del(data[key])

                # Remove pointers
                scene.mclo.cloth_pointers.remove(i)
    
def visible_objects_and_duplis(context):
    """Loop over (object, matrix) pairs (mesh only)"""

    for obj in context.visible_objects:
        if obj.type == 'MESH':
            if obj.mclo.enable:    
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


# sewing --------->>>
class ModelingClothSew(bpy.types.Operator):
    """For connected two edges with sew lines"""
    bl_idname = "object.modeling_cloth_create_sew_lines"
    bl_label = "Modeling Cloth Create Sew Lines"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        
        #ob = get_last_object() # returns tuple with list and last cloth objects or None
        #if ob is not None:
            #obj = ob[1]
        #else:
        obj = bpy.context.object
        
        #bpy.context.scene.objects.active = obj
        mode = obj.mode
        if mode != "EDIT":
            bpy.ops.object.mode_set(mode="EDIT")
        
        create_sew_edges()
        bpy.ops.object.mode_set(mode="EDIT")            

        return {'FINISHED'}
# sewing --------->>>


class ModelingClothPin(bpy.types.Operator):
    """Modal ray cast for placing pins"""
    bl_idname = "view3d.modeling_cloth_pin"
    bl_label = "Modeling Cloth Pin"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.space_data.type == 'VIEW_3D' and any(context.scene.mclo.cloth_pointers)

    def __init__(self):
        self.obj = None
        self.latest_hit = None
        self.closest = None

    def invoke(self, context, event):
        #bpy.ops.object.select_all(action='DESELECT')    
        context.scene.mclo.pin_alert = True

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
                pin = self.obj.mclo.pins.add()
                pin.vertex_id = self.closest
                pin.hook = e
                self.latest_hit = None
                self.obj = None
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            delete_guide()
            cloths = [i for i in bpy.data.objects if i.mclo.enable] # so we can select an empty and keep the settings menu up
            context.scene.mclo.pin_alert = False
            if len(cloths) > 0:                                        #
                ob = context.scene.mclo.last_object
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
        return context.space_data.type == 'VIEW_3D' and any(context.scene.mclo.cloth_pointers)

    def __init__(self):
        self.clicked = False
        self.stored_mouse = None
        self.matrix = None

    def invoke(self, context, event):
        scene = context.scene
        extra_data = scene.modeling_cloth_data_set_extra
        scene.mclo.drag_alert = True
        #bpy.ops.object.select_all(action='DESELECT')    

        extra_data['vidx'] = None # Vertex ids of dragged face
        extra_data['stored_vidx'] = None # Vertex coordinates
        extra_data['move'] = None # Direction of drag

        for cp in scene.mclo.cloth_pointers:
            if cp.ob:
                cp.ob.mclo.clicked = False
            
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def main_drag(self, context, event):
        # get the context arguments
        scene = context.scene
        extra_data = scene.modeling_cloth_data_set_extra
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
                best_obj.mclo.clicked = True
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
        #data = scene.modeling_cloth_data_set
        extra_data = scene.modeling_cloth_data_set_extra
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
            for cp in scene.mclo.cloth_pointers:
                if cp.ob:
                    cp.ob.mclo.clicked = False
            
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            self.clicked = False
            self.stored_mouse = None
            bpy.context.window.cursor_set("DEFAULT")
            scene.mclo.drag_alert = False
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

        for i, pin in reversed(list(enumerate(ob[1].mclo.pins))):
            bpy.data.objects.remove(pin.hook)
            ob[1].mclo.pins.remove(i)

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
        #bpy.ops.object.select_all(action='DESELECT')
        for pin in ob[1].mclo.pins:
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
            pin = ob.mclo.pins.add()
            pin.vertex_id = v
            pin.hook = e
            ob.select = False
        bpy.ops.object.mode_set(mode='EDIT')       
        
        return {'FINISHED'}


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

class ModelingClothObject(bpy.types.PropertyGroup):
    ob = PointerProperty(type=bpy.types.Object)

class ModelingClothCollider(bpy.types.PropertyGroup):
    ob = PointerProperty(type=bpy.types.Object)

class ModelingClothGlobals(bpy.types.PropertyGroup):

    cloth_pointers = CollectionProperty(
            name="Modeling Cloth Objects", 
            description = 'List of cloth objects for quick pointers',
            type=ModelingClothObject)

    collider_pointers = CollectionProperty(
            name="Modeling Cloth Colliders", 
            description = 'List of collider objects for quick pointers',
            type=ModelingClothCollider)

    drag_alert = BoolProperty(default=False)
    pin_alert = BoolProperty(default=False)
    last_object = PointerProperty(type=bpy.types.Object)

class ModelingClothPinObject(bpy.types.PropertyGroup):
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
        default=1, precision=4, min=0, max=2.5)#, update=refresh_noise)

    push_springs = FloatProperty(name="Modeling Cloth Spring Force", 
        description="Set the spring force", 
        #default=0.2, precision=4, min=0, max=1.5)#, update=refresh_noise)
        default=1, precision=4, min=0, max=2.5)#, update=refresh_noise)
    # -------------------------->>>

    gravity = FloatProperty(name="Modeling Cloth Gravity", 
        description="Modeling cloth gravity", 
        default=0.0, precision=4, min= -10, max=10)#, update=refresh_noise_decay)

    iterations = IntProperty(name="Iterations", 
        description="How stiff the cloth is", 
        default=2, min=1, max=500)#, update=refresh_noise_decay)

    velocity = FloatProperty(name="Velocity", 
        description="Cloth keeps moving", 
        default=.98, min= -200, max=200, soft_min= -1, soft_max=1)#, update=refresh_noise_decay)

    # Wind. Note, wind should be measured agains normal and be at zero when normals are at zero. Squared should work
    wind_x = FloatProperty(name="Wind X", 
        description="Not the window cleaner", 
        default=0, min= -1, max=1, soft_min= -10, soft_max=10)#, update=refresh_noise_decay)

    wind_y = FloatProperty(name="Wind Y", 
        description="Y? Because wind is cool", 
        default=0, min= -1, max=1, soft_min= -10, soft_max=10)#, update=refresh_noise_decay)

    wind_z = FloatProperty(name="Wind Z", 
        description="It's windzee outzide", 
        default=0, min= -1, max=1, soft_min= -10, soft_max=10)#, update=refresh_noise_decay)

    turbulence = FloatProperty(name="Wind Turbulence", 
        description="Add Randomness to wind", 
        default=0, min=0, max=1, soft_min= -10, soft_max=10)#, update=refresh_noise_decay)

    # self collision ----->>>
#    self_collision = BoolProperty(name="Modeling Cloth Self Collsion", 
#        description="Toggle self collision", 
#        default=False, update=collision_data_update)
#
#    self_collision_force = FloatProperty(name="recovery force", 
#        description="Self colide faces repel", 
#        default=.17, precision=4, min= -1.1, max=1.1, soft_min= 0, soft_max=1)
#
#    self_collision_margin = FloatProperty(name="Margin", 
#        description="Self colide faces margin", 
#        default=.08, precision=4, min= -1, max=1, soft_min= 0, soft_max=1)
#
#    self_collision_cy_size = FloatProperty(name="Cylinder size", 
#        description="Self colide faces cylinder size", 
#        default=1, precision=4, min= 0, max=4, soft_min= 0, soft_max=1.5)
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

    #collision_animated = bpy.props.BoolProperty(name="Modeling Cloth Collsion Animated", 
        #description="Treat collide object as animated. (turn off for speed on static objects)", 
        #default=True)#, update=collision_object_update)
    
    object_collision_detect = BoolProperty(name="Modeling Cloth Self Collsion", 
        description="Detect collision objects", 
        default=False, update=cloth_object_update)    

    object_collision_outer_margin = FloatProperty(name="Modeling Cloth Outer Margin", 
        description="Collision margin on positive normal side of face", 
        default=0.04, precision=4, min=0, max=100, soft_min=0, soft_max=1000)
        
    object_collision_inner_margin = FloatProperty(name="Modeling Cloth Inner Margin", 
        description="Collision margin on negative normal side of face", 
        default=0.1, precision=4, min=0, max=100, soft_min=0, soft_max=1000)        
    # ---------------------------->>>

    # more collision stuff ------->>>
    grid_size = IntProperty(name="Modeling Cloth Grid Size", 
        description="Max subdivisions for the dynamic broad phase grid", 
        default=10, min=0, max=1000, soft_min=0, soft_max=1000)

    # Not for manual editing ----->>>        
    waiting = BoolProperty(name='Pause Cloth Update',
            default=False)

    clicked = BoolProperty(name='Click for drag event',
            default=False)

    pins = CollectionProperty(name="Modeling Cloth Pins", 
            type=ModelingClothPinObject)

    virtual_springs = CollectionProperty(name="Modeling Cloth Virtual Springs", 
            type=ModelingClothVirtualSpring)


def create_properties():            

    bpy.types.Scene.mclo = PointerProperty(type=ModelingClothGlobals)
    bpy.types.Object.mclo = PointerProperty(type=ModelingClothObjectProps)

    # property dictionaries
    bpy.types.Scene.modeling_cloth_data_set = {} 
    bpy.types.Scene.modeling_cloth_data_set_colliders = {}
    bpy.types.Scene.modeling_cloth_data_set_extra = {} 
    
        
def remove_properties():            
    '''Drives to the grocery store and buys a sandwich'''
    # No need to remove properties because yolo
    pass

@persistent
def refresh_cloth_data(scene):

    # Create new data based on available clothes and colliders
    scene = bpy.context.scene
    for cp in scene.mclo.cloth_pointers:
        if cp.ob:
            create_cloth_data(cp.ob)

    for cp in scene.mclo.collider_pointers:
        if cp.ob:
            create_collider_data(cp.ob)


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
        
        # tools
        col = layout.column(align=True)
        col.label(text="Tools")        
        col.operator("object.modeling_cloth_create_sew_lines", text="Sew Lines", icon="MOD_UVPROJECT")
        
        # modeling cloth
        col = layout.column(align=True)
        col.label(text="Modeling Cloth")
        ob = bpy.context.object
        cloths = [i for i in bpy.data.objects if i.mclo.enable] # so we can select an empty and keep the settings menu up
        if len(cloths) > 0:
            status = scene.mclo.pin_alert
            if ob is not None:
                if ob.type != 'MESH' or status:
                    ob = scene.mclo.last_object

        if ob is not None:
            if ob.type == 'MESH':
                col.prop(ob.mclo ,"enable", text="Modeling Cloth", icon='SURFACE_DATA')               

                col.prop(ob.mclo ,"object_collision", text="Collider", icon="STYLUS_PRESSURE")
                #if ob.mclo.object_collision:
                    #col.prop(ob.mclo ,"collision_animated", text="Animated", icon="POSE_DATA")
                if ob.mclo.object_collision:
                    col.prop(ob.mclo ,"object_collision_outer_margin", text="Outer Margin", icon="FORCE_FORCE")
                    col.prop(ob.mclo ,"object_collision_inner_margin", text="Inner Margin", icon="STICKY_UVS_LOC")
                
                col.label("Collide List:")
                for cp in scene.mclo.collider_pointers:
                    if cp.ob:
                        col.label(cp.ob.name)

                if ob.mclo.enable:
                    #col.label('Active: ' + ob.name)

                    # object collisions
                    col = layout.column(align=True)
                    col.label("Collisions")
                    col.prop(ob.mclo ,"object_collision_detect", text="Object Collisions", icon="PHYSICS")

                    col = layout.column(align=True)
                    col.scale_y = 2.0

                    col = layout.column(align=True)
                    col.scale_y = 1.4
                    col.prop(ob.mclo, "grid_size", text="Grid Boxes", icon="MESH_GRID")
                    col.prop(ob.mclo, "frame_update", text="Animation Update", icon="TRIA_RIGHT")
                    if ob.mclo.frame_update:    
                        col.prop(ob.mclo, "auto_reset", text="Frame 1 Reset")
                    col.prop(ob.mclo, "scene_update", text="Continuous Update", icon="TIME")
                    col = layout.column(align=True)
                    col.scale_y = 2.0
                    col.operator("object.modeling_cloth_reset", text="Reset")
                    col.alert = scene.mclo.drag_alert
                    col.operator("view3d.modeling_cloth_drag", text="Grab")
                    col = layout.column(align=True)
                        
                    col.prop(ob.mclo ,"iterations", text="Iterations")#, icon='OUTLINER_OB_LATTICE')               
                    col.prop(ob.mclo ,"spring_force", text="Stiffness")#, icon='OUTLINER_OB_LATTICE')               
                    col.prop(ob.mclo ,"push_springs", text="Push Springs")#, icon='OUTLINER_OB_LATTICE')               
                    col.prop(ob.mclo ,"noise", text="Noise")#, icon='PLAY')               
                    col.prop(ob.mclo ,"noise_decay", text="Decay Noise")#, icon='PLAY')               
                    col.prop(ob.mclo ,"gravity", text="Gravity")#, icon='PLAY')        
                    col.prop(ob.mclo ,"inflate", text="Inflate")#, icon='PLAY')        
                    col.prop(ob.mclo ,"sew", text="Sew Force")#, icon='PLAY')        
                    col.prop(ob.mclo ,"velocity", text="Velocity")#, icon='PLAY')        
                    col = layout.column(align=True)
                    col.label("Wind")                
                    col.prop(ob.mclo ,"wind_x", text="Wind X")#, icon='PLAY')        
                    col.prop(ob.mclo ,"wind_y", text="Wind Y")#, icon='PLAY')        
                    col.prop(ob.mclo ,"wind_z", text="Wind Z")#, icon='PLAY')        
                    col.prop(ob.mclo ,"turbulence", text="Turbulence")#, icon='PLAY')        
                    col.prop(ob.mclo ,"floor", text="Floor")#, icon='PLAY')        
                    col = layout.column(align=True)
                    col.scale_y = 1.5
                    col.alert = status
                    if ob.mclo.enable:    
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
                        #col.prop(ob.mclo ,"self_collision", text="Self Collision")#, icon='PLAY')        
                        #col.prop(ob.mclo ,"self_collision_force", text="Repel")#, icon='PLAY')        
                        #col.prop(ob.mclo ,"self_collision_margin", text="Margin")#, icon='PLAY')        
                        #col.prop(ob.mclo ,"self_collision_cy_size", text="Cylinder Size")#, icon='PLAY')        

                    
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

    # Register all classes if this file loaded separately
    if __name__ in {'__main__', 'ModelingCloth'}:
        bpy.utils.register_module(__name__)

    # Main handlers
    bpy.app.handlers.frame_change_post.append(handler_frame)
    bpy.app.handlers.scene_update_post.append(handler_scene)

    # Add load handlers
    bpy.app.handlers.load_post.append(refresh_cloth_data)


def unregister():
    # Remove load handlers
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
