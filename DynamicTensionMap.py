import bpy, bmesh, json
import numpy as np
from bpy.props import *
from numpy import newaxis as nax
from bpy.app.handlers import persistent


bl_info = {
    "name": "Dynamic Tension Map",
    "author": "Rich Colburn (email: the3dadvantage@gmail.com), Yusuf Umar (@ucupumar)",
    "version": (1, 0),
    "blender": (2, 79, 0),
    "location": "View3D > Extended Tools > Tension Map",
    "description": "Compares the current state of the mesh agains a stored version and displays stretched edges as a color",
    "warning": "'For the empire!' should not be shouted when paying for gum.",
    "wiki_url": "",
    "category": '3D View'}

MATERIAL_NAME = '__TensionMap'
VCOL_NAME = '__Tension'

def create_dynamic_tension_material():
    '''Creates a node material for displaying the vertex colors'''
    # Go to blender internal first
    scene = bpy.context.scene
    ori_engine = scene.render.engine

    # Create internal material nodes
    scene.render.engine = 'BLENDER_RENDER'
    mat = bpy.data.materials.new(MATERIAL_NAME)
    mat.use_nodes = True
    mat.specular_intensity = 0.1
    mat.specular_hardness = 17
    mat.use_transparency = True

    mat.node_tree.nodes.new(type="ShaderNodeGeometry")
    mat.node_tree.links.new(
        mat.node_tree.nodes['Geometry'].outputs['Vertex Color'], 
        mat.node_tree.nodes['Material'].inputs[0])
    mat.node_tree.nodes['Geometry'].color_layer = VCOL_NAME
    mat.node_tree.nodes['Material'].material = mat

    # Create cycles material nodes
    scene.render.engine = 'CYCLES'
    outp = mat.node_tree.nodes.new(type="ShaderNodeOutputMaterial")
    diff = mat.node_tree.nodes.new(type="ShaderNodeBsdfDiffuse")
    attr = mat.node_tree.nodes.new(type="ShaderNodeAttribute")
    attr.attribute_name = VCOL_NAME

    mat.node_tree.links.new(attr.outputs['Color'], diff.inputs[0])
    mat.node_tree.links.new(diff.outputs['BSDF'], outp.inputs[0])

    # Back to original engine
    scene.render.engine = ori_engine

    return mat


def get_set_dynamic_tension_material(ob):

    # Search for material in object data
    mat = ob.data.materials.get(MATERIAL_NAME)
    if not mat:
        # If not found search for global data
        mat = bpy.data.materials.get(MATERIAL_NAME)

        # Create material if still not found
        if not mat:
            mat = create_dynamic_tension_material()

        # Append material to object data
        ob.data.materials.append(mat)

    return mat


def hide_unhide_store(ob=None, unhide=True, storage=None):
    """Stores the hidden state of the geometry, """
    if ob is None:
        ob = bpy.context.object
    if ob is None:
        return 'No mesh object'
    
    if unhide:
        v = np.zeros(len(ob.data.vertices), dtype=np.bool)
        e = np.zeros(len(ob.data.edges), dtype=np.bool)
        f = np.zeros(len(ob.data.polygons), dtype=np.bool)
        vsel = np.zeros(len(ob.data.vertices), dtype=np.bool)
        esel = np.zeros(len(ob.data.edges), dtype=np.bool)
        fsel = np.zeros(len(ob.data.polygons), dtype=np.bool)        
        if storage is not None:
            ob.data.vertices.foreach_get('hide', v)
            ob.data.edges.foreach_get('hide', e)
            ob.data.polygons.foreach_get('hide', f)

            ob.data.vertices.foreach_get('select', vsel)
            ob.data.edges.foreach_get('select', esel)
            ob.data.polygons.foreach_get('select', fsel)

            
            storage['hide'] = {}
            storage['hide']['v'] = np.copy(v)
            storage['hide']['e'] = np.copy(e)
            storage['hide']['f'] = np.copy(f)
            
            storage['hide']['vsel'] = np.copy(vsel)
            storage['hide']['esel'] = np.copy(esel)
            storage['hide']['fsel'] = np.copy(fsel)

        
        v[:] = False
        e[:] = False
        f[:] = False

        ob.data.vertices.foreach_set('hide', v)
        ob.data.edges.foreach_set('hide', e)
        ob.data.polygons.foreach_set('hide', f)    
    else:
        ob.data.vertices.foreach_set('hide', storage['hide']['v'])
        ob.data.edges.foreach_set('hide', storage['hide']['e'])
        ob.data.polygons.foreach_set('hide', storage['hide']['f'])

        ob.data.vertices.foreach_set('select', storage['hide']['vsel'])
        ob.data.edges.foreach_set('select', storage['hide']['esel'])
        ob.data.polygons.foreach_set('select', storage['hide']['fsel'])
        

def get_key_coords(ob=None, key='Basis', proxy=False):
    '''Creates an N x 3 numpy array of vertex coords.
    from shape keys'''
    if key is None:
        return get_coords(ob)
    if proxy:
        mesh = proxy.to_mesh(bpy.context.scene, True, 'PREVIEW')
        verts = mesh.data.shape_keys.key_blocks[key].data
    else:
        verts = ob.data.shape_keys.key_blocks[key].data
    v_count = len(verts)
    coords = np.zeros(v_count * 3, dtype=np.float32)
    verts.foreach_get("co", coords)
    if proxy:
        bpy.data.meshes.remove(mesh)
    return coords.reshape(v_count, 3)


def get_coords(ob=None, proxy=False):
    '''Creates an N x 3 numpy array of vertex coords. If proxy is used the
    coords are taken from the object specified with modifiers evaluated.
    For the proxy argument put in the object: get_coords(ob, proxy_ob)'''
    if ob is None:
        ob = bpy.context.object
    if proxy:
        mesh = proxy.to_mesh(bpy.context.scene, True, 'PREVIEW')
        verts = mesh.vertices
    else:
        verts = ob.data.vertices
    v_count = len(verts)
    coords = np.zeros(v_count * 3, dtype=np.float32)
    verts.foreach_get("co", coords)
    if proxy:
        bpy.data.meshes.remove(mesh)
    return coords.reshape(v_count, 3)


def get_edge_idx(ob=None):
    if ob is None:
        ob = bpy.context.object
    ed = np.zeros(len(ob.data.edges)*2, dtype=np.int32)
    ob.data.edges.foreach_get('vertices', ed)
    return ed.reshape(len(ed)//2, 2)


def get_bmesh(ob=None):
    '''Returns a bmesh. Works either in edit or object mode.
    ob can be either an object or a mesh.'''
    obm = bmesh.new()
    if ob is None:
        mesh = bpy.context.object.data
    if 'data' in dir(ob):
        mesh = ob.data
        if ob.mode == 'OBJECT':
            obm.from_mesh(mesh)
        elif ob.mode == 'EDIT':
            obm = bmesh.from_edit_mesh(mesh)    
    else:
        mesh = ob
        obm.from_mesh(mesh)
    return obm


def reassign_mats(ob): #=None, type=None):
    '''Resets materials based on stored face indices'''
    if ob.dten.enable:
        idx = ob.data.materials.find(MATERIAL_NAME)
        mat_idx = np.full(len(ob.data.polygons), idx, dtype=np.int32)
        ob.data.polygons.foreach_set('material_index', mat_idx)
        ob.data.update()
    else:
        mat_idx = json.loads(ob.dten.mat_index_str)
        ob.data.polygons.foreach_set('material_index', mat_idx)
        ob.data.update()
        ob.dten.mat_index_str = ''

        # Search for dynamic tension material then remove it
        idx = ob.data.materials.find(MATERIAL_NAME)
        if idx != -1:    
            ob.data.materials.pop(idx, update_data=True)

def initialize(ob): #, key):
    '''Set up the indexing for viewing each edge per vert per face loop'''
    data = bpy.context.scene.dynamic_tension_map_dict
    data[ob.name] = dtdata = {}

    source = False
    key = None

    if ob.type != 'MESH': return
        
    keys = ob.data.shape_keys
    if keys != None:
        if 'RestShape' in keys.key_blocks:    
            key = 'RestShape'
        elif 'modeling cloth source key' in keys.key_blocks:
            key = 'modeling cloth source key'
            source = True
        else:
            key = keys.key_blocks[0].name

    ob.dten.source = source

    obm = get_bmesh(ob)
    ed_pairs_per_v = []
    for f in obm.faces:
        for v in f.verts:
            set = []
            for e in f.edges:
                if v in e.verts:
                    set.append(e.index)
            ed_pairs_per_v.append(set)    

    dtdata['ed_pairs_per_v'] = np.array(ed_pairs_per_v)
    dtdata['zeros'] = np.zeros(len(dtdata['ed_pairs_per_v']) * 3).reshape(len(dtdata['ed_pairs_per_v']), 3)

    key_coords = get_key_coords(ob, key)
    ed1 = get_edge_idx(ob)
    #linked = np.array([len(i.link_faces) for i in obm.edges]) > 0
    dtdata['edges'] = get_edge_idx(ob)#[linked]
    dif = key_coords[dtdata['edges'][:,0]] - key_coords[dtdata['edges'][:,1]]
    dtdata['mags'] = np.sqrt(np.einsum('ij,ij->i', dif, dif))

    # Store original material index
    mat_idx = np.zeros(len(ob.data.polygons), dtype=np.int64)
    ob.data.polygons.foreach_get('material_index', mat_idx)
    mat_idx_str = json.dumps(mat_idx.tolist())
    ob.dten.mat_index_str = mat_idx_str

    #if 'material' not in dtdata:
    print('ran this')
    get_set_dynamic_tension_material(ob)

    print('INFO: Dynamic tension data for', ob.name, 'is created!')

    return dtdata

@persistent
def refresh_dynamic_tension_data(scene):
    data = bpy.context.scene.dynamic_tension_map_dict
    for i, op in enumerate(bpy.context.scene.dten.object_pointers):
        if op.ob:
            initialize(op.ob)

@persistent
def dynamic_tension_handler(scene):
    scene = bpy.context.scene
    data = scene.dynamic_tension_map_dict
    stretch = scene.dten.max_stretch / 100

    cull_ids = []
    for i, op in enumerate(bpy.context.scene.dten.object_pointers):
        if op.ob and scene.objects.get(op.ob.name):
            update(ob=op.ob, max_stretch=stretch, bleed=0.2)   
        else:
            cull_ids.append(i)

    for i in reversed(cull_ids):
        cp = scene.dten.object_pointers[i]

        if cp.object and cp.object.name in data:
            del(data[cp.object.name])

        scene.dten.object_pointers.remove(i)

def prop_callback(self, context):
    stretch = bpy.context.scene.dten.max_stretch / 100
    edit=False
    ob = bpy.context.object
    if not ob.dten.enable:
        return
    if ob.mode == 'EDIT':
        bpy.ops.object.mode_set(mode='OBJECT')
        edit = True
    update(coords=None, ob=None, max_stretch=stretch, bleed=0.2)    
    if edit:
        bpy.ops.object.mode_set(mode='EDIT')
        
    
def update(coords=None, ob=None, max_stretch=1, bleed=0.2):
    '''Measure the edges against the stored lengths.
    Look up those distances with fancy indexing on
    a per-vertex basis.'''
    scene = bpy.context.scene
    data = scene.dynamic_tension_map_dict
    if ob is None:
        ob = bpy.context.object

    if not ob.dten.waiting and ob.mode != 'OBJECT':
        ob.dten.waiting = True

    if ob.dten.waiting:    
        if ob.mode == 'OBJECT':
            initialize(ob)
            ob.dten.waiting = False
        else:
            return

    try:
        dtdata = data[ob.name]
    except: 
        dtdata = initialize(ob)
    
    if ob.dten.source:
        coords = get_key_coords(ob, 'modeling cloth key')
    if scene.dten.show_from_flat:
        if ob.data.shape_keys is not None:
            if len(ob.data.shape_keys.key_blocks) > 1:
                coords = get_key_coords(ob, ob.data.shape_keys.key_blocks[1].name)
    if coords is None:
        coords = get_coords(ob, ob)
        
    dif = coords[dtdata['edges'][:,0]] - coords[dtdata['edges'][:,1]]
    mags = np.sqrt(np.einsum('ij,ij->i', dif, dif))
    if scene.dten.map_percentage:    
        div = (mags / dtdata['mags']) - 1
    else:
        div = mags - dtdata['mags']
    color = dtdata['zeros']
    eye = np.eye(3,3)
    G, B = eye[1], eye[2]
    ed_pairs = dtdata['ed_pairs_per_v']
    mix = np.mean(div[ed_pairs], axis=1)
    mid = (max_stretch) * 0.5     
    BG_range = mix < mid
    GR_range = -BG_range

    #to_y = np.array([ 0,  1, -1]) / np.clip(max_stretch, 0, 100)
    to_y = np.array([ 0,  1, -1]) / max_stretch
    #to_x = np.array([ 1, -1,  0]) / np.clip(max_stretch, 0, 100)
    to_x = np.array([ 1, -1,  0]) / max_stretch

    BG_blend = to_y * (mix[BG_range])[:, nax]
    GR_blend = to_x * (mix[GR_range])[:, nax]
    
    color[BG_range] = B
    color[BG_range] += BG_blend
    color[GR_range] = G
    color[GR_range] += GR_blend

    UV = np.nan_to_num(color / np.sqrt(np.einsum('ij,ij->i', color, color)[:, nax]))
    ob.data.vertex_colors[VCOL_NAME].data.foreach_set('color', UV.ravel())
    ob.data.update()


def toggle_enable(self, context):
    if type(self) == bpy.types.Object: 
        ob = self
    else: 
        ob = self.id_data
    scene = context.scene
    data = scene.dynamic_tension_map_dict

    if ob.dten.enable:

        # Set object vertex color
        if VCOL_NAME not in ob.data.vertex_colors:    
            ob.data.vertex_colors.new(VCOL_NAME)

        initialize(ob)
        op = scene.dten.object_pointers.add()
        op.ob = ob
    else:
        # Remove vertex color
        vcol = ob.data.vertex_colors.get(VCOL_NAME)
        if vcol:
            ob.data.vertex_colors.remove(vcol)

        # Remove pointer and data
        for i, op in enumerate(scene.dten.object_pointers):
            if op.ob == ob:
                if ob.name in data:
                    del(data[ob.name])
                scene.dten.object_pointers.remove(i)
                break

    reassign_mats(ob)
    prop_callback(context.scene, bpy.context)

    # Change viewport shade to material
    context.space_data.viewport_shade = 'MATERIAL'
    

def pattern_on_off(self, context):
    ob = self.id_data
    if ob.dten.flat_pattern_on:
        ob.data.shape_keys.key_blocks[1].value = 1
        ob.active_shape_key_index = 1
        
    else:
        ob.data.shape_keys.key_blocks[1].value = 0
        ob.active_shape_key_index = 0

def tension_from_flat_update(self, context):
    #toggle_enable(context.object, context)
    #prop_callback(context.scene, context)
    pass

        
# Create Properties----------------------------:
def percentage_prop_update(self, context):
    scene = self.id_data
    if scene.dten.map_percentage:
        scene.dten.map_edge = False
    else:
        scene.dten.map_edge = True        

    
def edge_prop_update(self, context):
    scene = self.id_data
    if scene.dten.map_edge:
        scene.dten.map_percentage = False
    else:
        scene.dten.map_percentage = True

class DynamicTensionObjectPointer(bpy.types.PropertyGroup):
    ob = PointerProperty(type=bpy.types.Object)

class DynamicTensionObjectProps(bpy.types.PropertyGroup):

    enable = BoolProperty(name="Dynamic Tension Map On", 
        description="Amount of stretch on an edge is blender units. ", 
        default=False, update=toggle_enable)

    flat_pattern_on = BoolProperty(name="Dynamic Tension Flat Pattern On", 
        description="Toggles the flat pattern shape on and off.", 
        default=False, update=pattern_on_off)

    # mat_index
    mat_index_str = StringProperty(default='')

    source = BoolProperty(default=False)

    waiting = BoolProperty(default=False)

class DynamicTensionSceneProps(bpy.types.PropertyGroup):

    max_stretch = FloatProperty(name="avatar height", 
        description="Stretch distance where tension map appears red", 
        default=20, min=0.001, max=200,  precision=2, update=prop_callback)

    show_from_flat = BoolProperty(name="Dynamic Tension From Flat", 
        description="Tension is displayed as the difference between the current shape and the source shape.", 
        default=False, update=tension_from_flat_update)

    map_percentage = BoolProperty(name="Dynamic Tension Map Percentage", 
        description="For toggling between percentage and geometry", 
        default=True, update=percentage_prop_update)

    map_edge = BoolProperty(name="Dynamic Tension Map Edge", 
        description="For toggling between percentage and geometry", 
        default=False, update=edge_prop_update)

    object_pointers = CollectionProperty(type=DynamicTensionObjectPointer)

        
def create_properties():            

    bpy.types.Object.dten = PointerProperty(type=DynamicTensionObjectProps)
    bpy.types.Scene.dten = PointerProperty(type=DynamicTensionSceneProps)
    
    # create data dictionary
    bpy.types.Scene.dynamic_tension_map_dict = {}
    bpy.types.Scene.dynamic_tension_map_selection = {}


def remove_properties():
        
    del(bpy.types.Scene.dynamic_tension_map_dict)
    del(bpy.types.Scene.dynamic_tension_map_selection)

    
# Create Classes-------------------------------:
class UpdatePattern(bpy.types.Operator):
    """Update Pattern"""
    bl_idname = "object.dynamic_tension_map_update_pattern"
    bl_label = "dynamic_tension_update_pattern"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        ob = bpy.context.object
        edit = False
        if ob.mode == 'EDIT':
            edit = True
            bpy.ops.object.mode_set(mode='OBJECT')
        storage = bpy.context.scene.dynamic_tension_map_selection
        hide_unhide_store(ob, True, storage)
        bpy.ops.object.shape_from_uv()
        hide_unhide_store(ob, False, storage)
        if edit:
            bpy.ops.object.mode_set(mode='EDIT')            
        #toggle_enable(bpy.context.object, context)
        #prop_callback(bpy.context.scene, bpy.context)
        return {'FINISHED'}


class DynamicTensionMap(bpy.types.Panel):
    """Dynamic Tension Map Panel"""
    bl_label = "Dynamic Tension Map"
    bl_idname = "dynamic tension map"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'
    bl_category = "Extended Tools"
    gt_show = True
    
    def draw(self, context):
        scene = context.scene
        layout = self.layout
        col = layout.column()
        col.label(text="Dynamic Tension Map")
        ob = bpy.context.object
        if ob is not None:
            if ob.dten.enable:    
                col.alert=True
            if ob.type == 'MESH':
                col.prop(ob.dten ,"enable", text="Toggle Dynamic Tension Map", icon='MOD_TRIANGULATE')
                col = layout.column()
                col.prop(scene.dten ,"max_stretch", text="Max Stretch", slider=True)
                col = layout.column(align=True)
                col.prop(scene.dten ,"show_from_flat", text="Show Tension From Flat", icon='MESH_GRID')
                if ob.data.shape_keys is not None:
                    if len(ob.data.shape_keys.key_blocks) > 1:
                        col.prop(ob.dten ,"flat_pattern_on", text="Show Pattern", icon='OUTLINER_OB_LATTICE')
                        col = layout.column()
                        col.scale_y = 2.0
                        col.operator("object.dynamic_tension_map_update_pattern", text="Update Pattern")
                col = layout.column(align=True)

                #col.prop(scene.dten ,"map_percentage", text="Percentage Based", icon='STICKY_UVS_VERT')
                #col.prop(scene.dten ,"map_edge", text="Edge Difference", icon='UV_VERTEXSEL')        
                return
            
        col.label(text="Select Mesh Object")


# Register Clases -------------->>>
def register():
    create_properties()

    # Register all classes if this file loaded separately
    if __name__ in {'__main__', 'DynamicTensionMap'}:
        bpy.utils.register_module(__name__)

    bpy.app.handlers.scene_update_post.append(dynamic_tension_handler)

    # Add load handlers
    bpy.app.handlers.load_post.append(refresh_dynamic_tension_data)


def unregister():

    # Add load handlers
    bpy.app.handlers.load_post.remove(refresh_dynamic_tension_data)

    bpy.app.handlers.scene_update_post.remove(dynamic_tension_handler)

    remove_properties()

    # Unregister all classes if this file loaded individually
    if __name__ in {'__main__', 'DynamicTensionMap'}:
        bpy.utils.unregister_module(__name__)

    
if __name__ == "__main__":
    register()
    
