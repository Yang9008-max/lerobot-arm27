"""Load the actual URDF in MuJoCo and verify geometry/poses against source CAD."""
import argparse
import json
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from inventory import ROOT, sha256_file, write_json
from joint_evidence import Evidence
from link_mesh_io import decode_obj, decode_stl, pose, transform_triangles
from verify_joint_motion import rotation_about_line
from build_full_robot import MODEL, FULL, meter_pose, remove_camera_annotation


def read_urdf():
    xml=(MODEL/'hzu_visual.urdf').read_text(encoding='utf-8')
    root=ET.fromstring(xml)
    paths={m.get('filename') for m in root.findall('./link/visual/geometry/mesh')}
    assets={p:(MODEL/p).read_bytes() for p in paths}
    model=mujoco.MjModel.from_xml_string(xml,assets=assets)
    model.opt.gravity[:]=0
    model.vis.global_.offwidth=1200
    model.vis.global_.offheight=900
    model.vis.headlight.active=1
    model.vis.headlight.ambient[:]=[.55,.55,.55]
    model.vis.headlight.diffuse[:]=[.85,.85,.85]
    model.vis.headlight.specular[:]=[.15,.15,.15]
    return root,assets,model


def read_origin(element):
    return pose(list(map(float,element.get('xyz','0 0 0').split())),
                list(map(float,element.get('rpy','0 0 0').split())))


def validate(root,assets,model,manifest):
    e=Evidence()
    if sha256_file(MODEL/'hzu_visual.urdf') != manifest['urdf_sha256']:
        raise ValueError('URDF changed after export')
    if sha256_file(ROOT/FULL)!=manifest['source_assembly_sha256']:
        raise ValueError('Original assembly changed')
    checked={FULL}
    max_vertex_error=0.
    asset_bounds={}
    for key,row in manifest['assets'].items():
        if sha256_file(ROOT/row['source_mesh_path'])!=row['source_mesh_sha256']:
            raise ValueError('Source mesh changed')
        if sha256_file(MODEL/row['obj_path'])!=row['obj_sha256'] or sha256_file(MODEL/row['stl_path'])!=row['stl_sha256']:
            raise ValueError('Export mesh changed')
        source,_=decode_stl((ROOT/row['source_mesh_path']).read_bytes())
        if row['removed_annotation_source_indices']:
            source,removed=remove_camera_annotation(source)
            if removed!=row['removed_annotation_source_indices']: raise ValueError('Camera annotation changed')
        keep=np.ones(len(source),bool)
        for indices in row['removed_degenerate_indices_after_annotation'].values(): keep[indices]=False
        reference=source[keep]
        exported=decode_obj(assets[row['obj_path']])
        if exported.shape!=reference.shape: raise ValueError('Export triangle count mismatch')
        error=float(np.max(np.linalg.norm(exported-reference,axis=-1)))
        max_vertex_error=max(max_vertex_error,error)
        low,high=exported.min(axis=(0,1)),exported.max(axis=(0,1))
        asset_bounds[key]=np.array([[low[0] if i&1 else high[0],low[1] if i&2 else high[1],
                                     low[2] if i&4 else high[2]] for i in range(8)])
    print('All mesh vertices read back; max error m:',max_vertex_error,flush=True)
    full_instances={r['instance_id']:r for r in e.assembly(FULL)['instances']}
    basis=np.linalg.inv(meter_pose(full_instances['392']['matrix_mm']))
    visual_frames={}
    for link in root.findall('link'):
        for visual in link.findall('visual'):
            identifier=visual.get('name')[5:].replace('_','/')
            visual_frames[identifier]=(link.get('name'),read_origin(visual.find('origin')),visual.get('name'))
    if len(visual_frames)!=len(manifest['components']): raise ValueError('Missing visual instances')
    joints=[]
    for element in root.findall('joint'):
        joints.append({'name':element.get('name'),'parent':element.find('parent').get('link'),
                       'child':element.find('child').get('link'),'origin':read_origin(element.find('origin')),
                       'axis':np.array(list(map(float,element.find('axis').get('xyz').split())))})
    if model.njnt!=10 or len(joints)!=10: raise ValueError('Wrong active joint count')
    homes={}
    for c in manifest['components']:
        instance=full_instances[c['instance_id']]
        homes[c['instance_id']]=basis@meter_pose(instance['matrix_mm'])
        for path,digest in [(c['native_path'],c['native_sha256'])]+[(p,c['geometry_source_sha256']) for p in c['geometry_source_paths']]:
            if path and path not in checked:
                if sha256_file(ROOT/path)!=digest: raise ValueError('Original part changed: '+path)
                checked.add(path)
    qvalues=[np.zeros(10)]
    for i in range(10):
        for sign in [-1,1]:
            q=np.zeros(10);q[i]=sign*np.deg2rad(10);qvalues.append(q)
    qvalues.extend(np.random.default_rng(20261015).uniform(-.22,.22,(6,10)))
    data=mujoco.MjData(model)
    frames=manifest['link_frames_in_arm_m']
    reports=[]
    overall=0.
    for q in qvalues:
        data.qpos[:]=0
        by_joint={j['name']:angle for j,angle in zip(joints,q)}
        deltas={'base_link':np.eye(4)}
        for j in manifest['joints']:
            angle=by_joint[j['name']]
            axis=j['saved_pose_axis_in_arm_frame']
            deltas[j['child']]=deltas[j['parent']]@rotation_about_line(np.array(axis['point_mm'])/1000,axis['axis'],angle)
            data.joint(j['name']).qpos[:]=angle
        mujoco.mj_forward(model,data)
        max_body,max_visual,max_point=0.,0.,0.
        for name,home in frames.items():
            expected=deltas[name]@np.array(home)
            actual=np.eye(4);actual[:3,:3]=data.body(name).xmat.reshape(3,3);actual[:3,3]=data.body(name).xpos
            max_body=max(max_body,float(abs(expected-actual).max()))
        for c in manifest['components']:
            identifier=c['instance_id']
            link,local,visual_name=visual_frames[identifier]
            expected=deltas[c['link']]@homes[identifier]
            body=data.body(link)
            parent=np.eye(4);parent[:3,:3]=body.xmat.reshape(3,3);parent[:3,3]=body.xpos
            actual=parent@local
            max_visual=max(max_visual,float(abs(actual-expected).max()))
            # Check the engine's imported geometry pose, including compiler mesh recentering.
            geom_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,visual_name)
            if geom_id<0: raise ValueError('URDF visual was dropped: '+visual_name)
            mesh_id=model.geom_dataid[geom_id]
            mesh_rotation=np.empty(9)
            mujoco.mju_quat2Mat(mesh_rotation,model.mesh_quat[mesh_id])
            imported=np.eye(4)
            imported[:3,:3]=data.geom_xmat[geom_id].reshape(3,3)@mesh_rotation.reshape(3,3).T
            imported[:3,3]=data.geom_xpos[geom_id]-imported[:3,:3]@model.mesh_pos[mesh_id]
            corners=asset_bounds[c['asset']]
            a=corners@imported[:3,:3].T+imported[:3,3]
            b=corners@expected[:3,:3].T+expected[:3,3]
            max_point=max(max_point,float(np.max(np.linalg.norm(a-b,axis=1))))
        overall=max(overall,max_body,max_visual,max_point)
        reports.append({'q_rad':q.tolist(),'max_body_matrix_error':max_body,
                        'max_visual_matrix_error':max_visual,'max_imported_mesh_corner_error_m':max_point})
    result={'schema_version':1,'urdf_actually_loaded':True,'engine':'MuJoCo '+mujoco.__version__,
            'links':len(frames),'active_joints':model.njnt,'visual_geoms':model.ngeom,'unique_meshes':model.nmesh,
            'configurations':len(reports),'mesh_vertex_max_error_m':max_vertex_error,
            'max_pose_error':overall,'pose_checks':reports,'original_files_hash_checked':len(checked),
            'wheel_coaxial_checks':len(manifest['wheel_axis_checks']),
            'all_checks_passed':overall<1e-6 and max_vertex_error<1e-6,
            'scope':'all exported vertices, all occurrence poses, source hashes, four hub axes and 27 kinematic configurations',
            'not_validated':['physical collision/travel/dynamics','motor encoder mapping','passive suspension/rollers and differential drive'],
            'no_time_integration_performed':True,'mujoco_mass_inertia':'compiler numerical floors only; not physical parameters'}
    write_json(MODEL/'validation.json',result)
    if not result['all_checks_passed']: raise ValueError('Whole-robot validation failed: '+str(overall))
    print(json.dumps({k:v for k,v in result.items() if k not in ['pose_checks']},ensure_ascii=False),flush=True)


def render(model):
    data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    renderer=mujoco.Renderer(model,width=1200,height=900)
    options=mujoco.MjvOption()
    images=[]
    try:
        for azimuth,elevation,lookat,distance in [
            (125,-22,[-.24,0,.30],1.95), (220,-18,[-.24,0,.30],1.95),
            (135,-22,[-.665,0,.18],.44), (110,-24,[.03,0,-.07],.94)]:
            camera.azimuth,camera.elevation=azimuth,elevation
            camera.lookat[:]=lookat;camera.distance=distance
            renderer.update_scene(data,camera=camera,scene_option=options)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
            images.append(Image.fromarray(renderer.render()))
    finally: renderer.close()
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',29)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
    canvas=Image.new('RGB',(2400,1970),'#111923');draw=ImageDraw.Draw(canvas)
    titles=['整机：底盘、四轮与机械臂','整机：另一侧','末端：GM6020、两个 M2006 与吸盘','底盘：4310、四个 3508、轮组与减震器']
    for i,img in enumerate(images):
        x,y=(i%2)*1200,70+(i//2)*930
        canvas.paste(img,(x,y));draw.text((x+20,y-43),titles[i],font=font,fill='white')
    draw.text((20,1930),'实际 hzu_visual.urdf 加载渲染 · 578 个显示实例 · 6 个机械臂输出关节 + 4 个轮轴 · 几何/运动学版',font=small,fill='#cbd7e3')
    canvas.save(MODEL/'whole_robot_overview.png')
    images[0].save(MODEL/'whole_robot.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--view',action='store_true',help='Open joint-slider kinematic inspection without integrating dynamics')
    args=parser.parse_args()
    root,assets,model=read_urdf()
    print('Loaded actual URDF:',model.nbody,model.njnt,model.ngeom,model.nmesh,flush=True)
    if args.view:
        import mujoco.viewer
        data=mujoco.MjData(model)
        with mujoco.viewer.launch_passive(model,data) as viewer:
            viewer.cam.lookat[:]=[-.24,0,.30];viewer.cam.distance=1.95
            while viewer.is_running():
                mujoco.mj_forward(model,data)
                viewer.sync();time.sleep(.02)
    else:
        manifest=json.loads((MODEL/'model_manifest.json').read_text(encoding='utf-8'))
        validate(root,assets,model,manifest)
        render(model)


if __name__=='__main__': main()
