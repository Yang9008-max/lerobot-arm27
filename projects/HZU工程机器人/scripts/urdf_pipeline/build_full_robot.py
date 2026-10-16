"""Build a traceable whole-robot visual/kinematic URDF from saved CAD evidence.

This is intentionally not a collision/dynamics model. No physical limits,
inertias, differential actuator mapping or suspension constraints are invented.
"""
import copy
import json
import warnings
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation

from inventory import ROOT, sha256_file, write_json
from joint_evidence import Evidence, load, line_relation, transform_axis
from link_mesh_io import (clean_degenerate_triangles, decode_stl, encode_obj,
                          encode_stl, link_frames, pose, require_rigid)

FULL = 'HZU工程完整装配.SLDASM'
MODEL = ROOT / 'models/hzu_robot'


def meter_pose(matrix_mm):
    matrix = np.array(matrix_mm, dtype=float)
    matrix[:3, 3] /= 1000
    return require_rigid(matrix)


def origin_fields(matrix):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)  # Euler singularities have exact alternatives.
        rpy = Rotation.from_matrix(matrix[:3, :3]).as_euler('xyz')
    xyz = matrix[:3, 3]
    if np.max(abs(pose(xyz, rpy)-matrix)) > 1e-9:
        raise ValueError('URDF RPY does not preserve saved rotation')
    return {'xyz':' '.join(format(v,'.16g') for v in xyz),
            'rpy':' '.join(format(v,'.16g') for v in rpy)}


def remove_camera_annotation(triangles):
    # Connectivity is only used for selection: original vertices are not welded or changed.
    vertices, inverse = np.unique(np.round(triangles.reshape(-1, 3), 7), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    a, b = np.r_[faces[:,0],faces[:,1],faces[:,2]], np.r_[faces[:,1],faces[:,2],faces[:,0]]
    _, labels = connected_components(coo_matrix((np.ones(len(a)),(a,b)),
                                   shape=(len(vertices),len(vertices))).tocsr(), directed=False)
    groups = labels[faces[:,0]]
    removed = []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        cloud = triangles[indices]
        if (cloud.max(axis=(0,1))-cloud.min(axis=(0,1))).max() > .3:
            removed.extend(indices.tolist())
    if len(removed) != 100:
        raise ValueError('Camera FOV component differs from inspected 100-triangle cone')
    keep = np.ones(len(triangles), bool)
    keep[removed] = False
    return triangles[keep], removed


def material(name, native_path):
    text = name + ' ' + (native_path or '')
    if any(s in text for s in ['电机','GM6020','麦轮','真空泵']):
        return 'motor_dark', [.13,.15,.18,1]
    if any(s in text for s in ['打印件','带轮3M','吸盘锥齿轮']):
        return 'printed_blue', [.19,.43,.67,1]
    if any(s in text for s in ['铝方管','轴承','铝柱','联轴器','加工轴','角连接','螺钉','螺栓','螺母','平垫']):
        return 'metal', [.70,.73,.77,1]
    if '吸盘' in text:
        return 'suction', [.25,.29,.33,1]
    if any(s in text for s in ['装甲','保护壳']):
        return 'cover', [.66,.22,.21,1]
    return 'structure', [.36,.40,.45,1]


def main():
    e = Evidence()
    full = e.assembly(FULL)['instances']
    by_id = {r['instance_id']:r for r in full}
    parents = {r['parent_instance_id'] for r in full}
    basis = np.linalg.inv(meter_pose(by_id['392']['matrix_mm']))
    mesh_rows = {r['source_sha256']:r for r in load('urdf_work/evidence/native_mesh_parse.json')['results']}
    mesh_rows.update({r['source_sha256']:r for r in load('urdf_work/evidence/native_mesh_retry.json')['results']
                      if r.get('mesh_path') and r['status']=='parsed_display_mesh'})
    native_map = {}
    for comparison in load('urdf_work/evidence/joint_saved_contexts.json')['comparisons']:
        for r in comparison['instances']:
            native_map[r['full_instance_id']] = (comparison['standalone_path'],r['own_instance_id'])

    joints = copy.deepcopy(load('urdf_work/config/joints.json')['joints'])
    joints[0]['parent'] = 'base_link'
    for j in joints:
        j['export_type'] = 'continuous'
        j['limit_policy'] = 'unbounded inspection coordinate; physical travel limits remain unknown'
    frames = link_frames({'joints':joints})
    wheel_members = {}
    wheel_checks = []
    for prefix, wheel_id, couplers, bearing in [
        ('626','629',['638','644'],'642'), ('683','686',['699','700'],'694'),
        ('712','629',['638','644'],'642'), ('713','686',['699','700'],'694')]:
        instance = by_id[prefix+'/'+wheel_id]
        world = basis @ meter_pose(instance['matrix_mm'])
        # The wheel's native Z axis passes through the hub. Use the saved hub mid-plane.
        bounds = np.array(list(map(float,instance['model_attributes']['swBoundingBox'].split())))
        center = (bounds[2]+bounds[5])/2
        point = world[:3,3] + world[:3,:3] @ [0,0,center]
        axis = world[:3,:3] @ [0,0,1]
        link = 'wheel_'+prefix
        frame = np.eye(4); frame[:3,3] = point
        frames[link] = frame
        j = {'name':'joint_wheel_'+prefix,'parent':'base_link','child':link,'export_type':'continuous',
             'origin_xyz_m':point.tolist(),'origin_rpy_rad':[0,0,0],'axis':axis.tolist(),
             'saved_pose_axis_in_arm_frame':{'point_mm':(point*1000).tolist(),'axis':axis.tolist()},
             'limit_policy':'wheel rotation; roller and suspension motion held at saved pose'}
        joints.append(j)
        reference = {'point_mm':(point*1000).tolist(),'axis':axis.tolist()}
        for component in couplers+[bearing]:
            pose_mm = basis @ meter_pose(by_id[prefix+'/'+component]['matrix_mm'])
            pose_mm[:3,3] *= 1000
            direction = [0,1,0] if component==bearing else [0,0,1]
            line = transform_axis(pose_mm,np.zeros(3),direction,{'instance_id':prefix+'/'+component})
            relation = line_relation(reference,line)
            if relation['line_distance_mm'] > .001 or relation['parallel_error'] > 1e-8:
                raise ValueError('Wheel/hub/bearing axes disagree: '+str(relation))
            wheel_checks.append({'wheel':prefix,'component':component,**relation})
        for component in [wheel_id]+couplers:
            wheel_members[prefix+'/'+component] = link

    def membership(identifier):
        if identifier in wheel_members: return wheel_members[identifier], 'coaxial wheel and hub output'
        if identifier in ['4/257','4/263']: return 'yaw_base','J1 output pulley and upper mounting pad'
        own = native_map.get(identifier)
        if own and '机械臂装配' in own[0]:
            number = own[1]
            if number.startswith('4/'): return 'yaw_base','arm base assembly; motor housings and passive internals held at saved angle'
            if number.startswith('101/') or number.startswith('212'):
                return 'upper_arm','upper arm and elbow motor housing'
            if number.startswith('153/') or number in ['203','206','209','224']:
                return 'forearm','forearm and attached camera; bearing races not split'
            return 'yaw_base','UWB mast mounted to yaw assembly'
        if own and '末端装配' in own[0]:
            number=own[1]
            if number=='4': return 'forearm','GM6020 housing before J4 output'
            if number in ['34','37','63','81','84','87']:
                return 'suction_tool','tool output; passive universal joint held at saved angle'
            if number in ['66','73','76','77','78','90','91','94']:
                return 'wrist_carrier','pitch carrier; composite bearing outer race approximation'
            return 'wrist_housing','wrist housing; differential input gear angles held at saved values'
        return 'base_link','chassis assembly at saved pose; suspension and auxiliary mechanisms held fixed'

    proxies = {'392/395/435/438':('8009-RAU_5005_外形尺寸-20240809.stp',.05),
               '392/603/438':('8009-RAU_5005_外形尺寸-20240809.stp',.05),
               '930':('保护壳/VT307V-..0).stp',.05)}
    for prefix in ['908','923','924','925','926','927','928','929']:
        proxies[prefix]=('其他/新建文件夹/KJ85T33装配体.STEP',.75)
    selected, coverage = [], []
    for instance in full:
        identifier = instance['instance_id']
        covered = next((p for p in proxies if identifier.startswith(p+'/')),None)
        if covered:
            coverage.append({'instance_id':identifier,'name':instance['name'],'status':'represented_by_STEP_assembly',
                             'representative_instance_id':covered})
            continue
        if identifier in parents and identifier not in proxies: continue
        row = {'instance_id':identifier,'name':instance['name']}
        if instance['hidden_in_tree'] or instance['suppressed_in_tree']:
            coverage.append({**row,'status':'excluded_hidden_or_suppressed'}); continue
        if '遮挡范围' in instance['name'] or instance['name']=='金矿石':
            coverage.append({**row,'status':'excluded_nonrobot_annotation_or_grasped_object'}); continue
        key=instance.get('source_sha256')
        source=mesh_rows.get(key,{})
        if identifier in proxies:
            suffix,tolerance=proxies[identifier]
            key, geometry = next((k,g) for k,g in e.geometry.items() if any(p.endswith(suffix) for p in g['source_paths']))
            source={'mesh_path':geometry['mesh_path'],'source_paths':geometry['source_paths'],'status':'STEP_assembly_proxy'}
            bounds=np.array([*geometry['bbox_min_mm'],*geometry['bbox_max_mm']])/1000
            stored=np.array(list(map(float,instance['model_attributes']['swBoundingBox'].split())))
            if np.max(abs(bounds-stored))*1000 > tolerance:
                raise ValueError('STEP assembly bounds mismatch '+identifier)
        if not source.get('mesh_path'):
            raise ValueError('Unrepresented robot occurrence '+str(row))
        link,basis_text=membership(identifier)
        world=basis @ meter_pose(instance['matrix_mm'])
        native_path=instance['source_resolution'].get('path')
        selected.append({**row,'link':link,'assignment_basis':basis_text,
                         'native_path':native_path,'native_sha256':instance.get('source_sha256'),
                         'geometry_source_sha256':key,'geometry_source_paths':source['source_paths'],
                         'source_mesh_path':source['mesh_path'],'geometry_method':source['status'],
                         'part_in_arm_m':world.tolist(),'part_in_link_m':(np.linalg.inv(frames[link])@world).tolist()})
        coverage.append({**row,'status':'included','link':link})

    out=MODEL/'meshes/visual'; out.mkdir(parents=True,exist_ok=True)
    assets,cache={},{}
    for component in selected:
        path=component['source_mesh_path']
        asset_key=component['geometry_source_sha256']
        if asset_key not in assets:
            original,_=decode_stl((ROOT/path).read_bytes())
            camera='图传云台-含视场角' in component['name']
            triangles,annotation = remove_camera_annotation(original) if camera else (original,[])
            triangles,removed=clean_degenerate_triangles(triangles,triangles)
            stl,obj=out/(asset_key+'.stl'),out/(asset_key+'.obj')
            stl.write_bytes(encode_stl(triangles)); obj.write_bytes(encode_obj(triangles))
            bounds=np.r_[original.min(axis=(0,1)),original.max(axis=(0,1))]
            assets[asset_key]={'source_mesh_path':path,'source_mesh_sha256':sha256_file(ROOT/path),
                               'obj_path':obj.relative_to(MODEL).as_posix(),'obj_sha256':sha256_file(obj),
                               'stl_path':stl.relative_to(MODEL).as_posix(),'stl_sha256':sha256_file(stl),
                               'original_bbox_m':bounds.tolist(),'source_triangles':len(original),'triangles':len(triangles),
                               'removed_annotation_source_indices':annotation,'removed_degenerate_indices_after_annotation':removed}
        asset=assets[asset_key]
        component['asset']=asset_key
        component['source_mesh_sha256']=asset['source_mesh_sha256']
        stored=np.array(list(map(float,by_id[component['instance_id']]['model_attributes']['swBoundingBox'].split())))
        bounds=np.array(asset['original_bbox_m'])
        outside=max(0.,float(max((stored[:3]-bounds[:3]).max(),(bounds[3:]-stored[3:]).max())*1000))
        if outside > .02:
            raise ValueError('Mesh extends outside saved native bounds '+str((component['name'],outside)))
        component['bbox_validation']={'outside_saved_bounds_mm':outside,
                                      'max_absolute_bound_difference_mm':float(abs(bounds-stored).max()*1000),
                                      'policy':'native bounds may enclose sketches, wires or conservative curved-surface bounds; no rescale or recenter'}

    root=ET.Element('robot',name='hzu_engineer_visual_kinematic')
    root.append(ET.Comment('Visual/kinematic model only. q=0 is saved CAD pose. No physical masses, collision meshes or travel limits.'))
    extension=ET.SubElement(root,'mujoco')
    extension.append(ET.Comment('Numerical minimum mass/inertia ONLY to compile a visual inspection model; never use as measured dynamics.'))
    ET.SubElement(extension,'compiler',discardvisual='false',fusestatic='false',boundmass='0.001',boundinertia='0.000001')
    links={name:ET.SubElement(root,'link',name=name) for name in frames}
    for index,c in enumerate(selected):
        visual=ET.SubElement(links[c['link']],'visual',name='part_'+c['instance_id'].replace('/','_'))
        ET.SubElement(visual,'origin',**origin_fields(np.array(c['part_in_link_m'])))
        geometry=ET.SubElement(visual,'geometry')
        ET.SubElement(geometry,'mesh',filename=assets[c['asset']]['obj_path'],scale='1 1 1')
        mat,rgba=material(c['name'],c['native_path'])
        color=ET.SubElement(visual,'material',name=mat)
        ET.SubElement(color,'color',rgba=' '.join(map(str,rgba)))
    for j in joints:
        element=ET.SubElement(root,'joint',name=j['name'],type=j['export_type'])
        ET.SubElement(element,'parent',link=j['parent']); ET.SubElement(element,'child',link=j['child'])
        ET.SubElement(element,'origin',**origin_fields(pose(j['origin_xyz_m'],j['origin_rpy_rad'])))
        ET.SubElement(element,'axis',xyz=' '.join(map(str,j['axis'])))
    ET.indent(root)
    urdf=MODEL/'hzu_visual.urdf'
    urdf.write_bytes(ET.tostring(root,encoding='utf-8',xml_declaration=True))
    manifest={'schema_version':1,'profile':'whole_robot_visual_kinematic_only','reference_frame':'saved arm CAD A; meters; q0 saved CAD pose, not encoder zero',
              'urdf':'hzu_visual.urdf','urdf_sha256':sha256_file(urdf),'original_cad_modified':False,
              'source_assembly':FULL,'source_assembly_sha256':e.records[FULL]['sha256'],
              'link_frames_in_arm_m':{k:v.tolist() for k,v in frames.items()},'joints':joints,
              'links':{k:[c['instance_id'] for c in selected if c['link']==k] for k in frames},
              'assets':assets,'components':selected,'coverage':coverage,'wheel_axis_checks':wheel_checks,
              'limitations':['No collision geometry or measured mass/inertia; MuJoCo compiler floors are inspection-only numerical placeholders',
                             'Arm physical travel/effort/velocity limits and encoder zeros unknown; continuous coordinates are for inspection only',
                             'Motor stators/rotors and bearing races are composite visuals; drivetrain angles held at saved values',
                             'Suspension, wheel rollers, universal joints, storage and protection mechanisms held at saved CAD pose',
                             'Differential motor-to-output signs and closed-loop drive not calibrated; LeRobot adapter not provided'],
              'summary':{'links':len(frames),'arm_output_joints':6,'wheel_output_joints':4,'visual_components':len(selected),
                         'unique_meshes':len(assets),'instanced_triangles':sum(assets[c['asset']]['triangles'] for c in selected),
                         'unrepresented_visible_robot_occurrences':0,'coverage_status':dict(Counter(r['status'] for r in coverage))}}
    write_json(MODEL/'model_manifest.json',manifest)
    write_json(MODEL/'joint_order.json',{'arm':[j['name'] for j in joints[:6]],'wheels':[j['name'] for j in joints[6:]],
                                       'units':'rad','q0':'saved CAD pose','not_encoder_zero':True,'suction':'separate boolean; not simulated'})
    print(json.dumps(manifest['summary'],ensure_ascii=False))


if __name__=='__main__': main()
