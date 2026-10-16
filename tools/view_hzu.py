"""Inspect the HZU model without connecting hardware or integrating dynamics."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def load_model(directory):
    text=(directory/'hzu_visual.urdf').read_text(encoding='utf-8')
    tree=ET.fromstring(text)
    assets={element.get('filename'):(directory/element.get('filename')).read_bytes()
            for element in tree.findall('./link/visual/geometry/mesh')}
    manifest=json.loads((directory/'model_manifest.json').read_text(encoding='utf-8'))
    if hashlib.sha256(text.encode('utf-8')).hexdigest()!=manifest['urdf_sha256']:
        raise ValueError('URDF hash mismatch')
    for asset in manifest['assets'].values():
        if hashlib.sha256(assets[asset['obj_path']]).hexdigest()!=asset['obj_sha256']:
            raise ValueError('Mesh hash mismatch: '+asset['obj_path'])
    model=mujoco.MjModel.from_xml_string(text,assets=assets)
    model.opt.gravity[:]=0
    model.vis.headlight.active=1
    model.vis.headlight.ambient[:]=[.55,.55,.55]
    model.vis.headlight.diffuse[:]=[.85,.85,.85]
    model.vis.headlight.specular[:]=[.15,.15,.15]
    model.vis.global_.offwidth=1200
    model.vis.global_.offheight=900
    expected=manifest['summary']
    if (model.njnt,model.ngeom,model.nmesh)!=(10,expected['visual_components'],expected['unique_meshes']):
        raise ValueError('Imported structure differs from distribution manifest')
    return model,manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=Path(__file__).resolve().parents[1]/'models/hzu_robot')
    parser.add_argument('--check',action='store_true',help='Verify hashes and MuJoCo loading without opening a window')
    parser.add_argument('--render',type=Path,help='Save a 1200x900 image instead of opening the viewer')
    args=parser.parse_args()
    model,manifest=load_model(args.model)
    data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    # Exercise every coordinate in the imported file; no motor commands or mj_step.
    for name in [j['name'] for j in manifest['joints']]:
        for angle in [-.1,.1]:
            data.qpos[:]=0
            data.joint(name).qpos[:]=angle
            mujoco.mj_forward(model,data)
            if not np.isfinite(data.geom_xpos).all(): raise ValueError('Nonfinite visual pose')
    data.qpos[:]=0
    mujoco.mj_forward(model,data)
    print(json.dumps({'mujoco':mujoco.__version__,'active_joints':model.njnt,'visual_instances':model.ngeom,
                      'unique_meshes':model.nmesh,'asset_hashes_verified':True,'coordinate_checks':20,
                      'dynamics_integrated':False}))
    if args.check: return
    if args.render:
        from PIL import Image
        camera=mujoco.MjvCamera()
        camera.azimuth,camera.elevation=125,-22
        camera.lookat[:]=[-.24,0,.30];camera.distance=1.95
        with mujoco.Renderer(model,width=1200,height=900) as renderer:
            renderer.update_scene(data,camera=camera)
            Image.fromarray(renderer.render()).save(args.render)
        return
    import mujoco.viewer as viewer_module
    with viewer_module.launch_passive(model,data) as viewer:
        viewer.cam.lookat[:]=[-.24,0,.30];viewer.cam.distance=1.95
        viewer.cam.azimuth,viewer.cam.elevation=125,-22
        while viewer.is_running():
            mujoco.mj_forward(model,data)
            viewer.sync()
            time.sleep(.02)


if __name__=='__main__': main()
