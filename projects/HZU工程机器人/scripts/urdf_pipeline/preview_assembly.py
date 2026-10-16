"""Render bbox-supported meshes at the saved CAD pose for inspection only."""
from __future__ import annotations

import json
import struct
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image

from inventory import ROOT, write_json


def main():
    assembly = json.loads((ROOT / "urdf_work/config/assembly.json").read_text(encoding="utf-8"))
    out = ROOT / "urdf_work/inspection"
    out.mkdir(parents=True, exist_ok=True)
    root = ET.Element("mujoco", model="saved_cad_pose_inspection_only")
    ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="false")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="800")
    ET.SubElement(visual, "rgba", haze="0.07 0.09 0.12 1")
    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="-0.3 -1 1", dir="0 1 -1", diffuse="0.8 0.8 0.8")
    meshes = {}
    virtual_assets = {}
    drawn = []
    for instance in assembly["instances"]:
        check = instance["geometry_check"]
        if not check.get("mesh_path") or instance["suppressed_in_tree"] or instance["hidden_in_tree"]:
            continue
        if instance["name"].casefold().startswith("uwb"):
            continue  # Keep the close inspection view focused on the arm frame.
        key = check["geometry_sha256"]
        if key not in meshes:
            name = "mesh_" + str(len(meshes))
            mesh_filename = name + ".stl"
            mesh_bytes = (ROOT / check["mesh_path"]).read_bytes()
            # MuJoCo's STL decoder caps each mesh at 200k faces. OBJ preserves
            # the same triangles without this format-specific decoder cap.
            faces = struct.unpack_from("<I", mesh_bytes, 80)[0]
            if faces > 200000:
                dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
                triangles = np.frombuffer(mesh_bytes, dtype=dtype, offset=84)["vertices"]
                vertices, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
                lines = ["v " + " ".join(format(float(x), ".9g") for x in v) for v in vertices]
                lines.extend("f " + " ".join(str(int(x)+1) for x in face) for face in inverse.reshape(-1, 3))
                mesh_bytes = ("\n".join(lines) + "\n").encode("ascii")
                mesh_filename = name + ".obj"
            (out / mesh_filename).write_bytes(mesh_bytes)
            virtual_assets[mesh_filename] = mesh_bytes
            ET.SubElement(asset, "mesh", name=name, file=mesh_filename)
            meshes[key] = name
        matrix = np.array(instance["matrix_mm"])
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, matrix[:3, :3].reshape(9))
        color = "0.38 0.70 0.96 1" if instance["instance_id"].startswith("4/") else \
                "0.96 0.66 0.32 1" if instance["instance_id"].startswith("101/") else "0.48 0.84 0.64 1"
        ET.SubElement(world, "geom", name="instance_" + instance["instance_id"].replace("/", "_"), type="mesh",
                      mesh=meshes[key], pos=" ".join(map(str, matrix[:3, 3] / 1000)),
                      quat=" ".join(map(str, quaternion)), rgba=color, contype="0", conaffinity="0")
        drawn.append({"instance_id": instance["instance_id"], "name": instance["name"], "mesh_path": check["mesh_path"]})
    # Measured axis markers, never active simulation joints.
    for index, x in enumerate([0, -.3]):
        ET.SubElement(world, "geom", name=f"measured_axis_marker_{index}", type="cylinder", size="0.002",
                      fromto=f"{x} -0.075 0.175 {x} 0.075 0.175", rgba="1 0.25 0.30 1", contype="0", conaffinity="0")
    filename = out / "arm_saved_pose.xml"
    ET.indent(root)
    ET.ElementTree(root).write(filename, encoding="utf-8", xml_declaration=True)
    # MuJoCo's Windows C file API does not open this Unicode workspace path.
    # In-memory XML plus its exact mesh bytes avoids changing system paths.
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), assets=virtual_assets)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [-.26, 0, .12]
    camera.distance = 1.05
    camera.azimuth = 110
    camera.elevation = -28
    renderer = mujoco.Renderer(model, height=800, width=1280)
    try:
        renderer.update_scene(data, camera=camera)
        Image.fromarray(renderer.render()).save(out / "arm_saved_pose.png")
    finally:
        renderer.close()
    write_json(out / "preview_manifest.json", {"purpose": "saved_pose_geometry_inspection_only", "instances": drawn,
               "mesh_instances": len(drawn), "unique_meshes": len(meshes), "simulation_joints": model.njnt,
               "all_visual_geoms_disable_contact": True,
               "limitations": "Incomplete source mesh coverage; UWB accessories omitted from close view; colors distinguish saved subassemblies, not rigid robot links"})
    print(json.dumps({"rendered_mesh_instances": len(drawn), "unique_meshes": len(meshes), "simulation_joints": model.njnt}))


if __name__ == "__main__":
    main()
