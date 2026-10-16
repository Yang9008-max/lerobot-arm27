"""Read the exported files back and compare posed vertices with source CAD poses."""
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from inventory import ROOT, sha256_file, write_json
from joint_evidence import Evidence, load
from link_mesh_io import clean_degenerate_triangles, decode_obj, decode_stl, link_frames, pose, transform_triangles
from verify_joint_motion import rotation_about_line


MODEL = ROOT / "models/hzu_arm"


def max_distance(a, b):
    return float(np.sqrt(np.max(np.sum((a-b)**2, axis=-1))))


def source_pose(e, component, joints):
    # Recover the pose anew from native-instance evidence, not from exported
    # part_in_link/home transforms. Geometry vertices are already in meters.
    assembly = component["assembly_path"]
    instance = e.instance(assembly, component["instance_id"])
    matrix = np.array(instance["matrix_mm"])
    if assembly == "末端零件/末端装配.SLDASM":
        matrix = np.array(joints["end_assembly_in_arm_matrix_mm"]) @ matrix
    elif assembly == "HZU工程完整装配.SLDASM":
        arm = np.array(e.instance(assembly, "392")["matrix_mm"])
        matrix = np.linalg.inv(arm) @ matrix
    matrix[:3, 3] /= 1000
    return matrix


def inspection_model(manifest, joints):
    root = ET.Element("mujoco", model="exported_link_visual_mesh_readback_only")
    ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="false", fusestatic="false")
    ET.SubElement(root, "option", gravity="0 0 0")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="900", offheight="650")
    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="-.3 -1 1.5", dir="0 1 -1", diffuse=".9 .9 .9")
    frames = link_frames(joints)
    root_name = next(iter(frames))
    bodies = {root_name: ET.SubElement(world, "body", name=root_name)}
    for joint in joints["joints"]:
        local = pose(joint["origin_xyz_m"], joint["origin_rpy_rad"])
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, local[:3, :3].reshape(9))
        body = ET.SubElement(bodies[joint["parent"]], "body", name=joint["child"],
                             pos=" ".join(map(str, local[:3, 3])), quat=" ".join(map(str, quat)))
        ET.SubElement(body, "inertial", pos="0 0 0", mass="1", diaginertia=".01 .01 .01")
        ET.SubElement(body, "joint", name=joint["name"], type="hinge", limited="false",
                      axis=" ".join(map(str, joint["axis"])))
        bodies[joint["child"]] = body
    assets = {}
    for link in manifest["links"]:
        filename = link["name"] + ".obj"
        assets[filename] = (MODEL / link["obj_path"]).read_bytes()
        ET.SubElement(asset, "mesh", name=link["name"], file=filename)
        ET.SubElement(bodies[link["name"]], "geom", name=link["name"] + "_visual", type="mesh",
                      mesh=link["name"], rgba=" ".join(map(str, link["rgba"])), contype="0", conaffinity="0")
    ET.indent(root)
    # Files are referenced relative to this portable package, independent of
    # source CAD locations. C API on Windows uses the same bytes through VFS.
    for element in asset:
        element.set("file", "meshes/visual/" + element.get("file"))
    xml = ET.tostring(root, encoding="unicode")
    (MODEL / "link_mesh_inspection.xml").write_text(xml, encoding="utf-8")
    assets = {"meshes/visual/" + name: data for name, data in assets.items()}
    return mujoco.MjModel.from_xml_string(xml, assets=assets)


def validate(manifest, joints, model):
    e = Evidence()
    for path, expected in {**manifest["input_artifacts"], **manifest["verified_input_files"]}.items():
        if sha256_file(ROOT / path) != expected:
            raise ValueError(f"Input changed since export: {path}")
    source_cache, exported, checks = {}, {}, []
    for link in manifest["links"]:
        for extension in ["stl", "obj"]:
            path = MODEL / link[extension + "_path"]
            if sha256_file(path) != link[extension + "_sha256"]:
                raise ValueError(f"Output hash mismatch: {path}")
        stl, normals = decode_stl((MODEL / link["stl_path"]).read_bytes())
        obj = decode_obj((MODEL / link["obj_path"]).read_bytes())
        if len(stl) != link["triangles"] or len(obj) != len(stl):
            raise ValueError("Exported triangle count changed")
        obj_cross = np.cross(obj[:, 1]-obj[:, 0], obj[:, 2]-obj[:, 0])
        obj_degenerate = int(np.all(obj_cross == 0, axis=1).sum())
        if obj_degenerate:
            raise ValueError("Degenerate faces remain in output OBJ")
        crossed = np.cross(stl[:, 1]-stl[:, 0], stl[:, 2]-stl[:, 0])
        lengths = np.linalg.norm(crossed, axis=1)
        nonzero = lengths > 0
        alignment = np.sum(normals[nonzero] * (crossed[nonzero]/lengths[nonzero, None]), axis=1)
        if alignment.min(initial=1) < 1-1e-6:
            raise ValueError("STL normals do not agree with face winding")
        offset = 0
        for part in link["components"]:
            if part["triangle_start"] != offset or part["triangle_count"] <= 0:
                raise ValueError("Component triangle ranges overlap or contain gaps")
            offset += part["triangle_count"]
            path = part["source_mesh_path"]
            if path not in source_cache:
                source_cache[path] = decode_stl((ROOT / path).read_bytes())[0]
            if len(source_cache[path]) != part["source_triangle_count"]:
                raise ValueError("Source triangle count changed")
            native = source_pose(e, part, joints)
            if np.max(abs(native-part["part_in_arm_m"])) > 1e-9:
                raise ValueError("Export manifest does not match saved native component pose")
            part["_native_home_m"] = native
            frames = link_frames(joints)
            local = transform_triangles(source_cache[path], np.linalg.inv(frames[link["name"]]) @ native)
            _, removed = clean_degenerate_triangles(local, source_cache[path])
            if removed != part["removed_source_triangle_indices"]:
                raise ValueError("Unexplained source triangle removal")
            keep = np.ones(len(source_cache[path]), dtype=bool)
            for indices in removed.values():
                keep[indices] = False
            if int(keep.sum()) != part["triangle_count"]:
                raise ValueError("Retained source triangles do not match exported count")
            part["_source_keep"] = keep
        if offset != len(stl):
            raise ValueError("Component ranges do not cover full output mesh")
        exported[link["name"]] = {"stl": stl, "obj": obj}
        checks.append({"link": link["name"], "components": len(link["components"]), "triangles": len(stl),
                       "stl_obj_max_vertex_difference_m": max_distance(stl, obj),
                       "degenerate_triangles": int((~nonzero).sum()),
                       "obj_degenerate_triangles": obj_degenerate,
                       "normal_min_alignment": float(alignment.min(initial=1))})
        if not nonzero.all():
            raise ValueError("Degenerate faces remain in output STL")

    qvalues = [np.zeros(len(joints["joints"]))]
    for i in range(len(joints["joints"])):
        for sign in [-1, 1]:
            q = np.zeros(len(joints["joints"]))
            q[i] = sign*np.deg2rad(10)
            qvalues.append(q)
    qvalues.extend(np.random.default_rng(20261015).uniform(-.25, .25, (8, len(joints["joints"]))))
    data = mujoco.MjData(model)
    pose_checks = []
    for q in qvalues:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        screw = np.eye(4)
        independent = {joints["joints"][0]["parent"]: screw.copy()}
        for angle, joint in zip(q, joints["joints"]):
            axis = joint["saved_pose_axis_in_arm_frame"]
            screw = screw @ rotation_about_line(np.array(axis["point_mm"])/1000, axis["axis"], angle)
            independent[joint["child"]] = screw.copy()
        errors = {"stl": 0., "obj": 0.}
        for link in manifest["links"]:
            body = data.body(link["name"])
            actual_frame = np.eye(4)
            actual_frame[:3, :3] = body.xmat.reshape(3, 3)
            actual_frame[:3, 3] = body.xpos
            for part in link["components"]:
                begin, count = part["triangle_start"], part["triangle_count"]
                expected = transform_triangles(source_cache[part["source_mesh_path"]][part["_source_keep"]],
                                               independent[link["name"]] @ part["_native_home_m"])
                for extension in errors:
                    drawn = transform_triangles(exported[link["name"]][extension][begin:begin+count], actual_frame)
                    errors[extension] = max(errors[extension], max_distance(drawn, expected))
        pose_checks.append({"angles_rad": q.tolist(), "stl_max_vertex_error_m": errors["stl"],
                            "obj_max_vertex_error_m": errors["obj"]})
    maximum_stl = max(r["stl_max_vertex_error_m"] for r in pose_checks)
    maximum_obj = max(r["obj_max_vertex_error_m"] for r in pose_checks)
    return {"schema_version": 1, "purpose": "link-local visual mesh export readback and source-pose preservation",
            "links": len(checks), "visual_components": sum(c["components"] for c in checks),
            "triangles": sum(c["triangles"] for c in checks), "mujoco_active_joints": model.njnt,
            "configurations_checked": len(qvalues), "link_checks": checks, "pose_checks": pose_checks,
            "max_stl_reassembly_vertex_error_m": maximum_stl, "max_obj_reassembly_vertex_error_m": maximum_obj,
            "tolerance_m": 1e-6, "all_checks_passed": maximum_stl < 1e-6 and maximum_obj < 1e-6,
            "mesh_hashes_and_source_hashes_verified": True, "all_retained_source_triangles_preserved": True,
            "source_triangles": manifest["summary"]["source_triangles"],
            "removed_degenerate_triangles": manifest["summary"]["removed_degenerate_triangles"],
            "all_triangle_removals_explained_and_verified": True,
            "limits": "Numerical mesh preservation only; not full CAD rigidity, travel/collision or dynamic validation",
            "mass_and_inertia_in_inspection_model": "explicit numerical placeholders; not physical parameters"}


def render(model, manifest, joints):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation = 110, -25
    renderer = mujoco.Renderer(model, width=900, height=650)
    frames = []
    original = model.body_pos.copy()
    offsets = {"chassis_mount": [0, 0, -.08], "yaw_base": [0, .10, -.03], "upper_arm": [0, 0, .10],
               "forearm": [-.02, -.10, .22], "wrist_housing": [-.12, .08, .32],
               "wrist_carrier": [-.23, -.08, .39], "suction_tool": [-.34, .02, .46]}
    try:
        for exploded in [False, True]:
            if exploded:
                parents = {j["child"]: j["parent"] for j in joints["joints"]}
                for link in manifest["links"]:
                    name = link["name"]
                    parent_offset = np.array(offsets[parents[name]]) if name in parents else 0
                    model.body_pos[model.body(name).id] += np.array(offsets[name])-parent_offset
                mujoco.mj_forward(model, data)
            camera.lookat[:] = [-.30, 0, .14] if not exploded else [-.42, 0, .30]
            camera.distance = 1.18 if not exploded else 1.65
            renderer.update_scene(data, camera=camera)
            frames.append(Image.fromarray(renderer.render()))
    finally:
        model.body_pos[:] = original
        renderer.close()
    canvas = Image.new("RGB", (1800, 810), "#101821")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 23)
    small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 17)
    draw.text((20, 12), "按 link 整理后的网格 · 左：零姿态重装配  右：刚体分组展开", font=font, fill="white")
    canvas.paste(frames[0], (0, 50))
    canvas.paste(frames[1], (900, 50))
    for i, link in enumerate(manifest["links"]):
        x, y = 20+(i % 4)*440, 706+(i//4)*32
        color = tuple(round(c*255) for c in link["rgba"][:3])
        draw.rectangle((x, y+4, x+18, y+22), fill=color)
        draw.text((x+26, y), f"{link['name']}  ({link['component_count']} 件)", font=small, fill="white")
    draw.text((20, 770), "7 个 link / 31 个显示实例；根部仅含机械臂安装板，电机外形未补齐；仅为 visual 网格。", font=small, fill="#bac5d2")
    canvas.save(MODEL / "link_mesh_overview.png")


def main():
    manifest = json.loads((MODEL / "link_mesh_manifest.json").read_text(encoding="utf-8"))
    joints = load("urdf_work/config/joints.json")
    model = inspection_model(manifest, joints)
    report = validate(manifest, joints, model)
    write_json(MODEL / "link_mesh_validation.json", report)
    if not report["all_checks_passed"]:
        raise ValueError("Exported mesh verification failed")
    render(model, manifest, joints)
    # Remove validation-only arrays added to the in-memory component records.
    for link in manifest["links"]:
        for part in link["components"]:
            part.pop("_native_home_m", None)
            part.pop("_source_keep", None)
    manifest["stage"] = "link_visual_meshes_readback_verified"
    manifest["validation_report"] = "link_mesh_validation.json"
    write_json(MODEL / "link_mesh_manifest.json", manifest)
    print(json.dumps({k: report[k] for k in ["links", "visual_components", "triangles", "configurations_checked",
                                            "max_stl_reassembly_vertex_error_m", "max_obj_reassembly_vertex_error_m", "all_checks_passed"]}))


if __name__ == "__main__":
    main()
