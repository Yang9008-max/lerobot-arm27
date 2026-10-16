"""Build an output-side inspection chain and check it against world screw FK.

Gravity/contact are disabled. Placeholder inertias are numerical requirements,
not physical estimates. This checks frame conversion, not actual joint limits.
"""
import json
import struct
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from inventory import ROOT, normalized_name, write_json
from joint_evidence import Evidence, load


def numbers(values):
    return " ".join(format(float(x), ".15g") for x in values)


def rotation_about_line(point, axis, angle):
    """World-frame screw exponential, independent of nested body translations."""
    u = np.asarray(axis, dtype=float)
    u /= np.linalg.norm(u)
    skew = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    rotation = np.eye(3) + np.sin(angle)*skew + (1-np.cos(angle))*(skew @ skew)
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = (np.eye(3)-rotation) @ point
    return result


def make_model(joints):
    e = Evidence()
    assembly = load("urdf_work/config/assembly.json")
    root = ET.Element("mujoco", model="six_output_joint_kinematic_inspection_only")
    ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="false")
    ET.SubElement(root, "option", gravity="0 0 0")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="720", offheight="480")
    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="-.3 -1 1.5", dir="0 1 -1", diffuse=".9 .9 .9")
    bodies = {"chassis_mount": ET.SubElement(world, "body", name="chassis_mount")}
    origins = {"chassis_mount": np.zeros(3)}
    for joint in joints["joints"]:
        p = np.array(joint["saved_pose_axis_in_arm_frame"]["point_mm"]) / 1000
        body = ET.SubElement(bodies[joint["parent"]], "body", name=joint["child"],
                             pos=numbers(p-origins[joint["parent"]]))
        ET.SubElement(body, "inertial", pos="0 0 0", mass="1", diaginertia=".01 .01 .01")
        ET.SubElement(body, "joint", name=joint["name"], type="hinge", limited="false",
                      axis=numbers(joint["saved_pose_axis_in_arm_frame"]["axis"]))
        bodies[joint["child"]] = body
        origins[joint["child"]] = p
    assets, meshes, instances = {}, {}, []
    colors = {"yaw_base": ".35 .66 .95 1", "upper_arm": ".95 .65 .25 1",
              "forearm": ".4 .83 .6 1", "wrist_housing": ".6 .5 .92 1",
              "wrist_carrier": ".95 .45 .5 1", "suction_tool": ".3 .83 .88 1"}

    def add_mesh(instance, link, matrix, geometry, kind):
        path = geometry["mesh_path"]
        if path not in meshes:
            name = "mesh_" + str(len(meshes))
            data = (ROOT / path).read_bytes()
            filename = name + ".stl"
            if struct.unpack_from("<I", data, 80)[0] > 200000:
                dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
                triangles = np.frombuffer(data, dtype=dtype, offset=84)["vertices"]
                vertices, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
                lines = ["v " + numbers(v) for v in vertices]
                lines += ["f " + " ".join(str(int(x)+1) for x in face) for face in inverse.reshape(-1, 3)]
                data = ("\n".join(lines) + "\n").encode("ascii")
                filename = name + ".obj"
            assets[filename] = data
            meshes[path] = name
            ET.SubElement(asset, "mesh", name=name, file=filename)
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, matrix[:3, :3].reshape(9))
        name = "part_" + str(len(instances))
        ET.SubElement(bodies[link], "geom", name=name, type="mesh", mesh=meshes[path],
                      pos=numbers(matrix[:3, 3]/1000-origins[link]), quat=numbers(quaternion),
                      rgba=colors[link], contype="0", conaffinity="0")
        instances.append({"geom": name, "instance_id": instance["instance_id"], "name": instance["name"],
                          "link": link, "mesh_path": path, "geometry_mapping": kind,
                          "home_position_m": (matrix[:3, 3]/1000).tolist(),
                          "home_rotation": matrix[:3, :3].tolist()})

    for instance in assembly["instances"]:
        check = instance["geometry_check"]
        if not check.get("mesh_path") or instance["hidden_in_tree"] or instance["suppressed_in_tree"]:
            continue
        identifier = instance["instance_id"]
        if identifier.startswith("4/"):
            link = "yaw_base"
        elif identifier.startswith("101/") or identifier == "212/47":
            link = "upper_arm"
        elif identifier.startswith("153/"):
            link = "forearm"
        else:
            continue  # Composite bearing races and UWB not assigned by guess.
        add_mesh(instance, link, np.array(instance["matrix_mm"]), check, "STEP_bbox_supported")

    end = "末端零件/末端装配.SLDASM"
    basis = np.array(joints["end_assembly_in_arm_matrix_mm"])
    printed = load("urdf_work/evidence/print_geometry.json")["results"]
    groups = {"wrist_housing": ["7", "10", "13", "16", "61", "69", "72"],
              "wrist_carrier": ["78", "91", "94"], "suction_tool": ["81", "84", "87"]}
    for link, identifiers in groups.items():
        for identifier in identifiers:
            instance = e.instance(end, identifier)
            candidates = [(key, g) for key, g in {**e.geometry, **printed}.items()
                          if normalized_name(instance["name"]) in
                          {normalized_name(p.rsplit("/", 1)[-1].rsplit(".", 1)[0]) for p in g["source_paths"]}]
            bounds = np.array(list(map(float, instance["model_attributes"]["swBoundingBox"].split()))) * 1000
            candidates = [(key, g, float(abs(bounds-np.array([*g["bbox_min_mm"], *g["bbox_max_mm"]])).max()))
                          for key, g in candidates]
            if len(candidates) != 1:
                raise ValueError(f"Nonunique mesh for end instance {identifier}")
            key, g, error = candidates[0]
            # The coarser printed rotor meshes are shown as candidates, never
            # silently promoted to STEP's tighter .05mm correspondence check.
            tolerance = .20 if key in printed else .05
            if error > tolerance:
                raise ValueError(f"Inspection mesh exceeds declared bbox tolerance: {identifier}: {error}")
            add_mesh(instance, link, basis @ np.array(instance["matrix_mm"]), g,
                     {"source": "3MF_candidate" if key in printed else "STEP_bbox_supported",
                      "max_bbox_error_mm": error, "shape_equivalence_verified": False})

    # Asymmetric colored tool reference makes axial spin visible even if the
    # supplied suction rotor is nearly rotationally symmetric.
    ET.SubElement(bodies["suction_tool"], "site", name="tool_reference", pos="-.02 .015 .01", size=".003", rgba="1 1 0 1")
    ET.SubElement(bodies["suction_tool"], "geom", type="capsule", size=".002", fromto="0 0 0 0 .025 .012",
                  rgba="1 .95 .15 1", contype="0", conaffinity="0")
    ET.indent(root)
    out = ROOT / "urdf_work/inspection/joint_motion"
    out.mkdir(parents=True, exist_ok=True)
    (out / "six_joint_inspection.xml").write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    for name, data in assets.items():
        (out / name).write_bytes(data)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), assets=assets)
    return model, origins, instances, out


def verify(model, origins, instances, joints):
    data = mujoco.MjData(model)
    axes = [j["saved_pose_axis_in_arm_frame"] for j in joints["joints"]]
    configurations = [np.zeros(6)]
    for i in range(6):
        for sign in [-1, 1]:
            q = np.zeros(6)
            q[i] = sign*np.deg2rad(10)
            configurations.append(q)
    configurations.extend(np.random.default_rng(20261015).uniform(-np.deg2rad(20), np.deg2rad(20), (32, 6)))
    errors = []
    for q in configurations:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        transforms = {"chassis_mount": np.eye(4)}
        transform = np.eye(4)
        for angle, axis, joint in zip(q, axes, joints["joints"]):
            transform = transform @ rotation_about_line(np.array(axis["point_mm"])/1000, axis["axis"], angle)
            transforms[joint["child"]] = transform.copy()
        position_errors, rotation_errors = [], []
        for name, origin in origins.items():
            expected = transforms[name]
            position = expected[:3, :3] @ origin + expected[:3, 3]
            body = data.body(name)
            position_errors.append(float(np.linalg.norm(body.xpos-position)))
            rotation_errors.append(float(abs(body.xmat.reshape(3, 3)-expected[:3, :3]).max()))
        # MuJoCo preprocesses mesh coordinate frames; geom_xpos is the compiled
        # mesh's frame, so compare body frames/site rather than raw CAD origins.
        tool = transforms["suction_tool"]
        site_home = origins["suction_tool"] + [-.02, .015, .01]
        expected_site = tool[:3, :3] @ site_home + tool[:3, 3]
        site_error = float(np.linalg.norm(data.site("tool_reference").xpos-expected_site))
        errors.append({"angles_rad": q.tolist(), "max_link_position_error_m": max(position_errors),
                       "max_rotation_entry_error": max(rotation_errors), "tool_site_error_m": site_error})
    # Parent-side structures stay fixed when a single downstream joint moves.
    invariance = []
    data.qpos[:] = 0
    mujoco.mj_forward(model, data)
    home = {name: (data.body(name).xpos.copy(), data.body(name).xmat.copy()) for name in origins}
    for i, joint in enumerate(joints["joints"]):
        data.qpos[:] = 0
        data.qpos[i] = np.deg2rad(10)
        mujoco.mj_forward(model, data)
        upstream = ["chassis_mount"] + [j["child"] for j in joints["joints"][:i]]
        fixed = all(np.max(abs(data.body(n).xpos-home[n][0])) < 1e-10 and
                    np.max(abs(data.body(n).xmat-home[n][1])) < 1e-10 for n in upstream)
        moving = [j["child"] for j in joints["joints"][i:]]
        oriented = all(np.max(abs(data.body(n).xmat-home[n][1])) > 1e-4 for n in moving)
        invariance.append({"joint": joint["name"], "unchanged_upstream_links": upstream,
                           "rotating_descendant_links": moving, "passed": bool(fixed and oriented)})
    max_position = max(r["max_link_position_error_m"] for r in errors)
    max_rotation = max(r["max_rotation_entry_error"] for r in errors)
    report = {"schema_version": 1, "purpose": "verify output-chain local frames and descendant motion against independent world-axis FK",
              "simulation_joints": model.njnt, "mesh_instances": len(instances), "configurations_checked": len(errors),
              "max_link_position_error_m": max_position, "max_rotation_entry_error": max_rotation,
              "single_joint_hierarchy_checks": invariance,
              "all_kinematic_checks_passed": bool(max_position < 1e-9 and max_rotation < 1e-9 and
                                                   max(r["tool_site_error_m"] for r in errors) < 1e-9 and
                                                   all(r["passed"] for r in invariance)),
              "configurations": errors, "visual_instances": instances,
              "reference_pose": "CAD saved pose chosen as inspection q=0 only; not hardware encoder zero",
              "not_validated": ["CAD active mate solving", "physical travel limits and collisions", "actual inertias and torque dynamics",
                                "hardware motor encoder mapping", "complete mesh coverage", "all fastener/rotor/race rigid memberships"],
              "numeric_inertias_are_placeholders": True, "gravity_and_contacts_disabled": True}
    return report


def render(model, joints, out):
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=720, height=480)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)
    small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 16)
    frames = []
    try:
        for phase in range(16):
            angle = np.deg2rad(15)*np.sin(2*np.pi*phase/16)
            canvas = Image.new("RGB", (2160, 1070), "#101821")
            draw = ImageDraw.Draw(canvas)
            draw.text((20, 14), "六轴运动核查 · 保存姿态为模型参考姿态 · 黄色为吸盘自转参考标记", font=font, fill="white")
            for i, joint in enumerate(joints["joints"]):
                data.qpos[:] = 0
                data.qpos[i] = angle
                mujoco.mj_forward(model, data)
                camera.lookat[:] = [-.27, 0, .17] if i < 3 else [-.59, .001, .175]
                camera.distance = 1.12 if i < 3 else .45
                camera.azimuth, camera.elevation = 110, -25
                renderer.update_scene(data, camera=camera)
                frame = Image.fromarray(renderer.render())
                x, y = (i % 3)*720, 50+(i//3)*510
                canvas.paste(frame, (x, y+28))
                draw.text((x+12, y+3), f"J{i+1} {joint['label_zh']}  {np.rad2deg(angle):+.1f}°", font=font, fill="#f0d27a")
            draw.text((20, 1042), "仅验证运动坐标与随动关系；演示角度不是机械限位，质量/接触/电机耦合尚未用于动力学。", font=small, fill="#c0c9d4")
            frames.append(canvas)
    finally:
        renderer.close()
    frames[4].save(out / "six_joint_motion_overview.png")
    frames[0].save(out / "six_joint_motion.gif", save_all=True, append_images=frames[1:], duration=150, loop=0)


def main():
    joints = load("urdf_work/config/joints.json")
    model, origins, instances, out = make_model(joints)
    report = verify(model, origins, instances, joints)
    write_json(ROOT / "urdf_work/evidence/joint_motion_verification.json", report)
    if not report["all_kinematic_checks_passed"]:
        raise ValueError("Kinematic verification failed; inspect report")
    render(model, joints, out)
    # Freeze a transparent model-frame convention after numerical verification.
    # Every body frame is aligned with A at q=0; its origin is its joint point.
    # Consequently the measured A-axis is also the local axis at this reference
    # pose. This is an explicit construction, not a generic global-to-local rule.
    wrist = load("urdf_work/evidence/wrist_verification.json")
    for joint in joints["joints"]:
        joint.update({"type": "revolute", "origin_xyz_m": (origins[joint["child"]]-origins[joint["parent"]]).tolist(),
                      "origin_rpy_rad": [0, 0, 0], "axis": joint["saved_pose_axis_in_arm_frame"]["axis"],
                      "origin_status": "model frame verified against independent FK",
                      "zero_definition": "saved CAD reference chosen as model q=0; hardware zero unknown",
                      "motion_status": "CAD_supported_output_chain_with_verified_model_frames",
                      "verification_report": "urdf_work/evidence/joint_motion_verification.json"})
        if joint["number"] in [5, 6]:
            joint["transmission"].update({"type": "two_motor_bevel_differential_CAD_supported",
                                          "measured_print_mesh_teeth": wrist["teeth_measured_from_supplied_print_meshes"],
                                          "ideal_side_pinion_to_output_mapping": wrist["ideal_differential"],
                                          "motor_to_joint_matrix_status": "hardware_signs_and_reductions_unverified",
                                          "verification_report": "urdf_work/evidence/wrist_verification.json"})
            joint["actuators"] = ["M2006_end_instance_43_ref_2", "M2006_end_instance_46_ref_4"]
    joints["stage"] = "CAD_supported_six_output_chain_model_frames_verified"
    joints["model_frame_convention"] = "All link frames align with A at saved reference pose; child origin is its measured joint point"
    joints["verification_summary"] = {"axis_checks_passed": joints["summary"]["axis_checks_passed"] + len(wrist["coaxial_checks"]),
                                      "kinematic_configurations_checked": report["configurations_checked"],
                                      "hierarchy_checks_passed": 6, "hardware_calibrated": False,
                                      "physical_limits_and_collisions_verified": False}
    write_json(ROOT / "urdf_work/config/joints.json", joints)
    write_json(ROOT / "urdf_work/config/joint_order.json", {
        "schema_version": 1, "status": "output_joint_order_selected_after_CAD_review",
        "joint_names": [j["name"] for j in joints["joints"]], "unit": "rad",
        "reference_zero": "CAD saved reference; not hardware encoder zero",
        "suction": "separate on/off channel; not one of the six rotational joints",
        "hardware_channel_mapping": None})
    print(json.dumps({k: report[k] for k in ["simulation_joints", "mesh_instances", "configurations_checked",
                                            "max_link_position_error_m", "max_rotation_entry_error", "all_kinematic_checks_passed"]}))


if __name__ == "__main__":
    main()
