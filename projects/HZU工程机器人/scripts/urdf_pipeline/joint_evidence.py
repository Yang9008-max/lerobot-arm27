"""Recover measured joint-axis candidates and raw mate-name evidence, not a mate solver."""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import numpy as np

from inventory import ROOT, sha256_file, write_json


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def line_relation(a, b):
    p, q = np.array(a["point_mm"]), np.array(b["point_mm"])
    u, v = np.array(a["axis"]), np.array(b["axis"])
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    cross = np.cross(u, v)
    parallel_error = float(np.linalg.norm(cross))
    distance = float(abs(np.dot(q-p, cross)) / parallel_error) if parallel_error > 1e-8 else float(np.linalg.norm(np.cross(q-p, u)))
    return {"line_distance_mm": distance, "parallel_error": parallel_error,
            "absolute_axis_dot": float(abs(np.dot(u, v)))}


def transform_axis(matrix, point, direction, evidence):
    matrix = np.asarray(matrix, dtype=float)
    axis = matrix[:3, :3] @ direction
    axis /= np.linalg.norm(axis)
    return {"point_mm": (matrix[:3, :3] @ point + matrix[:3, 3]).tolist(),
            "axis": axis.tolist(), "evidence": evidence}


class Evidence:
    def __init__(self):
        self.inventory = load("urdf_work/evidence/inventory.json")
        self.index = load("urdf_work/evidence/native_index.json")
        self.geometry = load("urdf_work/evidence/geometry.json")["results"]
        self.records = {r["path"]: r for r in self.inventory["files"]}
        self.documents = {}

    def assembly(self, path):
        if path not in self.documents:
            record = self.records[path]
            self.documents[path] = load("urdf_work/evidence/native/" + record["sha256"] + ".assembly.json")
        return self.documents[path]

    def instance(self, path, instance_id):
        return next(r for r in self.assembly(path)["instances"] if r["instance_id"] == instance_id)

    def axis(self, path, instance_id, point, direction, basis, geometry_path=None, radius=None):
        instance = self.instance(path, instance_id)
        if geometry_path:
            key = next(k for k, g in self.geometry.items() if geometry_path in g["source_paths"])
            geometry = self.geometry[key]
            candidates = [c for c in geometry["cylinder_candidates"] if
                          np.linalg.norm(np.array(c["point_mm"]) - point) < 1e-5 and
                          abs(np.dot(c["axis"], direction)) > 1-1e-8 and
                          (radius is None or abs(c["radius_mm"] - radius) < 1e-5)]
            if not candidates:
                raise ValueError(f"Measured cylinder not found: {geometry_path} {point} {radius}")
            stored = np.array([float(v) for v in instance["model_attributes"]["swBoundingBox"].split()]) * 1000
            measured = np.array([*geometry["bbox_min_mm"], *geometry["bbox_max_mm"]])
            error = float(np.max(abs(stored-measured)))
            if error > .05:
                raise ValueError(f"Native/STEP coordinate bounds mismatch: {geometry_path}: {error} mm")
        evidence = {"assembly_path": path, "assembly_sha256": self.records[path]["sha256"],
                    "instance_id": instance_id, "name": instance["name"],
                    "native_path": instance["source_resolution"]["path"],
                    "native_sha256": instance["source_sha256"], "local_point_mm": point, "local_axis": direction,
                    "geometry_path": geometry_path, "radius_mm": radius,
                    "method": "STEP_cylinder_and_saved_instance_transform" if geometry_path else "saved_part_axis_hypothesis"}
        return transform_axis(basis @ np.array(instance["matrix_mm"]), point, direction, evidence)


def extract_mate_tokens(evidence):
    sys.path.insert(0, str(ROOT / "urdf_work/tools/swformat/src"))
    from swformat.io.reader import read_document
    from swformat.carchive.cstring import read_cstring
    documents = []
    pattern = re.compile(r"^(同心|重合|距离|角度|平行|垂直|对称|齿轮|齿条|螺旋|锁定|宽度|槽口|皮带|凸轮|相切|万向节|限制距离|限制角度)\d+$")
    for path in ["HZU工程完整装配.SLDASM", "机械臂1/机械臂装配.SLDASM", "机械臂1/机械臂底座装配.SLDASM",
                 "机械臂1/大臂装配.SLDASM", "机械臂1/小臂装配.SLDASM", "末端零件/末端装配.SLDASM"]:
        source = ROOT / path
        if sha256_file(source) != evidence.records[path]["sha256"]:
            raise ValueError(f"Source changed: {path}")
        streams = read_document(source).streams()
        mates = []
        for stream, data in streams.items():
            if "MatesList" not in stream:
                continue
            tokens, start = [], 0
            while True:
                offset = data.find(b"\xff\xfe\xff", start)
                if offset < 0:
                    break
                start = offset + 3
                decoded = read_cstring(data, offset)
                if decoded and decoded[0]:
                    tokens.append({"decoded_offset": offset, "value": decoded[0]})
            names = [(i, t) for i, t in enumerate(tokens) if pattern.fullmatch(t["value"])]
            for index, (position, token) in enumerate(names):
                end = names[index+1][0] if index+1 < len(names) else len(tokens)
                refs = []
                for t in tokens[position+1:end]:
                    v = t["value"]
                    if "@" in v and "\n" not in v and "\r" not in v and len(v) < 500 and v not in refs:
                        refs.append(v)
                mates.append({"stream": stream, "decoded_offset": token["decoded_offset"],
                              "mate_name_token": token["value"], "nearby_component_reference_tokens": refs[:12]})
        documents.append({"path": path, "source_sha256": evidence.records[path]["sha256"], "mate_name_groups": mates})
    result = {"schema_version": 1, "method": "bounded CString scan; groups are heuristic evidence windows",
              "not_a_mate_solver": True, "constraint_semantics_and_active_status_verified": False, "documents": documents}
    write_json(ROOT / "urdf_work/evidence/mate_tokens.json", result)
    return result


def compare_saved_contexts(evidence):
    """Compare occurrence poses across independent saved assembly documents."""
    full = evidence.assembly("HZU工程完整装配.SLDASM")
    table = {r["instance_id"]: r for r in full["instances"]}
    results = []
    for path, prefix in [("机械臂1/机械臂装配.SLDASM", "392"), ("末端零件/末端装配.SLDASM", "734")]:
        own = evidence.assembly(path)
        own_table = {r["instance_id"]: r for r in own["instances"]}

        def signature(instance, lookup, stop=None):
            parts = []
            while instance and instance["instance_id"] != stop:
                parts.append((instance["name"], instance["reference_number"]))
                instance = lookup.get(instance["parent_instance_id"])
            return tuple(reversed(parts))

        by_signature = {signature(r, own_table): r for r in own["instances"]}
        inverse = np.linalg.inv(table[prefix]["matrix_mm"])
        comparisons = []
        for instance in full["instances"]:
            if not instance["instance_id"].startswith(prefix + "/"):
                continue
            match = by_signature.get(signature(instance, table, prefix))
            if match is None:
                comparisons.append({"name": instance["name"], "status": "unmatched"})
                continue
            a = inverse @ np.array(instance["matrix_mm"])
            b = np.array(match["matrix_mm"])
            translation = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
            rotation = float(np.max(np.abs(a[:3, :3] - b[:3, :3])))
            comparisons.append({"name": instance["name"], "own_instance_id": match["instance_id"],
                                "full_instance_id": instance["instance_id"], "translation_difference_mm": translation,
                                "rotation_max_entry_difference": rotation,
                                "status": "consistent" if translation < .05 and rotation < 1e-8 else "different_saved_pose"})
        results.append({"standalone_path": path, "full_assembly_prefix": prefix, "instances": comparisons,
                        "consistent": sum(r["status"] == "consistent" for r in comparisons),
                        "differences": [r for r in comparisons if r["status"] != "consistent"]})
    write_json(ROOT / "urdf_work/evidence/joint_saved_contexts.json", {"schema_version": 1, "comparisons": results})
    return results


def main():
    e = Evidence()
    full = "HZU工程完整装配.SLDASM"
    arm = "机械臂1/机械臂装配.SLDASM"
    end = "末端零件/末端装配.SLDASM"
    full_arm = np.array(e.instance(full, "392")["matrix_mm"])
    full_end = np.array(e.instance(full, "734")["matrix_mm"])
    end_in_arm = np.linalg.inv(full_arm) @ full_end
    world_in_arm = np.linalg.inv(full_arm)
    eye = np.eye(4)
    sources = []
    sources.append(e.axis(full, "4/245", [0, 0, 0], [0, 0, 1], world_in_arm,
                          "加工/碳板/底盘/机械臂-固定板X1.STEP", 26.5))
    sources.append(e.axis(full, "4/263", [0, 0, 0], [0, 0, 1], world_in_arm,
                          "加工/碳板/底盘/交叉滚子轴承上垫板X1.STEP", 15))
    sources.append(e.axis(arm, "4/7", [0, 0, 0], [0, 0, 1], eye,
                          "加工/碳板/机械臂底座/底板X1.STEP", 10))
    yaw = {"point_mm": [0, 0, 0], "axis": [0, 0, 1]}
    shoulder = {"point_mm": [0, 0, 175], "axis": [0, 1, 0]}
    elbow = {"point_mm": [-300, 0, 175], "axis": [0, 1, 0]}
    shoulder_sources = [e.axis(arm, "4/23", [0, 70, 0], [0, 0, 1], eye,
                              "加工/碳板/机械臂底座/8009侧板X1.STEP", 5),
                        e.axis(arm, "101/107", [150, 0, 0], [0, 0, 1], eye,
                              "加工/碳板/大臂/侧板大（左）X1.STEP", 5)]
    elbow_sources = [e.axis(arm, "101/110", [150, 0, 0], [0, 0, 1], eye,
                           "加工/碳板/大臂/侧板大（右）X1.STEP", 9),
                     e.axis(arm, "153/159", [105, 0, 0], [0, 0, 1], eye,
                           "加工/碳板/小臂/侧板小（右）X1.STEP", 6)]
    roll_sources = [e.axis(arm, "153/156", [0, 0, 0], [0, 0, 1], eye,
                          "加工/碳板/小臂/固定板X1.STEP", 12.5),
                    e.axis(end, "7", [0, 0, 0], [0, 0, 1], end_in_arm,
                           "加工/碳板/末端/joint4输出垫高板x1.STEP", 12.5),
                    e.axis(end, "10", [43.825, 0, 35], [0, 1, 0], end_in_arm,
                           "加工/碳板/末端/link3连接板(改)x1.STEP", 12.5)]
    # Origin is an explicit frame choice at the measured output plate's z=0
    # plane. A joint's line is invariant to shifting its origin along the axis.
    roll = {"point_mm": roll_sources[1]["point_mm"], "axis": [1, 0, 0]}
    pitch_sources = [e.axis(end, "69", [0, 38.25, 0], [0, 0, 1], end_in_arm,
                           "加工/碳板/末端/2006固定板x2.STEP", 10.5),
                     e.axis(end, "72", [0, 38.25, 0], [0, 0, 1], end_in_arm,
                            "加工/碳板/末端/2006固定板x2.STEP", 10.5),
                     e.axis(end, "91", [0, 0, 0], [0, 0, 1], end_in_arm,
                            "加工/碳板/末端/吸盘连接侧板X2.STEP", 10.5),
                     e.axis(end, "94", [0, 0, 0], [0, 0, 1], end_in_arm,
                            "加工/碳板/末端/吸盘连接侧板X2.STEP", 10.5)]
    pitch_center = np.mean([np.array(pitch_sources[i]["point_mm"]) for i in [0, 1]], axis=0)
    pitch = {"point_mm": pitch_center.tolist(), "axis": pitch_sources[0]["axis"]}
    spin_sources = [e.axis(end, "78", [0, 0, 0], [0, 0, 1], end_in_arm,
                          "加工/碳板/末端/吸盘固定板X1.STEP", 21),
                    e.axis(end, "66", [0, 0, 0], [0, 0, 1], end_in_arm)]
    spin = {"point_mm": spin_sources[0]["point_mm"], "axis": spin_sources[0]["axis"]}
    groups = [("yaw_coaxial", yaw, sources), ("shoulder_coaxial", shoulder, shoulder_sources),
              ("elbow_coaxial", elbow, elbow_sources), ("forearm_roll_coaxial", roll, roll_sources),
              ("wrist_pitch_coaxial", pitch, pitch_sources), ("tool_spin_coaxial", spin, spin_sources)]
    checks = []
    for name, axis, data in groups:
        for source in data:
            relation = line_relation(axis, source)
            checks.append({"name": name, "source": source, **relation,
                           "passed": relation["line_distance_mm"] < .05 and relation["parallel_error"] < 1e-8})
    relation = line_relation(pitch, spin)
    checks.append({"name": "wrist_axes_intersection_and_perpendicularity", **relation,
                   "passed": relation["line_distance_mm"] < .05 and relation["absolute_axis_dot"] < 1e-8})
    definitions = [
        ("joint_1_base_yaw", "底座回转", "chassis_mount", "yaw_base", yaw, sources, ["4310"],
         {"type": "timing_belt", "driver_teeth_candidate": 25, "driven_teeth_candidate": 56,
          "output_per_motor_output_candidate": 25/56, "ratio_evidence": "CAD component names", "ratio_confirmed": False}),
        ("joint_2_shoulder_pitch", "大臂俯仰", "yaw_base", "upper_arm", shoulder, shoulder_sources, ["8009_base"],
         {"type": "timing_belt", "driver_teeth_candidate": 28, "driven_teeth_candidate": 56,
          "output_per_motor_output_candidate": .5, "ratio_evidence": "CAD component names and two separated axes", "ratio_confirmed": False}),
        ("joint_3_elbow_pitch", "小臂俯仰", "upper_arm", "forearm", elbow, elbow_sources, ["8009_elbow"],
         {"type": "coaxial_output_candidate", "ratio_confirmed": False}),
        ("joint_4_forearm_roll", "前臂轴向旋转", "forearm", "wrist_housing", roll, roll_sources, ["GM6020"],
         {"type": "coaxial_output_candidate", "ratio_confirmed": False}),
        ("joint_5_wrist_pitch", "腕部俯仰", "wrist_housing", "wrist_carrier", pitch, pitch_sources, ["M2006_left", "M2006_right"],
         {"type": "two_motor_differential_candidate", "coupled_with": "joint_6_tool_roll", "motor_to_joint_matrix": None,
          "motor_to_joint_matrix_status": "unconfirmed", "ratio_confirmed": False}),
        ("joint_6_tool_roll", "吸盘自转", "wrist_carrier", "suction_tool", spin, spin_sources, ["M2006_left", "M2006_right"],
         {"type": "two_motor_differential_candidate", "coupled_with": "joint_5_wrist_pitch", "motor_to_joint_matrix": None,
          "motor_to_joint_matrix_status": "unconfirmed", "ratio_confirmed": False}),
    ]
    joints = []
    for n, (name, label, parent, child, axis, geometry_sources, motors, transmission) in enumerate(definitions, 1):
        joints.append({"number": n, "name": name, "label_zh": label, "parent": parent, "child": child,
                       "type_candidate": "revolute", "type": None,
                       "saved_pose_axis_in_arm_frame": axis, "geometry_evidence": geometry_sources,
                       "origin_xyz_m": None, "origin_rpy_rad": None, "axis": None,
                       "origin_status": "measured_axis_line_available; final local joint frame not frozen",
                       "zero_definition": None, "limits": None, "actuators": motors, "transmission": transmission,
                       "geometry_status": "coaxial_geometry_supported", "motion_status": "candidate_pending_review",
                       "positive_direction_status": "not_motor_calibrated", "user_confirmed": False})
    result = {"schema_version": 1, "stage": "six_axis_motion_candidate", "ready_for_urdf_release": False,
              "reference_frame": "saved mechanical arm assembly frame A: mm for measurements; directions dimensionless",
              "numbering": "new proposed base-to-tool numbering; source CAD joint-number names are retained as evidence only",
              "saved_pose_is_control_zero": False, "end_assembly_in_arm_matrix_mm": end_in_arm.tolist(),
              "joints": joints, "checks": checks,
              "excluded": [{"name": "金矿石", "reason": "external grasped object, not a robot link"},
                           {"name": "真空泵/吸盘吸附", "reason": "separate on/off action, not an extra rotational joint"},
                           {"name": "图传视场/UWB遮挡范围", "reason": "annotation/sensor range, not physical robot geometry"}],
              "summary": {"candidate_rotational_dof": 6, "axis_checks": len(checks),
                          "axis_checks_passed": sum(c["passed"] for c in checks),
                          "limits_confirmed": 0, "control_zero_confirmed": 0,
                          "all_geometric_checks_passed": all(c["passed"] for c in checks)}}
    write_json(ROOT / "urdf_work/config/joints.json", result)
    mate_tokens = extract_mate_tokens(e)
    contexts = compare_saved_contexts(e)
    result["summary"]["mate_name_groups"] = sum(len(d["mate_name_groups"]) for d in mate_tokens["documents"])
    result["summary"]["saved_pose_contexts"] = [{"path": r["standalone_path"], "consistent": r["consistent"],
                                                "different": len(r["differences"])} for r in contexts]
    write_json(ROOT / "urdf_work/config/joints.json", result)
    print(json.dumps(result["summary"]))
    print(json.dumps([{"number": j["number"], "label": j["label_zh"], **j["saved_pose_axis_in_arm_frame"]} for j in joints], ensure_ascii=False))
    if not result["summary"]["all_geometric_checks_passed"]:
        print(json.dumps([c for c in checks if not c["passed"]], ensure_ascii=False, indent=2))
        raise SystemExit("Geometric checks failed; inspect evidence before freezing joints")


if __name__ == "__main__":
    main()
