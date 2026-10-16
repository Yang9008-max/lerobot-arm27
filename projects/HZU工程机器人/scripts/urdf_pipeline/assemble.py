"""Export reviewed visual components into explicit link-local meter meshes.

Components are concatenated without booleans/decimation; exact zero-area faces
are removed with source indices recorded. This is a visual mesh
package, not a URDF description, collision model, or mass-property estimate.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from inventory import ROOT, sha256_file, write_json
from joint_evidence import Evidence, load
from link_mesh_io import clean_degenerate_triangles, decode_stl, encode_obj, encode_stl, link_frames, require_rigid, transform_triangles


MODEL = ROOT / "models/hzu_arm"
ARM = "机械臂1/机械臂装配.SLDASM"
END = "末端零件/末端装配.SLDASM"
FULL = "HZU工程完整装配.SLDASM"
COLORS = {"chassis_mount": [.55, .60, .68, 1], "yaw_base": [.35, .66, .95, 1],
          "upper_arm": [.95, .65, .25, 1], "forearm": [.4, .83, .6, 1],
          "wrist_housing": [.6, .5, .92, 1], "wrist_carrier": [.95, .45, .5, 1],
          "suction_tool": [.3, .83, .88, 1]}


def native_to_m(matrix_mm):
    matrix = np.array(matrix_mm, dtype=float)
    matrix[:3, 3] /= 1000  # Only CAD translations need mm -> m here.
    return require_rigid(matrix)


def selected_components(e, joints, frames):
    arm = load("urdf_work/config/assembly.json")
    by_id = {r["instance_id"]: r for r in arm["instances"]}
    reviewed = load("urdf_work/evidence/joint_motion_verification.json")
    if not reviewed["all_kinematic_checks_passed"]:
        raise ValueError("Reviewed output chain has not passed kinematic checks")
    catalog = {**e.geometry, **load("urdf_work/evidence/print_geometry.json")["results"]}
    by_mesh = {g["mesh_path"]: (key, g) for key, g in catalog.items() if g.get("mesh_path")}
    selected = []
    for visual in reviewed["visual_instances"]:
        identifier = visual["instance_id"]
        path = ARM if "/" in identifier else END
        instance = by_id[identifier] if path == ARM else e.instance(path, identifier)
        key, geometry = by_mesh[visual["mesh_path"]]
        world = native_to_m(instance["matrix_mm"])
        if path == END:
            world = native_to_m(joints["end_assembly_in_arm_matrix_mm"]) @ world
        expected = np.eye(4)
        expected[:3, :3] = visual["home_rotation"]
        expected[:3, 3] = visual["home_position_m"]
        if np.max(abs(expected-world)) > 1e-9:
            raise ValueError(f"Reviewed component pose changed: {path} {identifier}")
        selected.append(component_record(e, path, instance, visual["link"], key, geometry, world, frames))

    # Add the fixed mounting plate already used to establish the J1 axis.
    # This is one chassis attachment, not the full chassis or its rotating belt.
    instance = e.instance(FULL, "4/245")
    key, geometry = next((k, g) for k, g in e.geometry.items()
                         if "加工/碳板/底盘/机械臂-固定板X1.STEP" in g["source_paths"])
    in_arm = np.linalg.inv(native_to_m(e.instance(FULL, "392")["matrix_mm"])) @ native_to_m(instance["matrix_mm"])
    selected.append(component_record(e, FULL, instance, "chassis_mount", key, geometry, in_arm, frames))
    identities = {(r["assembly_path"], r["instance_id"]) for r in selected}
    if len(identities) != len(selected):
        raise ValueError("Component assigned to multiple links")
    return selected


def component_record(e, assembly_path, instance, link, key, geometry, in_arm, frames):
    if instance["suppressed_in_tree"] or instance["hidden_in_tree"]:
        raise ValueError("Reviewed display component is hidden/suppressed")
    if geometry.get("mesh_unit") != "m":
        raise ValueError("Source mesh meter unit not established")
    native_bounds = np.array(list(map(float, instance["model_attributes"]["swBoundingBox"].split()))) * 1000
    bounds = np.array([*geometry["bbox_min_mm"], *geometry["bbox_max_mm"]])
    error = float(abs(native_bounds-bounds).max())
    printed = geometry["source_paths"][0].lower().endswith(".3mf")
    tolerance = .20 if printed else .05
    if error > tolerance:
        raise ValueError(f"Component local bounds mismatch: {instance['name']} {error} mm")
    if link not in frames:
        raise ValueError("Unknown link assignment")
    assumptions = []
    if "8009-RAU" in instance["name"]:
        assumptions.append("Composite motor exterior attached as housing visual approximation; rotor not split")
    if printed:
        assumptions.append("Supplied 3MF shape candidate; full equivalence with native CAD not proved")
    return {"assembly_path": assembly_path, "assembly_sha256": e.records[assembly_path]["sha256"],
            "instance_id": instance["instance_id"], "name": instance["name"], "link": link,
            "native_path": instance["source_resolution"].get("path"), "native_sha256": instance.get("source_sha256"),
            "source_geometry_paths": geometry["source_paths"], "source_geometry_sha256": key,
            "source_mesh_path": geometry["mesh_path"],
            "source_mesh_sha256": geometry.get("mesh_sha256") or geometry["mesh"]["sha256"],
            "geometry_correspondence": {"type": "3MF_candidate" if printed else "STEP_bbox_supported",
                                        "max_bbox_error_mm": error, "bbox_tolerance_mm": tolerance,
                                        "full_shape_equivalence_verified": False},
            "part_in_arm_m": in_arm.tolist(),
            "part_in_link_m": (np.linalg.inv(frames[link]) @ in_arm).tolist(),
            "assignment_basis": "previously reviewed major-structure output chain" if link != "chassis_mount" else
                                "fixed chassis mounting plate; independently measured J1 bore",
            "assumptions": assumptions}


def verify_inputs(e, selected):
    checked = {}
    for record in selected:
        paths = [(record["assembly_path"], record["assembly_sha256"]),
                 (record["source_mesh_path"], record["source_mesh_sha256"])]
        paths += [(p, record["source_geometry_sha256"]) for p in record["source_geometry_paths"]]
        if record["native_path"]:
            paths.append((record["native_path"], record["native_sha256"]))
        for path, digest in paths:
            if path in checked:
                if checked[path] != digest:
                    raise ValueError("Conflicting source hashes")
                continue
            if sha256_file(ROOT / path) != digest:
                raise ValueError(f"Source or generated mesh changed: {path}")
            checked[path] = digest
    return checked


def coverage(e, selected):
    items = []
    for path in [ARM, END]:
        instances = e.assembly(path)["instances"]
        parents = {r["parent_instance_id"] for r in instances}
        ids = {r["instance_id"]: r for r in selected if r["assembly_path"] == path}
        for instance in instances:
            identifier = instance["instance_id"]
            item = {"assembly_path": path, "instance_id": identifier, "name": instance["name"],
                    "native_path": instance["source_resolution"].get("path"), "leaf": identifier not in parents}
            if identifier in ids:
                item.update(status="included_visual", link=ids[identifier]["link"])
            elif any(identifier.startswith(parent + "/") for parent in ids):
                item.update(status="represented_by_parent_assembly_visual_proxy",
                            reason="Parent motor exterior STEP is included; internal rotor/race geometry is not separately assigned")
            elif identifier in parents:
                item.update(status="assembly_container", reason="Children are accounted for individually")
            elif instance["suppressed_in_tree"] or instance["hidden_in_tree"]:
                item.update(status="omitted", reason="hidden_or_suppressed_in_saved_tree")
            elif instance["name"] == "金矿石" or any(s in instance["name"] for s in ["视场", "遮挡", "UWB"]):
                item.update(status="omitted", reason="grasped_object_or_sensor_annotation_outside_arm_visual_scope")
            elif path == END and identifier in ["22", "25", "40", "47", "51", "52"]:
                item.update(status="omitted", reason="moving_differential_drivetrain_not_rigidly_attached_to_output_links")
            elif path == END and identifier in ["4", "43", "46"]:
                item.update(status="omitted", reason="native_motor_pose_known_but_no_validated_exterior_mesh")
            else:
                item.update(status="omitted", reason="mesh_correspondence_or_rigid_membership_not_yet_verified")
            items.append(item)
    return {"scope": "arm and end assembly occurrences; full chassis scope includes only mounting plate 4/245",
            "full_chassis_and_wheels_included": False,
            "items": items, "occurrences": len(items),
            "included_selected_components": len(selected),
            "omitted_leaf_instances": sum(r["leaf"] and r["status"] == "omitted" for r in items),
            "not_complete_robot_geometry": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "urdf_work/config/assembly.json")
    args = parser.parse_args()
    if args.config.resolve() != (ROOT / "urdf_work/config/assembly.json").resolve():
        raise ValueError("This reviewed export supports only the configured arm assembly")
    e = Evidence()
    joints = load("urdf_work/config/joints.json")
    frames = link_frames(joints)
    for joint in joints["joints"]:
        measured = joint["saved_pose_axis_in_arm_frame"]
        frame = frames[joint["child"]]
        if np.linalg.norm(frame[:3, 3]-np.array(measured["point_mm"])/1000) > 1e-9 or np.linalg.norm(
                frame[:3, :3] @ joint["axis"]-measured["axis"]) > 1e-9:
            raise ValueError("Local link frame no longer matches measured output axis")
    selected = selected_components(e, joints, frames)
    hashes = verify_inputs(e, selected)
    out = MODEL / "meshes/visual"
    out.mkdir(parents=True, exist_ok=True)
    cache, links = {}, []
    for link, frame in frames.items():
        components, transformed, offset = [], [], 0
        for record in [r for r in selected if r["link"] == link]:
            path = record["source_mesh_path"]
            if path not in cache:
                cache[path] = decode_stl((ROOT / path).read_bytes())[0]
            triangles = transform_triangles(cache[path], record["part_in_link_m"])
            triangles, removed = clean_degenerate_triangles(triangles, cache[path])
            transformed.append(triangles)
            components.append({**record, "triangle_start": offset, "triangle_count": len(triangles),
                               "source_triangle_count": len(cache[path]), "removed_source_triangle_indices": removed})
            offset += len(triangles)
        if not transformed:
            raise ValueError(f"No source visual components for {link}")
        triangles = np.concatenate(transformed)
        stl, obj = out / f"{link}.stl", out / f"{link}.obj"
        stl.write_bytes(encode_stl(triangles))
        obj.write_bytes(encode_obj(triangles))
        links.append({"name": link, "link_in_arm_m": frame.tolist(), "rgba": COLORS[link],
                      "mesh_unit": "m", "urdf_visual_origin_xyz": [0, 0, 0], "urdf_visual_origin_rpy": [0, 0, 0],
                      "urdf_mesh_scale": [1, 1, 1],
                      "stl_path": stl.relative_to(MODEL).as_posix(), "stl_sha256": sha256_file(stl),
                      "obj_path": obj.relative_to(MODEL).as_posix(), "obj_sha256": sha256_file(obj),
                      "triangles": len(triangles), "component_count": len(components), "components": components,
                      "bbox_min_m": triangles.min(axis=(0, 1)).tolist(), "bbox_max_m": triangles.max(axis=(0, 1)).tolist()})
    manifest = {"schema_version": 1, "stage": "link_visual_meshes_exported_pending_readback_validation",
                "purpose": "link-local visual mesh preparation for subsequent URDF export",
                "ready_for_urdf_release": False, "robot_description_generated": False,
                "frame_convention": "column-vector transforms; root aligned to arm CAD A; input vertices already meters",
                "operation": "rigid transform and triangle concatenation; remove only exact degenerate faces; no rescale, booleans, target decimation or inferred physics",
                "input_artifacts": {p: sha256_file(ROOT / p) for p in ["urdf_work/config/joints.json",
                                    "urdf_work/config/assembly.json", "urdf_work/evidence/joint_motion_verification.json"]},
                "verified_input_files": hashes, "links": links,
                "summary": {"links": len(links), "visual_components": len(selected),
                            "triangles": sum(l["triangles"] for l in links),
                            "source_triangles": sum(c["source_triangle_count"] for l in links for c in l["components"]),
                            "removed_degenerate_triangles": sum(len(indices) for l in links for c in l["components"]
                                                                for indices in c["removed_source_triangle_indices"].values()),
                            "full_chassis_included": False,
                            "motor_exteriors_included": ["8009_base", "8009_elbow"],
                            "motor_exteriors_missing": ["4310", "GM6020", "M2006_ref_2", "M2006_ref_4"]},
                "limitations": ["Visual geometry only; not collision meshes or physical mass groups",
                                "Two composite 8009 models approximate housings; internal rotor motion omitted",
                                "Incomplete fastener, motor, bearing and drivetrain geometry",
                                "3MF/native shape equivalence unproved; per-component bbox tolerances retained",
                                "Only fixed mounting plate represents chassis; wheels and full chassis not included"]}
    write_json(MODEL / "link_mesh_manifest.json", manifest)
    write_json(MODEL / "geometry_coverage.json", coverage(e, selected))
    write_json(MODEL / "joint_order.json", load("urdf_work/config/joint_order.json"))
    print(json.dumps(manifest["summary"]))


if __name__ == "__main__":
    main()
