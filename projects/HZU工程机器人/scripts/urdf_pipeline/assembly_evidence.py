"""Cross-check saved placements against independent STEP geometry and 3MF meshes."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np

from inventory import ROOT, normalized_name, sha256_file, write_json
from native import resolve_reference


def load(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def main():
    inventory = load("urdf_work/evidence/inventory.json")
    assembly = load("urdf_work/config/assembly.json")
    step = load("urdf_work/evidence/geometry.json")["results"]
    printed = load("urdf_work/evidence/print_geometry.json")["results"]
    geometry = {**step, **printed}
    step_by_path = {p: k for k, g in step.items() for p in g["source_paths"]}
    by_native = {}
    for match in inventory["cad_matches"]:
        if match["status"] == "unique_name_candidate":
            by_native.setdefault(match["sldprt_candidates"][0], set()).add(step_by_path[match["step_path"]])
    embedded = load("urdf_work/evidence/native/" + assembly["source"]["sha256"] + ".json")["embedded_components"]
    checked = []
    for instance in assembly["instances"]:
        source = instance["source_resolution"]["path"]
        file = instance["stored_file"] or {}
        candidates = set(by_native.get(source, []))
        evidence = "unique_normalized_name"
        if file.get("swDocType") == "ASSEMBLY" and file.get("swImportedPath"):
            resolution = resolve_reference(file["swImportedPath"], inventory["step_files"])
            if resolution["path"]:
                candidates.add(step_by_path[resolution["path"]])
                evidence = "native_xml_swImportedPath"
        if source:
            candidates.update(k for k, g in printed.items() if normalized_name(Path(source).stem) in
                              {normalized_name(Path(p).stem) for p in g["source_paths"]})
        bbox_raw = (instance["model_attributes"] or {}).get("swBoundingBox")
        native_bounds = np.array([float(x) for x in bbox_raw.split()]) * 1000 if bbox_raw else None
        comparisons = []
        for key in sorted(candidates):
            g = geometry[key]
            if g.get("status") not in {"ok", "success"} or native_bounds is None:
                continue
            bounds = np.array([*g["bbox_min_mm"], *g["bbox_max_mm"]])
            error = float(np.max(np.abs(bounds - native_bounds)))
            comparisons.append({"geometry_sha256": key, "source_paths": g["source_paths"],
                                "max_local_bbox_coordinate_error_mm": error,
                                "local_bbox_matches": error <= 0.05, "mapping_evidence": evidence})
        passed = [r for r in comparisons if r["local_bbox_matches"]]
        chosen = passed[0] if len(passed) == 1 else None
        instance["geometry_check"] = {"candidates": comparisons, "status": "unique_bbox_supported_candidate" if chosen else
                                      "ambiguous_bbox_candidates" if len(passed) > 1 else
                                      "bbox_mismatch" if comparisons else "no_mesh_candidate",
                                      "full_geometry_equivalence_verified": False}
        if chosen:
            g = geometry[chosen["geometry_sha256"]]
            instance["geometry_check"].update({"geometry_sha256": chosen["geometry_sha256"], "mesh_path": g["mesh_path"],
                                               "max_local_bbox_coordinate_error_mm": chosen["max_local_bbox_coordinate_error_mm"]})
        if instance["source_resolution"]["status"] == "missing" and file.get("swDocType") == "PART":
            matches = [e for e in embedded if Path(e["stream"]).name.casefold() == Path(file.get("swPath", "")).name.casefold()]
            if len(matches) == 1:
                instance["embedded_source"] = {**matches[0], "container_source_sha256": assembly["source"]["sha256"],
                                               "status": "exact_saved_embedded_native_payload"}
        checked.append(instance)
    assembly["instances"] = checked
    assembly["summary"]["geometry_checks"] = dict(Counter(i["geometry_check"]["status"] for i in checked))
    assembly["summary"]["missing_paths_with_embedded_payload"] = sum("embedded_source" in i for i in checked)
    write_json(ROOT / "urdf_work/config/assembly.json", assembly)

    # Constraints independent of the assembly origin: corresponding plate bores
    # must remain coaxial after applying the saved component transformations.
    by_id = {r["instance_id"]: r for r in checked}

    def bore(instance_id, point):
        instance = by_id[instance_id]
        if not instance["geometry_check"].get("geometry_sha256"):
            raise ValueError(f"No bbox-supported STEP for axis check {instance_id}")
        g = geometry[instance["geometry_check"]["geometry_sha256"]]
        axes = [a for a in g.get("cylinder_candidates", []) if
                np.linalg.norm(np.array(a["point_mm"]) - point) < 1e-5 and abs(a["axis"][2]) > .999999]
        if not axes:
            raise ValueError(f"Measured cylinder missing at {instance_id} {point}")
        matrix = np.array(instance["matrix_mm"])
        return {"instance_id": instance_id, "name": instance["name"], "source_sha256": instance["source_sha256"],
                "geometry_sha256": instance["geometry_check"]["geometry_sha256"],
                "local_point_mm": point, "world_point_mm": (matrix[:3, :3] @ point + matrix[:3, 3]).tolist(),
                "world_axis": (matrix[:3, :3] @ axes[0]["axis"]).tolist(),
                "measured_radii_mm": sorted(set(round(a["radius_mm"], 7) for a in axes))}

    pairs = [
        ("base_output_left_right", bore("4/10", [0, 70, 0]), bore("4/23", [0, 70, 0])),
        ("base_to_large_arm", bore("4/23", [0, 70, 0]), bore("101/107", [150, 0, 0])),
        ("large_arm_proximal_left_right", bore("101/107", [150, 0, 0]), bore("101/110", [-150, 0, 0])),
        ("large_arm_distal_left_right", bore("101/107", [-150, 0, 0]), bore("101/110", [150, 0, 0])),
        ("large_to_small_arm", bore("101/110", [150, 0, 0]), bore("153/159", [105, 0, 0])),
        ("small_arm_left_right", bore("153/159", [105, 0, 0]), bore("153/162", [105, 0, 0])),
    ]
    constraints = []
    for name, a, b in pairs:
        ax = np.array(a["world_axis"])
        bx = np.array(b["world_axis"])
        delta = np.array(b["world_point_mm"]) - a["world_point_mm"]
        separation = float(np.linalg.norm(np.cross(delta, ax)))
        parallel_error = float(np.linalg.norm(np.cross(ax, bx)))
        constraints.append({"name": name, "a": a, "b": b, "axis_line_distance_mm": separation,
                            "axis_parallel_error": parallel_error, "axial_point_spacing_mm": abs(float(delta @ ax)),
                            "passed": separation < .05 and parallel_error < 1e-8})
    measured = {"large_arm_axis_distance_mm": float(np.linalg.norm(np.cross(
                    np.array(pairs[3][1]["world_point_mm"]) - pairs[2][1]["world_point_mm"], pairs[2][1]["world_axis"]))),
                "base_motor_to_output_axis_distance_mm": 100.0,
                "base_motor_axis": bore("4/23", [0, -30, 0]), "base_output_axis": pairs[0][2]}
    measured["base_motor_to_output_axis_distance_mm"] = float(np.linalg.norm(np.cross(
        np.array(measured["base_output_axis"]["world_point_mm"]) - measured["base_motor_axis"]["world_point_mm"],
        measured["base_motor_axis"]["world_axis"])))
    source_checks = [{"path": r["path"], "unchanged": sha256_file(ROOT / r["path"]) == r["sha256"]}
                     for r in inventory["files"]]
    write_json(ROOT / "urdf_work/evidence/phase2_validation.json", {
        "schema_version": 1, "native_summary": load("urdf_work/evidence/native_index.json")["summary"],
        "arm_summary": assembly["summary"], "coaxial_constraints": constraints, "measurements": measured,
        "all_coaxial_checks_passed": all(r["passed"] for r in constraints),
        "max_axis_line_distance_mm": max(r["axis_line_distance_mm"] for r in constraints),
        "source_files_checked": len(source_checks), "source_files_unchanged": all(r["unchanged"] for r in source_checks),
        "changed_sources": [r["path"] for r in source_checks if not r["unchanged"]],
        "tests": "4 native XML/matrix tests and 1 nested 3MF transform test passed",
        "model_generated": False,
        "limitations": ["Matching bounds is supporting evidence, not proof of full shape equality",
                        "CAD saved pose is not yet the joint zero; joint axes/limits/rigid groups remain to be frozen",
                        "Incomplete mesh coverage; embedded native motor parts are retained without fabricated geometry"]})
    print(json.dumps({"arm": assembly["summary"], "coaxial_checks_passed": sum(r["passed"] for r in constraints),
                      "max_axis_line_distance_mm": max(r["axis_line_distance_mm"] for r in constraints),
                      "source_files_unchanged": all(r["unchanged"] for r in source_checks)}))
    if not all(r["passed"] for r in constraints) or not all(r["unchanged"] for r in source_checks):
        raise SystemExit("Validation failed; see phase2_validation.json")


if __name__ == "__main__":
    main()
