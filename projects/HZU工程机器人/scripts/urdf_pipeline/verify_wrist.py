"""Measure supplied gear teeth and cross-check the saved differential layout.

This does not calibrate motor encoders or solve SolidWorks' active mate set.
"""
import json

import numpy as np

from inventory import ROOT, sha256_file, write_json
from joint_evidence import Evidence, line_relation, load, transform_axis
from mesh_sections import measure_sections, read_binary_stl


def main():
    e = Evidence()
    printed = load("urdf_work/evidence/print_geometry.json")["results"]
    end = "末端零件/末端装配.SLDASM"
    joints = load("urdf_work/config/joints.json")
    basis = np.array(joints["end_assembly_in_arm_matrix_mm"])
    # IDs refer to the standalone end assembly. The motor/pulley connections
    # have supporting raw mate-name/reference windows, not decoded constraints.
    parts = [("87", "output_bevel"), ("40", "side_bevel"),
             ("25", "side_pulley"), ("22", "motor_pulley")]
    measured = {}
    for instance_id, role in parts:
        instance = e.instance(end, instance_id)
        matches = [(key, g) for key, g in printed.items()
                   if any(p.rsplit("/", 1)[-1].rsplit(".", 1)[0] == instance["name"]
                          for p in g["source_paths"])]
        if len(matches) != 1:
            raise ValueError(f"Nonunique print mesh for {instance_id}")
        key, g = matches[0]
        for path in g["source_paths"]:
            if sha256_file(ROOT / path) != key:
                raise ValueError(f"Print source changed: {path}")
        if sha256_file(ROOT / g["mesh_path"]) != g["mesh_sha256"]:
            raise ValueError("Generated mesh changed")
        native_bounds = np.array(list(map(float, instance["model_attributes"]["swBoundingBox"].split()))) * 1000
        bounds = np.array([*g["bbox_min_mm"], *g["bbox_max_mm"]])
        measured[role] = {"instance_id": instance_id, "source_paths": g["source_paths"],
                          "source_sha256": key, "mesh_sha256": g["mesh_sha256"],
                          "max_native_mesh_bbox_error_mm": float(abs(bounds-native_bounds).max()),
                          "full_native_mesh_equivalence_verified": False,
                          "measurements": measure_sections(read_binary_stl(ROOT / g["mesh_path"]) * 1000)}
    write_json(ROOT / "urdf_work/evidence/gear_mesh_measurements.json", measured)

    def axis(instance_id):
        instance = e.instance(end, instance_id)
        return transform_axis(basis @ np.array(instance["matrix_mm"]), [0, 0, 0], [1, 0, 0],
                              {"instance_id": instance_id, "name": instance["name"],
                               "method": "printed gear section about local X plus native saved transform"})

    pitch, spin = [j["saved_pose_axis_in_arm_frame"] for j in joints["joints"][4:]]
    coaxial = []
    for instance_id in ["25", "40", "51", "52"]:
        relation = line_relation(pitch, axis(instance_id))
        coaxial.append({"instance_id": instance_id, **relation,
                        "passed": relation["line_distance_mm"] < .05 and relation["parallel_error"] < 1e-8})
    relation = line_relation(spin, axis("87"))
    coaxial.append({"instance_id": "87", **relation,
                    "passed": relation["line_distance_mm"] < .05 and relation["parallel_error"] < 1e-8})
    belt_layout = []
    for motor, driver, driven, bevel in [("43", "22", "52", "51"), ("46", "47", "25", "40")]:
        a, b = axis(driver), axis(driven)
        separation = line_relation(a, b)
        delta = np.array(b["point_mm"]) - a["point_mm"]
        belt_layout.append({"motor_instance_id": motor, "motor_reference_number": e.instance(end, motor)["reference_number"],
                            "driver_pulley_instance_id": driver, "driven_pulley_instance_id": driven,
                            "side_bevel_instance_id": bevel,
                            "parallel_error": separation["parallel_error"],
                            "axis_separation_mm": separation["line_distance_mm"],
                            "pulley_origin_axial_offset_mm": abs(float(delta @ a["axis"])),
                            "belt_routing_verified": False,
                            "note": "Parallel axes on same side support pairing; belt centerplanes and routing are not solved"})
    teeth = {role: r["measurements"]["measured_periodic_order"] for role, r in measured.items()}
    if any(v is None for v in teeth.values()):
        raise ValueError("Gear tooth count inconclusive")
    k = teeth["side_bevel"] / teeth["output_bevel"]
    # Define virtual inputs at the side pinions, both measured about the same
    # housing-fixed J5 axis. Choose the difference sign by definition; its
    # correspondence to the measured J6 positive direction remains unverified.
    ideal = np.array([[.5, .5], [k/2, -k/2]])
    report = {"schema_version": 1, "teeth_measured_from_supplied_print_meshes": teeth,
              "coaxial_checks": coaxial, "belt_layout": belt_layout,
              "external_belt_ratio_magnitude": teeth["motor_pulley"] / teeth["side_pulley"],
              "ideal_differential": {"input": "two side pinion angles about common housing-fixed J5 axis",
                                     "output": "carrier angle, spin angle with algebraically selected sign",
                                     "matrix": ideal.tolist(), "rank": int(np.linalg.matrix_rank(ideal)),
                                     "spin_sign_to_measured_J6_axis": None,
                                     "assumptions": ["both 18-tooth pinions mesh with the supplied 51-tooth gear",
                                                     "ideal rigid no-slip gearing; no backlash or elastic slip"],
                                     "hardware_motor_to_joint_matrix": None,
                                     "motor_internal_reduction_and_encoder_signs_included": False},
              "all_coaxial_checks_passed": all(c["passed"] for c in coaxial),
              "conclusion": "Geometry and full-rank ideal differential support two independent wrist outputs; hardware calibration remains open",
              "limitations": ["3MF/native full shape identity not proved; maximum bbox differences are reported per part",
                              "Raw mate windows support topology but active CAD constraints are not decoded",
                              "Belt routing, motor signs, internal reductions, end stops and encoder zeros are not measured"]}
    write_json(ROOT / "urdf_work/evidence/wrist_verification.json", report)
    if not report["all_coaxial_checks_passed"]:
        raise ValueError("Wrist axis verification failed")
    print(json.dumps({"teeth": teeth, "wrist_axis_checks_passed": len(coaxial),
                      "ideal_differential_rank": report["ideal_differential"]["rank"],
                      "belt_axis_separations_mm": [b["axis_separation_mm"] for b in belt_layout]}))


if __name__ == "__main__":
    main()
