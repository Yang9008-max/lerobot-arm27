"""Analyze STEP solids and export source-part meshes in meters, not a robot model."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import struct
import time

import cadquery as cq
import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Circle, GeomAbs_Cylinder

from inventory import ROOT, sha256_file, write_json

ANALYSIS_VERSION = 1


def vector(value) -> list[float]:
    return [float(value.X()), float(value.Y()), float(value.Z())]


def canonical_axis(value) -> list[float]:
    values = np.asarray(vector(value), dtype=float)
    values /= np.linalg.norm(values)
    for entry in values:
        if abs(entry) > 1e-10:
            if entry < 0:
                values *= -1
            break
    return [0.0 if abs(x) < 1e-12 else float(x) for x in values]


def analyze_shape(shape: cq.Shape) -> dict:
    solids = shape.Solids()
    if not solids:
        raise ValueError("No solid geometry; a surface-only model requires separate processing")
    volume = shape.Volume()
    if not math.isfinite(volume) or volume <= 0:
        raise ValueError("Non-positive or non-finite solid volume")
    bounds = shape.BoundingBox()
    center = shape.Center()
    cylinders = {}
    circles = {}
    for index, face in enumerate(shape.Faces()):
        surface = BRepAdaptor_Surface(face.wrapped)
        if surface.GetType() != GeomAbs_Cylinder:
            continue
        cylinder = surface.Cylinder()
        direction = canonical_axis(cylinder.Axis().Direction())
        location = np.asarray(vector(cylinder.Location()))
        # Closest point on the infinite axis to the part origin unifies split faces.
        point = location - np.dot(location, direction) * np.asarray(direction)
        key = tuple(round(x, 7) for x in [*point, *direction, cylinder.Radius()])
        if key not in cylinders:
            cylinders[key] = {"point_mm": point.tolist(), "axis": direction,
                              "radius_mm": float(cylinder.Radius()), "face_indices": []}
        cylinders[key]["face_indices"].append(index)
    for edge in shape.Edges():
        curve = BRepAdaptor_Curve(edge.wrapped)
        if curve.GetType() != GeomAbs_Circle:
            continue
        circle = curve.Circle()
        center_point = vector(circle.Location())
        direction = canonical_axis(circle.Axis().Direction())
        key = tuple(round(x, 7) for x in [*center_point, *direction, circle.Radius()])
        if key not in circles:
            circles[key] = {"center_mm": center_point, "axis": direction,
                            "radius_mm": float(circle.Radius()), "edge_count": 0}
        circles[key]["edge_count"] += 1
    inertia = np.asarray(cq.Shape.matrixOfInertia(shape), dtype=float)
    # Unit-density geometric inertia scales with length^5, volume with length^3.
    return {
        "solid_count": len(solids), "face_count": len(shape.Faces()),
        "input_analysis_unit": "mm", "mesh_unit": "m", "mesh_scale": 0.001,
        "bbox_min_mm": [bounds.xmin, bounds.ymin, bounds.zmin],
        "bbox_max_mm": [bounds.xmax, bounds.ymax, bounds.zmax],
        "bbox_size_mm": [bounds.xlen, bounds.ylen, bounds.zlen],
        "volume_mm3": volume, "volume_m3": volume * 1e-9,
        "geometric_centroid_mm": [center.x, center.y, center.z],
        "unit_density_inertia_at_centroid_m5": (inertia * 1e-15).tolist(),
        "mass_density_assumption": "uniform unit density for geometry only; actual material/mass unverified",
        "cylinder_candidates": list(cylinders.values()), "circle_candidates": list(circles.values()),
        "candidates_are_confirmed_robot_joints": False,
    }


def read_stl_bounds(path: Path) -> dict:
    with path.open("rb") as stream:
        header = stream.read(84)
        if len(header) != 84:
            raise ValueError("Truncated STL header")
        count = struct.unpack_from("<I", header, 80)[0]
        if path.stat().st_size != 84 + count * 50 or count == 0:
            raise ValueError("Invalid binary STL size or empty mesh")
        lower, upper = np.full(3, np.inf), np.full(3, -np.inf)
        for _ in range(count):
            row = stream.read(50)
            vertices = np.asarray(struct.unpack_from("<9f", row, 12)).reshape(3, 3)
            lower = np.minimum(lower, vertices.min(axis=0))
            upper = np.maximum(upper, vertices.max(axis=0))
        if not np.isfinite([lower, upper]).all():
            raise ValueError("Non-finite STL coordinates")
        return {"triangles": count, "bbox_min_m": lower.tolist(), "bbox_max_m": upper.tolist(),
                "bbox_size_m": (upper - lower).tolist()}


def process_step(path: Path, mesh_path: Path) -> dict:
    shape = cq.importers.importStep(str(path)).val()
    result = analyze_shape(shape)
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(shape.scale(0.001), str(mesh_path), exportType="STL",
                        tolerance=0.0001, angularTolerance=0.15)
    mesh = read_stl_bounds(mesh_path)
    # Sampling a curved surface can shrink its bounding box slightly.
    expected_min = np.asarray(result["bbox_min_mm"]) * 0.001
    expected_max = np.asarray(result["bbox_max_mm"]) * 0.001
    delta = max(np.max(np.abs(expected_min - mesh["bbox_min_m"])),
                np.max(np.abs(expected_max - mesh["bbox_max_m"])))
    if delta > 0.001:
        raise ValueError(f"STL bounds differ from CAD by {delta:.6g} m")
    return {**result, "mesh": {**mesh, "cad_bbox_error_m": float(delta), "sha256": sha256_file(mesh_path)}}


def export_csv(path: Path, results: dict) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sha256", "status", "source_path", "solid_count", "x_mm", "y_mm", "z_mm",
                         "volume_mm3", "cylinder_candidates", "circle_candidates", "mesh_path", "error"])
        for key, row in results.items():
            writer.writerow([key, row["status"], row["source_paths"][0], row.get("solid_count", ""),
                             *row.get("bbox_size_mm", ["", "", ""]), row.get("volume_mm3", ""),
                             len(row.get("cylinder_candidates", [])), len(row.get("circle_candidates", [])),
                             row.get("mesh_path", ""), row.get("error", "")])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--scope", choices=["all", "arm"], default="all")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    evidence = root / "urdf_work/evidence"
    inventory = json.loads((evidence / "inventory.json").read_text(encoding="utf-8"))
    geometry_path = evidence / "geometry.json"
    catalog = json.loads(geometry_path.read_text(encoding="utf-8")) if geometry_path.exists() else {
        "schema_version": ANALYSIS_VERSION, "stage": "source_part_geometry_only", "results": {}}
    if catalog["schema_version"] != ANALYSIS_VERSION:
        raise ValueError("Analysis version differs; use a new output directory")
    by_hash = {}
    for record in inventory["step_files"]:
        by_hash.setdefault(record["sha256"], []).append(record)
    selected = [(key, rows) for key, rows in by_hash.items()
                if args.scope == "all" or any(row["category"].startswith("arm") for row in rows)]
    for index, (key, rows) in enumerate(selected, start=1):
        paths = [row["path"] for row in rows]
        previous = catalog["results"].get(key)
        if previous and previous["status"] == "ok":
            existing = root / previous["mesh_path"]
            if existing.exists() and sha256_file(existing) == previous["mesh"]["sha256"]:
                previous["source_paths"] = paths
                continue
        if previous and previous["status"] == "error" and not args.retry_errors:
            continue
        source = root / paths[0]
        if sha256_file(source) != key:
            raise ValueError("Source changed since inventory; rebuild inventory first")
        mesh_relative = f"urdf_work/assets/part_meshes/{key}.stl"
        begin = time.perf_counter()
        row = {"source_paths": paths, "category": rows[0]["category"]}
        try:
            row.update(process_step(source, root / mesh_relative))
            row.update(status="ok", mesh_path=mesh_relative)
        except Exception as error:
            row.update(status="error", error=f"{type(error).__name__}: {error}")
        row["elapsed_s"] = round(time.perf_counter() - begin, 3)
        catalog["results"][key] = row
        catalog["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(geometry_path, catalog)
        print(json.dumps({"progress": f"{index}/{len(selected)}", "source": paths[0],
                          "status": row["status"], "elapsed_s": row["elapsed_s"]}, ensure_ascii=True), flush=True)
    export_csv(evidence / "geometry.csv", catalog["results"])
    selected_results = [catalog["results"][key] for key, _ in selected]
    summary = {"scope": args.scope, "selected_unique_contents": len(selected),
               "ok": sum(row["status"] == "ok" for row in selected_results),
               "errors": sum(row["status"] == "error" for row in selected_results),
               "source_step_paths": sum(len(rows) for _, rows in selected),
               "model_generated": False}
    write_json(evidence / "geometry_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
