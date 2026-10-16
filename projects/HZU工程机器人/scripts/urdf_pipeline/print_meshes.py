"""Read standard 3MF build/component transforms and export meters-based STL."""
from __future__ import annotations

import io
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from inventory import ROOT, sha256_file, write_json

NS = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
UNITS_M = {"micron": 1e-6, "millimeter": 0.001, "centimeter": 0.01,
           "inch": 0.0254, "foot": 0.3048, "meter": 1.0}


def transform(raw):
    if raw is None:
        return np.eye(4)
    values = np.asarray([float(v) for v in raw.split()])
    if values.size != 12 or not np.isfinite(values).all():
        raise ValueError("Invalid 3MF component transform")
    result = np.eye(4)
    result[:3, :] = values.reshape(4, 3).T
    return result


def read_3mf(data: bytes) -> tuple[np.ndarray, dict]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        # Resolve the standard package's root model relationship explicitly.
        rels = ET.fromstring(archive.read("_rels/.rels"))
        models = [r.get("Target") for r in rels if r.get("Type", "").endswith("/3dmodel")]
        if len(models) != 1:
            raise ValueError("Expected one root 3MF model relationship")
        target = models[0].lstrip("/")
        model = ET.fromstring(archive.read(target))
        scale = UNITS_M[model.get("unit", "millimeter")]
        objects = {o.get("id"): o for o in model.findall("m:resources/m:object", NS)}
        triangles = []

        def emit(object_id, matrix, chain):
            if object_id in chain:
                raise ValueError("Cycle in 3MF components")
            obj = objects[object_id]
            mesh = obj.find("m:mesh", NS)
            if mesh is not None:
                vertices = np.array([[float(v.get(k)) for k in ("x", "y", "z")]
                                     for v in mesh.findall("m:vertices/m:vertex", NS)])
                indices = np.array([[int(t.get(k)) for k in ("v1", "v2", "v3")]
                                    for t in mesh.findall("m:triangles/m:triangle", NS)], dtype=int)
                if not np.isfinite(vertices).all() or np.any(indices < 0) or np.any(indices >= len(vertices)):
                    raise ValueError("Invalid 3MF vertex or triangle index")
                world = vertices @ matrix[:3, :3].T + matrix[:3, 3]
                faces = world[indices] * scale
                if np.linalg.det(matrix[:3, :3]) < 0:
                    faces = faces[:, [0, 2, 1], :]
                triangles.append(faces)
            for child in obj.findall("m:components/m:component", NS):
                if any(k.rsplit("}", 1)[-1] == "path" for k in child.attrib):
                    raise ValueError("External 3MF component resource is unsupported")
                emit(child.get("objectid"), matrix @ transform(child.get("transform")), [*chain, object_id])

        for item in model.findall("m:build/m:item", NS):
            if item.get("printable", "true") != "false":
                emit(item.get("objectid"), transform(item.get("transform")), [])
        if not triangles:
            raise ValueError("No build triangles")
        faces = np.concatenate(triangles)
        return faces, {"source_unit": model.get("unit", "millimeter"), "mesh_unit": "m",
                       "triangles": len(faces), "build_transforms_applied": True,
                       "bbox_min_mm": (faces.min(axis=(0, 1)) * 1000).tolist(),
                       "bbox_max_mm": (faces.max(axis=(0, 1)) * 1000).tolist()}


def write_stl(path, triangles):
    edges1 = triangles[:, 1] - triangles[:, 0]
    edges2 = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(edges1, edges2)
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.where(lengths == 0, 1, lengths)[:, None]
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    records = np.zeros(len(triangles), dtype=dtype)
    records["normal"] = normals
    records["vertices"] = triangles
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"HZU 3MF geometry in meters".ljust(80, b"\0") + struct.pack("<I", len(triangles)) + records.tobytes())


def main():
    inventory = json.loads((ROOT / "urdf_work/evidence/inventory.json").read_text(encoding="utf-8"))
    results = {}
    for record in inventory["files"]:
        if record["extension"] != ".3mf":
            continue
        try:
            source = ROOT / record["path"]
            if sha256_file(source) != record["sha256"]:
                raise ValueError("Source hash mismatch")
            triangles, result = read_3mf(source.read_bytes())
            output = ROOT / "urdf_work/assets/print_meshes" / (record["sha256"] + ".stl")
            write_stl(output, triangles)
            result.update({"source_paths": [record["path"]], "status": "success",
                           "mesh_path": output.relative_to(ROOT).as_posix(), "mesh_sha256": sha256_file(output),
                           "mass_properties": "withheld: tessellated mesh; solidity not validated"})
        except Exception as exc:
            result = {"source_paths": [record["path"]], "status": "error", "error": str(exc)}
        results[record["sha256"]] = result
    write_json(ROOT / "urdf_work/evidence/print_geometry.json", {"schema_version": 1, "results": results})
    print(json.dumps({"3mf_files": len(results), "successful": sum(r["status"] == "success" for r in results.values()),
                      "triangles": sum(r.get("triangles", 0) for r in results.values())}))


if __name__ == "__main__":
    main()
