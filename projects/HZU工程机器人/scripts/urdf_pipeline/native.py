"""Recover last-saved SolidWorks XML assembly evidence; never edit source CAD."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path, PureWindowsPath
import sys
import xml.etree.ElementTree as ET

import numpy as np

from inventory import ROOT, sha256_file, write_json

SWFORMAT_COMMIT = "69b80c860241fdcbfa334f67cb951f76398c1a71"
XML_STREAMS = ("swXmlContents/COMPINSTANCETREE", "swXmlContents/Features")


def tag(element):
    return element.tag.rsplit("}", 1)[-1]


def parse_transform(raw: str) -> dict:
    # COMPINSTANCETREE stores an ordinary row-vector homogeneous matrix,
    # NOT the different 16-element MathTransform.ArrayData API layout.
    values = np.array([float(v) for v in raw.split()])
    if values.size != 16 or not np.isfinite(values).all():
        raise ValueError("Transform requires 16 finite values")
    matrix = values.reshape(4, 4).T
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-10, rtol=0):
        raise ValueError("Transform is not affine in the observed XML convention")
    rotation = matrix[:3, :3]
    error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    determinant = float(np.linalg.det(rotation))
    valid = error < 1e-8 and abs(determinant - 1) < 1e-8
    return {"raw": raw, "matrix_m": matrix.tolist(),
            "rotation_orthogonality_error": error, "rotation_determinant": determinant,
            "rigid_valid": valid}


def parse_xml(payload: bytes) -> dict:
    root = ET.fromstring(payload)  # XML declaration handles UTF-8 and entities.
    files = {e.get("id"): dict(e.attrib) for e in root.iter() if tag(e) == "swFile"}
    models = {}
    for element in root.iter():
        if tag(element) != "swModel":
            continue
        model = {"attributes": dict(element.attrib), "file": files.get(element.get("swFileRef")),
                 "references": []}
        for child in element:
            if tag(child) != "swReference":
                continue
            reference = {"attributes": dict(child.attrib), "transform": None}
            if child.get("swTransform"):
                try:
                    reference["transform"] = parse_transform(child.get("swTransform"))
                except ValueError as exc:
                    reference["transform_error"] = str(exc)
            model["references"].append(reference)
        models[element.get("id")] = model
    configs = [dict(e.attrib) for e in root.iter() if tag(e) == "swConfiguration"]
    recent = [c for c in configs if c.get("swMostRecentConfiguration") == "YES"]
    selected = recent[0] if len(recent) == 1 else configs[0] if len(configs) == 1 else None
    return {"xml_attributes": dict(root.attrib), "files": files, "models": models,
            "configurations": configs, "root_model_id": selected.get("swModelRef") if selected else None,
            "configuration_selection": "unique_most_recent" if len(recent) == 1 else
            "only_configuration" if len(configs) == 1 else "ambiguous"}


def resolve_reference(stored: str | None, records: list[dict]) -> dict:
    if not stored:
        return {"status": "no_stored_path", "candidates": [], "path": None}
    old = [p.casefold() for p in PureWindowsPath(stored).parts]
    scored = []
    for record in records:
        parts = [p.casefold() for p in Path(record["path"]).parts]
        score = 0
        for a, b in zip(reversed(old), reversed(parts)):
            if a != b:
                break
            score += 1
        if score:
            scored.append((score, record))
    if not scored:
        return {"status": "missing", "candidates": [], "path": None}
    best = max(s for s, _ in scored)
    candidates = [r for s, r in scored if s == best]
    identical = len({r["sha256"] for r in candidates}) == 1
    chosen = sorted(candidates, key=lambda r: r["path"])[0] if identical else None
    return {"status": "unique_suffix_match" if len(candidates) == 1 else
            "equivalent_content_candidates" if identical else "ambiguous",
            "matched_suffix_parts": best, "candidates": [r["path"] for r in candidates],
            "path": chosen["path"] if chosen else None,
            "sha256": chosen["sha256"] if chosen else None}


def expand_tree(parsed: dict, records: list[dict]) -> list[dict]:
    instances = []

    def visit(model_id, parent_id, parent_matrix, ancestry, chain, suppressed, hidden, flexible):
        if model_id in ancestry:
            raise ValueError(f"Cycle in saved model tree: {model_id}")
        model = parsed["models"].get(model_id)
        if model is None:
            return
        for ref in model["references"]:
            a = ref["attributes"]
            rid = a["id"]
            instance_id = "/".join([*chain, rid])
            local = np.array(ref["transform"]["matrix_m"]) if ref["transform"] else None
            world = parent_matrix @ local if local is not None and parent_matrix is not None else None
            target = parsed["models"].get(a.get("swModelRef"))
            target_file = target["file"] if target else None
            resolution = resolve_reference(target_file.get("swPath") if target_file else None, records)
            is_suppressed = suppressed or a.get("swSuppressed") == "YES"
            is_hidden = hidden or a.get("swHidden") == "YES"
            is_flexible = flexible or a.get("swFlexible") == "YES"
            mm = world.copy() if world is not None else None
            if mm is not None:
                mm[:3, 3] *= 1000
            instances.append({"instance_id": instance_id, "parent_instance_id": parent_id,
                              "name": a.get("swComponentName", a.get("swName")),
                              "reference_number": a.get("swReferenceNumber"),
                              "model_id": a.get("swModelRef"), "saved_attributes": a,
                              "source_resolution": resolution, "source_sha256": resolution.get("sha256"),
                              "stored_file": target_file, "model_attributes": target["attributes"] if target else None,
                              "local_transform": ref["transform"], "matrix_mm": mm.tolist() if mm is not None else None,
                              "link": None, "suppressed_in_tree": is_suppressed, "hidden_in_tree": is_hidden,
                              "flexible_ancestry": is_flexible,
                              "confidence": "saved_xml_pending_geometric_validation",
                              "evidence": {"stream": "swXmlContents/COMPINSTANCETREE", "reference_id": rid,
                                           "ancestor_reference_ids": chain, "state": "last_saved"}})
            if target:
                visit(a.get("swModelRef"), instance_id, world, [*ancestry, model_id], [*chain, rid],
                      is_suppressed, is_hidden, is_flexible)

    visit(parsed["root_model_id"], None, np.eye(4), [], [], False, False, False)
    return instances


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / "urdf_work/tools/swformat/src"))
    from swformat.io.reader import read_document

    inventory = json.loads((root / "urdf_work/evidence/inventory.json").read_text(encoding="utf-8"))
    records = [r for r in inventory["files"] if r["role"] == "source_input"]
    native_records = [r for r in records if r["extension"] in {".sldprt", ".sldasm"}]
    out = root / "urdf_work/evidence/native"
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for i, record in enumerate(native_records):
        key = record["sha256"]
        result_path = out / f"{key}.json"
        try:
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                source = root / record["path"]
                if sha256_file(source) != key:
                    raise ValueError("Source hash no longer matches inventory")
                document = read_document(source)
                chunks = []
                streams = {}
                for chunk in document.chunks:
                    payload = chunk.decompressed()
                    chunks.append({"name": chunk.name, "offset": chunk.offset,
                                   "declared_bytes": chunk.usz, "decoded_bytes": len(payload) if payload is not None else None,
                                   "size_matches": payload is not None and len(payload) == chunk.usz})
                    if payload is not None and chunk.name not in streams:
                        streams[chunk.name] = payload
                result = {"path": record["path"], "source_sha256": key, "format": document.fmt,
                          "chunks": chunks, "parsed": None, "preview_path": None}
                target = root / "urdf_work/assets/native_metadata" / key
                for stream in XML_STREAMS:
                    if stream in streams:
                        target.mkdir(parents=True, exist_ok=True)
                        filename = target / (stream.rsplit("/", 1)[-1] + ".xml")
                        filename.write_bytes(streams[stream])
                        result["xml_path"] = filename.relative_to(root).as_posix()
                        result["parsed"] = parse_xml(streams[stream])
                        break
                if streams.get("PreviewPNG", b"").startswith(b"\x89PNG\r\n\x1a\n"):
                    target.mkdir(parents=True, exist_ok=True)
                    preview = target / "preview.png"
                    preview.write_bytes(streams["PreviewPNG"])
                    result["preview_path"] = preview.relative_to(root).as_posix()
                result["embedded_components"] = []
                for name, payload in streams.items():
                    if name.startswith("ImportedComp/"):
                        target.mkdir(parents=True, exist_ok=True)
                        # Never allow an in-file name to act as an output path.
                        embedded = target / ("embedded_" + str(len(result["embedded_components"])) + ".bin")
                        embedded.write_bytes(payload)
                        result["embedded_components"].append({"stream": name, "path": embedded.relative_to(root).as_posix(),
                                                              "bytes": len(payload), "sha256": sha256_file(embedded)})
                write_json(result_path, result)
            results.append({"path": record["path"], "source_sha256": key, "result_path": result_path.relative_to(root).as_posix(),
                            "format": result["format"], "preview_path": result["preview_path"],
                            "parsed": result["parsed"] is not None, "chunks": len(result["chunks"]),
                            "size_mismatches": sum(not c["size_matches"] for c in result["chunks"])})
            if record["extension"] == ".sldasm" and result["parsed"]:
                instances = expand_tree(result["parsed"], records)
                write_json(out / f"{key}.assembly.json", {"schema_version": 1, "source": record,
                           "root_model_id": result["parsed"]["root_model_id"], "instances": instances,
                           "convention": "column vectors; local XML matrix transposed; translation m converted to mm",
                           "limitations": ["Last-saved XML mirror, not an authoritative mate solver",
                                           "Flexible subassembly model reuse must be checked against top-level saved geometry"],
                           "summary": {"instances": len(instances), "leaf_instances": sum(
                               r["stored_file"] is not None and r["stored_file"].get("swDocType") == "PART" for r in instances),
                               "resolutions": dict(Counter(r["source_resolution"]["status"] for r in instances)),
                               "invalid_or_missing_transforms": sum(r["local_transform"] is None or
                                   not r["local_transform"]["rigid_valid"] for r in instances)}})
            print(f"{i+1}/{len(native_records)} {record['path']}", flush=True)
        except Exception as exc:
            results.append({"path": record["path"], "error": f"{type(exc).__name__}: {exc}"})
            print(f"ERROR {record['path']}: {exc}", flush=True)
    report = {"schema_version": 1, "reader": {"repository": "https://github.com/KenM76/swformat",
              "commit": SWFORMAT_COMMIT, "license": "Apache-2.0", "usage": "read_only"}, "files": results,
              "summary": {"native_files": len(results), "parsed_xml": sum(r.get("parsed", False) for r in results),
                          "previews": sum(bool(r.get("preview_path")) for r in results),
                          "errors": sum("error" in r for r in results),
                          "chunk_size_mismatches": sum(r.get("size_mismatches", 0) for r in results)}}
    write_json(root / "urdf_work/evidence/native_index.json", report)
    arm = next(r for r in records if r["path"] == "机械臂1/机械臂装配.SLDASM")
    assembly = json.loads((out / f"{arm['sha256']}.assembly.json").read_text(encoding="utf-8"))
    assembly["status"] = "saved_assembly_evidence_pending_geometry_and_joint_validation"
    write_json(root / "urdf_work/config/assembly.json", assembly)
    print(json.dumps(report["summary"], ensure_ascii=True))


if __name__ == "__main__":
    main()
