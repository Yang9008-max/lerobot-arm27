"""Index original CAD inputs without changing them."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
EXCLUDED = {"docs", "scripts", "tests", "urdf_work", "models", "datasets", ".git", ".codex", ".agents"}
SCHEMA_VERSION = 1


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_name(stem: str) -> str:
    text = unicodedata.normalize("NFKC", stem).casefold()
    text = re.sub(r"x\d+$", "", text)
    return re.sub(r"\s+", "", text)


def classify(path: str) -> str:
    if "末端" in path:
        return "arm_end"
    if any(name in path for name in ("机械臂", "大臂", "小臂")):
        return "arm"
    if "轮组" in path:
        return "wheels"
    if "底架" in path or "底盘" in path:
        return "chassis"
    if "存矿" in path:
        return "storage"
    return "other"


def step_summary(path: Path) -> dict:
    # Latin-1 is lossless for STEP entity names. The inventory path is UTF-8.
    text = path.read_bytes().decode("latin-1")
    count = lambda entity: len(re.findall(r"\b" + entity + r"\s*\(", text))
    versions = sorted(set(re.findall(r"SolidWorks\s+\d{4}", text, re.I)))
    return {
        "products": len(re.findall(r"(?m)^\s*#\d+\s*=\s*PRODUCT\s*\(", text)),
        "assembly_instances": count("NEXT_ASSEMBLY_USAGE_OCCURRENCE"),
        "brep_solids": count("MANIFOLD_SOLID_BREP"),
        "exporter_versions": versions,
    }


def build_inventory(root: Path) -> dict:
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED for part in relative.parts):
            continue
        records.append({
            "path": relative.as_posix(), "bytes": path.stat().st_size,
            "extension": path.suffix.lower(), "sha256": sha256_file(path),
            "category": classify(relative.as_posix()),
            "role": "solidworks_lock_file" if path.name.startswith("~$") else "source_input",
        })
    content = defaultdict(list)
    native = defaultdict(list)
    for record in records:
        content[record["sha256"]].append(record["path"])
        if record["extension"] == ".sldprt" and record["role"] == "source_input":
            native[normalized_name(Path(record["path"]).stem)].append(record["path"])
    steps, matches = [], []
    for record in records:
        if record["extension"] not in {".step", ".stp"}:
            continue
        steps.append({**record, **step_summary(root / record["path"])})
        candidates = native[normalized_name(Path(record["path"]).stem)]
        matches.append({
            "step_path": record["path"], "sldprt_candidates": candidates,
            "status": "unique_name_candidate" if len(candidates) == 1 else "ambiguous" if candidates else "no_name_match",
            "verified_geometry_equivalence": False,
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root), "files": records, "step_files": steps,
        "duplicate_groups": [{"sha256": key, "paths": paths} for key, paths in content.items() if len(paths) > 1],
        "cad_matches": matches,
        "summary": {
            "files": len(records), "bytes": sum(row["bytes"] for row in records),
            "source_inputs": sum(row["role"] == "source_input" for row in records),
            "solidworks_lock_files": sum(row["role"] == "solidworks_lock_file" for row in records),
            "extensions": dict(sorted(Counter(row["extension"] for row in records).items())),
            "step_paths": len(steps), "unique_step_contents": len({row["sha256"] for row in steps}),
            "step_paths_without_assembly_instances": sum(row["assembly_instances"] == 0 for row in steps),
            "cad_match_status": dict(Counter(row["status"] for row in matches)),
        },
    }


def environment_report() -> dict:
    packages = {}
    for name in ("cadquery", "cadquery-ocp", "numpy", "pillow", "py7zr", "mujoco", "pybullet", "lerobot"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "executable": sys.executable, "packages": packages}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    result = build_inventory(root)
    write_json(root / "urdf_work/evidence/inventory.json", result)
    write_json(root / "urdf_work/evidence/environment.json", environment_report())
    print(json.dumps(result["summary"], ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
