"""Audit recovered display meshes against saved full-assembly local bounds."""
import json
from collections import Counter
import numpy as np
from joint_evidence import Evidence, load
from inventory import ROOT, write_json
from link_mesh_io import decode_stl

def main():
    e = Evidence()
    meshes = {r['source_sha256']: r for r in load('urdf_work/evidence/native_mesh_parse.json')['results']}
    bounds = {}
    for key, row in meshes.items():
        if row['status'] == 'parsed_display_mesh':
            t, _ = decode_stl((ROOT / row['mesh_path']).read_bytes())
            bounds[key] = np.r_[t.min(axis=(0, 1)), t.max(axis=(0, 1))]
    instances = e.assembly('HZU工程完整装配.SLDASM')['instances']
    parents = {r['parent_instance_id'] for r in instances}
    rows = []
    for r in instances:
        if r['instance_id'] in parents: continue
        key = r.get('source_sha256')
        m = meshes.get(key, {})
        row = {k:r.get(k) for k in ['instance_id','name','source_sha256','source_resolution','hidden_in_tree','suppressed_in_tree']}
        row['status'] = m.get('status','unresolved_source')
        if key in bounds and r.get('model_attributes',{}).get('swBoundingBox'):
            saved = np.array(list(map(float, r['model_attributes']['swBoundingBox'].split())))
            row['bbox_error_mm'] = float(abs(bounds[key] - saved).max()*1000)
        rows.append(row)
    write_json(ROOT / 'urdf_work/evidence/full_mesh_audit.json', {'items':rows})
    print('STATUS', Counter(r['status'] for r in rows))
    print('BBOX WORST', json.dumps(sorted([r for r in rows if 'bbox_error_mm' in r],key=lambda r:r['bbox_error_mm'],reverse=True)[:15],ensure_ascii=False))
    print('FAILED',json.dumps([r for r in rows if r['status']!='parsed_display_mesh'],ensure_ascii=False))
    for path in ['机械臂1/机械臂装配.SLDASM','末端零件/末端装配.SLDASM']:
        print(path)
        print(json.dumps([{k:r.get(k) for k in ['instance_id','name','source_resolution','hidden_in_tree','suppressed_in_tree']} for r in e.assembly(path)['instances']],ensure_ascii=False))
if __name__ == '__main__': main()
