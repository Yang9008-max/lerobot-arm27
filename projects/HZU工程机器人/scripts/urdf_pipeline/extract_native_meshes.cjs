'use strict';
// Read-only adapter around the pinned MIT display-mesh reader. Output meters.
const fs = require('fs');
const path = require('path');
const S = require('../../urdf_work/tools/sldprt-format-research/src');
const plan = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const results = [];
for (let i = 0; i < plan.parts.length; i++) {
  const item = plan.parts[i];
  const row = {source_sha256:item.sha256, source_paths:item.paths};
  try {
    const parsed = S.parse(path.join(plan.root, item.paths[0]));
    row.errors = parsed.errors || [];
    row.rejected = parsed.rejected || [];
    row.warnings = parsed.warnings || [];
    row.faces = parsed.faces.length;
    if (row.errors.length || row.rejected.length || !row.faces) {
      row.status = 'incomplete_or_failed';
    } else {
      const model = S.convert.loadModel(parsed);
      const bytes = S.convert.toSTLBinary(model, {scale:1}); // cache vertices already m
      const target = path.join(plan.root, 'urdf_work/assets/native_meshes', item.sha256 + '.stl');
      fs.mkdirSync(path.dirname(target), {recursive:true});
      fs.writeFileSync(target, bytes);
      row.status = 'parsed_display_mesh';
      row.triangles = model.triangleCount;
      row.mesh_unit = 'm';
      row.mesh_path = 'urdf_work/assets/native_meshes/' + item.sha256 + '.stl';
    }
  } catch (error) {
    row.status = 'error'; row.error = error.message;
  }
  results.push(row);
  if ((i+1)%50 === 0) console.log(JSON.stringify({processed:i+1, total:plan.parts.length}));
}
fs.writeFileSync(path.join(plan.root,'urdf_work/evidence/native_mesh_parse.json'),JSON.stringify({
  schema_version:1, reader:plan.reader, parser_scale:1, original_files_modified:false, results
},null,2));
console.log(JSON.stringify({processed:results.length, success:results.filter(r=>r.status==='parsed_display_mesh').length,
  failed:results.filter(r=>r.status!=='parsed_display_mesh').map(r=>({paths:r.source_paths,status:r.status,error:r.error,errors:r.errors,rejected:r.rejected.length}))}));
