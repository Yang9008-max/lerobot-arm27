'use strict';
// Diagnostics and bounded retry; never change the vendored reader or source CAD.
const fs=require('fs'), path=require('path');
const S=require('../../urdf_work/tools/sldprt-format-research/src');
const plan=JSON.parse(fs.readFileSync('urdf_work/evidence/native_mesh_plan.json','utf8'));
const results=JSON.parse(fs.readFileSync('urdf_work/evidence/native_mesh_parse.json','utf8'));
const out=[];
for (const row of results.results.filter(r=>r.status!=='parsed_display_mesh')) {
  if (row.source_paths[0].includes('零件414')) continue; // unrelated 200+ MB part outside robot
  const buf=fs.readFileSync(path.join(plan.root,row.source_paths[0]));
  let dl;
  if (S.container.modern.isOLE2(buf)) {
    try { dl=S.container.ole.displayLists(buf,S.inflateZlib).bytes; } catch(e) {}
  } else dl=S.container.modern.findDisplayLists(buf,S.inflateRaw,S.inflateZlib);
  if (!dl || dl.length>128*1024*1024) { out.push({...row, retry:'no_bounded_display_stream'}); continue; }
  const parsed=S.display.extractDisplayLists(dl,S.container.modern.isOLE2(buf));
  const b=Buffer.from(dl);
  const diagnostics=parsed.rejected.map(r=>{
    const o=r.offset,n=b.readUInt32LE(o+12),end=o+16+4*n;
    const lengths=Array.from({length:n},(_,i)=>b.readUInt32LE(o+16+i*4));
    const header=Array.from({length:4},(_,i)=>b.readUInt32LE(end+i*4));
    const diag={...r,strip_count:n,strip_lengths:lengths,expected_positions_header:header};
    if(n===0 && header.join(',')==='12,100,2,0') {
      diag.recovery='empty_geometry_record_no_triangles';
    } else if(header.slice(0,3).join(',')==='12,100,2' && lengths.every(v=>v>=3) &&
              lengths.reduce((a,v)=>a+v,0)===header[3] && end+16+header[3]*12<=b.length) {
      // Positions and explicit strip partitions are sufficient for visual triangles.
      // This does not guess normals, edge annotations or B-rep topology.
      const vertices=new Float32Array(header[3]*3);
      for(let i=0;i<vertices.length;i++) {
        vertices[i]=b.readFloatLE(end+16+i*4);
        if(!Number.isFinite(vertices[i])) throw Error('nonfinite recovery position');
      }
      const indices=[];let base=0;
      for(const len of lengths) {
        for(let i=2;i<len;i++)indices.push(...(i%2===0?[base+i-2,base+i-1,base+i]:[base+i-1,base+i-2,base+i]));
        base+=len;
      }
      parsed.faces.push({offset:o,vertexCount:header[3],vertices,triangleIndices:new Uint32Array(indices),metadata:null});
      diag.recovery='positions_and_explicit_triangle_strip_partition';
      diag.recovered_triangles=indices.length/3;
    } else diag.recovery='unresolved';
    return diag;
  });
  const mesh='urdf_work/assets/native_meshes/'+row.source_sha256+'_retry.stl';
  if(parsed.faces.length) fs.writeFileSync(mesh,S.convert.toSTLBinary(S.convert.loadModel(parsed),{scale:1}));
  out.push({source_sha256:row.source_sha256,source_paths:row.source_paths,display_bytes:dl.length,faces:parsed.faces.length,
    mesh_path:parsed.faces.length?mesh:null,rejected:diagnostics,warnings:parsed.warnings,
    status:diagnostics.some(r=>r.recovery==='unresolved')?'partial_display_mesh':parsed.faces.length?'parsed_display_mesh':'empty_display_stream'});
}
fs.writeFileSync('urdf_work/evidence/native_mesh_retry.json',JSON.stringify({results:out},null,2));
console.log(JSON.stringify(out.map(({warnings,...r})=>r)));
