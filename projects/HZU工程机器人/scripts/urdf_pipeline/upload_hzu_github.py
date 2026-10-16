"""Publish the prepared model on an isolated branch using authenticated gh API.

No token is read or logged. Does not update main or overwrite existing branches.
Blob cache makes interrupted uploads resumable; branch publication is last.
"""
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time

from inventory import ROOT

REPO='Yang9008-max/lerobot-arm27'
BRANCH='hzu-urdf-20261015'
PAYLOAD=ROOT/'urdf_work/publish/hzu_upload'
STATE=ROOT/'urdf_work/publish/github_upload_state.json'
INPUTS=ROOT/'urdf_work/publish/api_inputs'
LOCK=threading.Lock()


def api(endpoint,body=None):
    cmd=['gh','api',endpoint]
    target=None
    if body is not None:
        INPUTS.mkdir(parents=True,exist_ok=True)
        target=INPUTS/(str(threading.get_ident())+'.json')
        target.write_text(json.dumps(body,ensure_ascii=True,separators=(',',':')),encoding='utf-8')
        cmd+=['--method','POST','--input',str(target)]
    try:
        for attempt in range(4):
            result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=300)
            if result.returncode==0:
                return json.loads(result.stdout)
            error=result.stderr.decode('utf-8',errors='replace')
            if 'HTTP 404' in error or 'HTTP 422' in error: raise RuntimeError(error[:1000])
            if attempt==3: raise RuntimeError(error[:1000])
            time.sleep(2*(attempt+1))
    finally:
        if target and target.is_file(): target.unlink()


def main():
    state=json.loads(STATE.read_text(encoding='utf-8')) if STATE.exists() else {'repo':REPO,'branch':BRANCH,'blobs':{}}
    if state['repo']!=REPO or state['branch']!=BRANCH: raise ValueError('Unexpected upload state')
    head=api('repos/'+REPO+'/git/ref/heads/main')['object']['sha']
    commit=api('repos/'+REPO+'/git/commits/'+head)
    base_tree=api('repos/'+REPO+'/git/trees/'+commit['tree']['sha']+'?recursive=1')
    remote={r['path']:r for r in base_tree['tree']}
    if base_tree.get('truncated'): raise ValueError('Cannot safely inspect truncated remote tree')
    if any(p.endswith('AGENTS.md') for p in remote):
        raise ValueError('Remote AGENTS.md appeared: read instructions before uploading')
    files={p.relative_to(PAYLOAD).as_posix():p.read_bytes() for p in PAYLOAD.rglob('*') if p.is_file()}
    collision=[p for p in files if p in remote and p not in ['README.md']]
    if collision: raise ValueError('Existing remote model files require review: '+str(collision[:10]))
    readme=api('repos/'+REPO+'/contents/README.md?ref='+head)
    content=base64.b64decode(readme['content']).decode('utf-8')
    section='''

## HZU 整机 URDF（独立模型）

新增 [HZU 模型说明与整机预览](models/hzu_robot/README.md)：包含底盘、四轮、主要电机与六轴机械臂；提供可由 MuJoCo 直接加载的 URDF、188 项网格和独立查看脚本。该模型与本仓库现有 `arm_27` 三轴硬件接口不同，当前用途为整机几何与输出侧运动学检查，物理与控制接入边界见模型说明。

```powershell
python -m pip install -r models/hzu_robot/requirements.txt
python tools/view_hzu.py --check
python tools/view_hzu.py
```
'''
    if '## HZU 整机 URDF（独立模型）' not in content:
        content=content.rstrip()+section
    files['README.md']=content.encode('utf-8')
    expected={p:hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest() for p,data in files.items()}
    print(json.dumps({'repo':REPO,'branch':BRANCH,'base_commit':head,'files':len(files),
                      'bytes':sum(map(len,files.values()))}),flush=True)

    def save():
        STATE.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

    def upload(item):
        path,data=item
        sha=expected[path]
        if sha in state['blobs']: return path,sha,True
        response=api('repos/'+REPO+'/git/blobs',{'encoding':'base64','content':base64.b64encode(data).decode('ascii')})
        if response['sha']!=sha: raise ValueError('Remote blob content mismatch: '+path)
        with LOCK:
            state['blobs'][sha]={'path':path,'bytes':len(data)}
            save()
        return path,sha,False

    # Start with large assets so upload problems surface before most small requests.
    items=sorted(files.items(),key=lambda item:len(item[1]),reverse=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(upload,item) for item in items]
        for done,future in enumerate(as_completed(futures),1):
            path,sha,cached=future.result()
            if done%10==0 or done==len(items):
                print(json.dumps({'uploaded_blobs':done,'total':len(items),'last_file':path}),flush=True)
    tree=api('repos/'+REPO+'/git/trees',{'base_tree':commit['tree']['sha'],
             'tree':[{'path':p,'mode':'100644','type':'blob','sha':expected[p]} for p in sorted(files)]})
    state['tree_sha']=tree['sha'];state['base_commit']=head;save()
    new_commit=api('repos/'+REPO+'/git/commits',{'message':'Add HZU whole-robot visual and kinematic URDF',
                    'tree':tree['sha'],'parents':[head]})
    state['commit_sha']=new_commit['sha'];save()
    # Do not advance an existing branch, main, tags or any other ref.
    api('repos/'+REPO+'/git/refs',{'ref':'refs/heads/'+BRANCH,'sha':new_commit['sha']})
    state['published']=True;save()
    final_tree=api('repos/'+REPO+'/git/trees/'+tree['sha']+'?recursive=1')
    actual={r['path']:r['sha'] for r in final_tree['tree'] if r['type']=='blob'}
    if any(actual.get(p)!=digest for p,digest in expected.items()):
        raise ValueError('Published tree mismatch')
    body='''Adds the HZU whole robot model under `models/hzu_robot` so it can be inspected independently in MuJoCo. Includes chassis, four wheel outputs, six arm outputs and ten main motor exteriors: 578 visual instances using 188 OBJ meshes.

The HZU model is separate from the existing three-axis `arm_27` hardware interface. Its README documents saved CAD zero, geometry provenance, composite motor visuals, fixed passive mechanisms and missing collision/dynamics/drive calibration. Original CAD and duplicate, unreferenced STL files are excluded.

Validation: the distributed assets passed SHA-256 checks, actual URDF loading in MuJoCo 3.14.0 and 20 single-coordinate smoke checks with `python tools/view_hzu.py --check`. The original export report includes all-vertex readback, 27 kinematic configurations, 12 wheel-axis checks and 192 source-file hashes. No dynamics or LeRobot training is claimed.
'''
    pr=api('repos/'+REPO+'/pulls',{'title':'Add HZU whole-robot URDF and standalone MuJoCo viewer',
                                'head':BRANCH,'base':'main','draft':True,'body':body})
    state['pull_request']=pr['html_url'];state['remote_blob_hashes_verified']=True;save()
    print(json.dumps({'status':'uploaded_and_verified','repo':REPO,'branch':BRANCH,'commit':state['commit_sha'],
                      'files':len(files),'pull_request':pr['html_url'],
                      'model_url':'https://github.com/'+REPO+'/tree/'+BRANCH+'/models/hzu_robot'},ensure_ascii=False),flush=True)


if __name__=='__main__': main()
