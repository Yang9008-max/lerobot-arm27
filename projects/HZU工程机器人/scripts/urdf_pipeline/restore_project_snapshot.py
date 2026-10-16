"""Verify and restore a split ZIP project snapshot using only Python's standard library."""
import argparse
import bisect
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


class SplitReader(io.RawIOBase):
    """Seek over byte-split ZIP volumes without copying them to a temp file."""
    def __init__(self,paths):
        super().__init__()
        self.paths=paths
        self.offsets=[0]
        for path in paths: self.offsets.append(self.offsets[-1]+path.stat().st_size)
        self.position=0
        self.handle=None
        self.part_index=None

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self,offset,whence=0):
        position=offset if whence==0 else self.position+offset if whence==1 else self.offsets[-1]+offset if whence==2 else -1
        if position<0: raise ValueError('Invalid ZIP seek')
        self.position=position
        return position

    def read(self,size=-1):
        remaining=max(0,self.offsets[-1]-self.position)
        size=remaining if size is None or size<0 else min(size,remaining)
        chunks=[]
        while size:
            index=bisect.bisect_right(self.offsets,self.position)-1
            if index!=self.part_index:
                if self.handle: self.handle.close()
                self.handle=self.paths[index].open('rb')
                self.part_index=index
            local=self.position-self.offsets[index]
            self.handle.seek(local)
            count=min(size,self.offsets[index+1]-self.position)
            data=self.handle.read(count)
            if len(data)!=count: raise ValueError('Truncated archive volume')
            chunks.append(data);size-=count;self.position+=count
        return b''.join(chunks)

    def close(self):
        if self.handle: self.handle.close()
        super().close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--overwrite',action='store_true',help='Replace differing files in the output directory')
    args=parser.parse_args()
    bundle=args.bundle.resolve()
    manifest=json.loads((bundle/'archive_manifest.json').read_text(encoding='utf-8'))
    output=(args.output or bundle).resolve()
    paths=[]
    for part in manifest['parts']:
        path=(bundle/part['path']).resolve()
        if not path.is_relative_to(bundle): raise ValueError('Unsafe part path')
        if path.stat().st_size!=part['bytes'] or sha256(path)!=part['sha256']:
            raise ValueError('Archive part size/hash mismatch: '+part['path'])
        paths.append(path)
    with SplitReader(paths) as packed:
        expected={r['path']:r for r in manifest['files']}
        with zipfile.ZipFile(packed) as archive:
            names=archive.namelist()
            if len(names)!=len(set(names)) or set(names)!=set(expected):
                raise ValueError('ZIP entries differ from manifest')
            # Validate every entry and check all overwrite conflicts before writing any file.
            for index,name in enumerate(names,1):
                target=(output/name).resolve()
                if not target.is_relative_to(output): raise ValueError('Unsafe ZIP path: '+name)
                row=expected[name]
                digest=hashlib.sha256()
                count=0
                with archive.open(name) as source:
                    while block:=source.read(1024*1024): count+=len(block);digest.update(block)
                if count!=row['bytes'] or digest.hexdigest()!=row['sha256']:
                    raise ValueError('Archived file hash mismatch: '+name)
                if not args.check and target.exists() and not args.overwrite and sha256(target)!=row['sha256']:
                    raise ValueError('Existing file differs; choose a new --output directory or use --overwrite: '+str(target))
                if index%100==0 or index==len(names):
                    print(json.dumps({'verified_files':index,'total':len(names)}),flush=True)
            if not args.check:
                for name in names:
                    target=output/name
                    if target.is_file() and sha256(target)==expected[name]['sha256']: continue
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with archive.open(name) as source,target.open('wb') as dest: shutil.copyfileobj(source,dest,1024*1024)
        print(json.dumps({'status':'verified' if args.check else 'restored','files':len(expected),
                          'original_source_files':manifest['original_source_files'],
                          'all_file_hashes_match':True,'output':None if args.check else str(output)},ensure_ascii=False),flush=True)


if __name__=='__main__': main()
