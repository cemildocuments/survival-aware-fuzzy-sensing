"""Verify bundled UCI archives, download absent archives, and extract source data."""
from pathlib import Path
import hashlib,json,zipfile
import requests
from data import ROOT,save_metadata

def main():
 manifest=json.loads((ROOT/'data/download_manifest.json').read_text())
 for name,item in manifest.items():
  path=ROOT/'data/raw'/f'{name}.zip';path.parent.mkdir(exist_ok=True,parents=True)
  if not path.exists():
   response=requests.get(item['url'],timeout=120);response.raise_for_status();path.write_bytes(response.content)
  if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise ValueError(f'Archive checksum mismatch: {path.name}')
  target=path.parent/name;target.mkdir(exist_ok=True)
  with zipfile.ZipFile(path) as archive:
   for member in archive.infolist():
    dest=(target/member.filename).resolve()
    if not dest.is_relative_to(target.resolve()):raise ValueError('Unsafe archive member')
   archive.extractall(target)
  print(name,'verified')
 save_metadata()
if __name__=='__main__':main()
