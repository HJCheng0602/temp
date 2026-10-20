import argparse,hashlib,json,pathlib
p=argparse.ArgumentParser();p.add_argument('--live',action='store_true');a=p.parse_args()
r=pathlib.Path(__file__).resolve().parent;m=json.loads((r/'source_manifest.json').read_text())
for rel,info in m['files'].items():
 assert hashlib.sha256((r/rel).read_bytes()).hexdigest()==info['sha256'],rel
 if a.live:assert hashlib.sha256(pathlib.Path(info['source']).read_bytes()).hexdigest()==info['sha256'],info['source']
print('VERIFIED',len(m['files']),'files; live originals checked:',a.live)
