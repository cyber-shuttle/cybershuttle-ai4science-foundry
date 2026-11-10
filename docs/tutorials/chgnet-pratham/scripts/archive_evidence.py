"""Copy a completed small run into tracked evidence; omit weights and private paths."""
import argparse,json,pathlib,shutil
root=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('run_id');args=p.parse_args()
if pathlib.Path(args.run_id).name!=args.run_id or args.run_id in ['.','..']:
    raise SystemExit('Supply a run ID, not a path')
src=root/'runs'/args.run_id;dest=root/'evidence'/args.run_id
required=['results.json','execution.log','environment-lock.txt','exit-status.txt','workflow.json','manifest.json']
for name in required:
    if not (src/name).is_file():raise SystemExit('Missing required evidence: '+name)
if dest.exists():raise SystemExit('Evidence directory already exists; no overwrite performed')
dest.mkdir()
for name in required+['nvidia-smi.txt','slurm-job.txt']:
    if (src/name).exists():shutil.copy(src/name,dest/name)
# A portable pip lock for reruns: install torch explicitly and CHGNet from pinned source.
lines=(src/'environment-lock.txt').read_text().splitlines()
lock=[s for s in lines if not s.lower().startswith(('chgnet','torch==','torch @'))]
(dest/'dependencies-lock.txt').write_text('\n'.join(lock)+'\n')
print('Archived evidence to',dest)
print('Review logs and local file URLs before publishing; preserve unedited originals privately.')
