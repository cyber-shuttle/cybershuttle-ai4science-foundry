"""Inference, provenance, reference comparison, and physical symmetry validation."""
import argparse, hashlib, importlib.metadata, json, os, platform, resource, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from pymatgen.core import Structure
from pymatgen.core.operations import SymmOp
from chgnet.model.model import CHGNet
from validation import smoke, compare, RTOL, UNITS

ROOT=Path(__file__).resolve().parents[1]
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    cfg=json.loads((ROOT/'config/workflow.json').read_text())
    ref=json.loads((ROOT/'config/reference.json').read_text())
    upstream=ROOT/'artifacts/upstream'
    commit=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
    if commit != cfg['source_commit']: raise RuntimeError('Upstream commit mismatch')
    manifest=json.loads((ROOT/'artifacts/manifest.json').read_text())
    for relative,expected in manifest['sha256'].items():
        if digest(upstream/relative) != expected: raise RuntimeError(f'Artifact changed: {relative}')
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use a GPU allocation or explicitly select --device cpu.')
    torch.set_num_threads(cfg['threads']); torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
    struct=Structure.from_file(upstream/cfg['input_relative_path'])
    start=time.perf_counter()
    model=CHGNet.load(model_name=cfg['checkpoint'],use_device=args.device);model.eval()
    if args.device=='cuda': torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    inference_start=time.perf_counter()
    baseline=model.predict_structure(struct)
    if args.device=='cuda':torch.cuda.synchronize()
    inference_seconds=time.perf_counter()-inference_start
    smoke_checks=smoke(baseline)
    reference_checks={k:compare(baseline[k],ref['outputs'][k],RTOL[k],1e-4) for k in UNITS}
    for key,item in reference_checks.items(): item['unit']=UNITS[key]
    symmetry={}
    for degrees in [30,60,120]:
        op=SymmOp.from_axis_angle_and_translation([1,2,3],degrees)
        rotated=struct.copy();rotated.apply_operation(op)
        R=op.rotation_matrix
        result=model.predict_structure(rotated)
        expected={'e':baseline['e'],'m':baseline['m'],
                  'f':np.asarray(baseline['f'])@R.T,
                  's':R@np.asarray(baseline['s'])@R.T}
        symmetry[f'rotation_{degrees}']={k:compare(result[k],expected[k],atol=cfg['symmetry_absolute_tolerance'][k]) for k in UNITS}
    shifted=struct.copy();shifted.translate_sites(list(range(8)),[0.13,0.27,0.19],frac_coords=True)
    p=model.predict_structure(shifted)
    symmetry['translation']={k:compare(p[k],baseline[k],atol=cfg['symmetry_absolute_tolerance'][k]) for k in UNITS}
    order=np.array([3,0,7,2,5,1,6,4])
    perm=Structure(struct.lattice,[struct.species[i] for i in order],struct.frac_coords[order])
    p=model.predict_structure(perm)
    symmetry['permutation']={k:compare(p[k],np.asarray(baseline[k])[order] if k in ['f','m'] else baseline[k],atol=cfg['symmetry_absolute_tolerance'][k]) for k in UNITS}
    if args.device=='cuda':torch.cuda.synchronize()
    passed=all(smoke_checks.values()) and all(x['passed'] for x in reference_checks.values()) and all(x['passed'] for v in symmetry.values() for x in v.values())
    report={'passed':passed,'timestamp_utc':datetime.now(timezone.utc).isoformat(),
      'workflow':cfg,'reference_source':ref['source'],'smoke':smoke_checks,
      'reference_checks':reference_checks,'symmetry_checks':symmetry,
      'predictions':{k:np.asarray(baseline[k]).tolist() for k in UNITS},
      'timing':{'inference_seconds':inference_seconds,'load_inference_validation_seconds':time.perf_counter()-start},
      'resources':{'hostname':platform.node(),'device':args.device,'threads':torch.get_num_threads(),
        'peak_process_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024 if platform.system()=='Linux' else 1024**2),
        'gpu':torch.cuda.get_device_name() if args.device=='cuda' else None,
        'gpu_peak_allocated_MiB':torch.cuda.max_memory_allocated()/1024**2 if args.device=='cuda' else 0,
        'gpu_peak_reserved_MiB':torch.cuda.max_memory_reserved()/1024**2 if args.device=='cuda' else 0},
      'python':platform.python_version(),'packages':{k:importlib.metadata.version(k) for k in ['chgnet','torch','numpy','pymatgen']},
      'artifacts':manifest,'script_sha256':digest(Path(__file__)),
      'reference_sha256':digest(ROOT/'config/reference.json'),
      'slurm':{k:v for k,v in os.environ.items() if k in ['SLURM_JOB_ID','SLURM_JOB_PARTITION','SLURM_JOB_ACCOUNT','SLURM_CPUS_PER_TASK','SLURM_MEM_PER_NODE']}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Smoke:',smoke_checks)
    for k,v in reference_checks.items():print(f'Reference {k}: {v}')
    for k,v in symmetry.items():print(f'Symmetry {k}: {all(x["passed"] for x in v.values())}')
    print('Overall:', 'PASS' if passed else 'FAIL', 'Saved:',args.output)
    return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
