"""LOGIN compilation only; freeze a separate executable and retain old binaries."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,shutil,socket,subprocess
R=Path(__file__).resolve().parent;root=R.parent
assert socket.gethostname().startswith('wm2-login')
proof=R/'BUILD_STATUS.json';record=dict(status='login_build_running',pid=os.getpid(),started_utc=datetime.datetime.utcnow().isoformat()+'Z')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save():
 p=proof.with_suffix('.tmp');p.write_text(json.dumps(record,indent=2));p.replace(proof)
with (R/'build.lock').open('w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 old=json.loads((root/'AH_BUILD_STATUS.json').read_text());assert digest(Path(old['binary']))==old['sha256']
 record['preserved_vacuum_TOV_binary']=old['binary'];record['preserved_sha256']=old['sha256']
 record['initializer_files']={str(p.relative_to(R)):digest(p) for p in (R/'TableBNS').rglob('*') if p.is_file()}
 record['source_commits']={name:subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(root/'Cactus/repos'/name)).decode().strip() for name in ['AsterX','CarpetX','SpacetimeX']}
 save()
 with (R/'build.log').open('w') as log:rc=subprocess.call(['bash',str(R/'build.sh')],stdout=log,stderr=subprocess.STDOUT)
 record['returncode']=rc
 if rc:record['status']='build_failed';save();raise SystemExit(rc)
 exe=root/'Cactus/exe/cactus_wm2cuda';sha=digest(exe);target=root/'bin'/('cactus-bns-'+sha[:12]);assert not target.exists();shutil.copyfile(exe,target);target.chmod(0o555)
 images=subprocess.check_output(['/lustre/software/cuda/12.6.0/bin/cuobjdump','--list-elf',str(target)]).decode();assert 'ELF file' in images;(R/'cuda-images.txt').write_text(images)
 assert digest(Path(old['binary']))==old['sha256']
 for rel,sha0 in record['initializer_files'].items():assert digest(R/rel)==sha0
 record.update(status='compiled_frozen_GPU_admission_required',binary=str(target),sha256=sha,CUDA_device_images_verified=True,finished_utc=datetime.datetime.utcnow().isoformat()+'Z');save()
 print(json.dumps(record,indent=2))
