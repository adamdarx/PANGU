"""Archive closed N128 matter snapshots without changing running jobs or inputs."""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
STATE = HERE / 'archive-state'
DISK = Path('/mnt/d/PANGU_TEST_DATA/PANGU_II_NR_DATA')
DEST = DISK / 'Operations'
RECORDS = DEST / '20261116-matter-n128/archive-records'
SSH = ['ssh', '-S', '/home/adamdarx/.ssh/pku-data-5d-v1', '-o', 'ControlMaster=no',
       '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15',
       '-o', 'ServerAliveCountMax=2', '2601110206@wm2-data.pku.edu.cn']
VALIDATOR = Path('/home/adamdarx/archive-local/ArchiveControl/20261116-single-incremental/remote_validate.py').read_text()
DISCOVER = r'''
ROOT = Path('/lustre/home/2601110206/CODEX_WORKPLACE/Operations')
entries = []
states = []
for name, job, target in [('tov','17355476',100.),('bns','17355477',200.)]:
    work = ROOT/'20261116-matter-n128'/name/'runs/n128'/job
    folder = work/'formal'
    summary = json.loads((work/'summary.json').read_text()) if (work/'summary.json').exists() else {}
    states.append(dict(case=name,job=job,status=summary.get('status','queued_or_not_started')))
    if not (folder/'run.log').is_file(): continue
    with (folder/'run.log').open('rb') as f:
        f.seek(max(0,(folder/'run.log').stat().st_size-131072))
        text = f.read().decode(errors='replace')
    stamps = re.findall(r'\btime=\s*([0-9.eE+-]+)',text)
    if not stamps: continue
    latest = float(stamps[-1])
    complete = (folder/'phase-validated.json').is_file()
    for path in sorted(folder.iterdir()):
        numbered = re.search(r'\.(\d{5})\.(phdf|rhdf)$',path.name)
        final = re.search(r'\.final\.(phdf|rhdf)$',path.name)
        if not numbered and not (final and complete): continue
        st = path.stat()
        if path.is_symlink() or st.st_uid!=71767 or time.time()-st.st_mtime<120: continue
        try:
            with h5py.File(path,'r') as f: stamp=float(f['Info'].attrs['Time'])
            if not math.isfinite(stamp) or not 0<=stamp<=target+.01: continue
            if not complete and latest<=stamp+.01: continue
            st2 = path.stat()
            if (st.st_size,st.st_mtime_ns)!=(st2.st_size,st2.st_mtime_ns): continue
            entries.append(dict(group=name,case=name,job=job,path=str(path.relative_to(ROOT)),
                absolute_source=str(path),bytes=st.st_size,mtime_ns=st.st_mtime_ns,time=stamp,
                kind='field' if path.suffix=='.phdf' else 'checkpoint',remove_after_verified=False,
                closure_basis='Immutable numbered file; writer advanced beyond its time; age>120s and stable stat, or completed phase validation',
                run_complete=complete,target_time=target,latest_logged_time=latest))
        except (OSError,ValueError): continue
print(json.dumps(dict(entries=entries,states=states)))
'''


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def numerical(source):
    script = "module load anaconda3/2024.10.1\npython - <<'MATTER_N128_ARCHIVE_PY'\n" + VALIDATOR + '\n' + source + '\nMATTER_N128_ARCHIVE_PY\n'
    return json.loads(subprocess.check_output(SSH+['bash -l -s'], input=script.encode(), timeout=2400))


def save_status(status):
    status['updated_utc'] = now()
    for base in [STATE, RECORDS]:
        if base==RECORDS and not os.path.ismount('/mnt/d'): continue
        base.mkdir(parents=True, exist_ok=True)
        path=base/'STATUS.json';tmp=base/'STATUS.tmp'
        tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(path)


def clean_completed_fields(archive,index,entries):
    """Only completed phases; retain every restart and final snapshot."""
    removed=0
    for entry in entries:
        if not entry['run_complete'] or entry['kind']!='field' or '.final.' in entry['path']:
            continue
        old=index['files'].get(entry['path'])
        if not old or old.get('server_removed'):continue
        target=Path(old['archive'])
        assert archive.archive_available() and target.is_file()
        assert archive.archive_digest(target)==old['sha256']
        receipt=dict(archive=str(target),sha256=old['sha256'],bytes=old['bytes'],verified_utc=now(),
            policy='Completed phase numbered field only; fresh source/decompressed D SHA; restarts and final fields retained')
        source=entry['absolute_source']
        script=('import os,json,hashlib,re\nfrom pathlib import Path\n'
            'p=Path('+repr(source)+')\n'
            'assert p.parent==Path('+repr(str(Path(source).parent))+')\n'
            'assert p.parts[-7:-3]==("'+ '20261116-matter-n128' +'",'+repr(entry['case'])+',"runs","n128")\n'
            'assert p.parent.name=="formal" and p.parent.parent.name=='+repr(entry['job'])+'\n'
            'assert (p.parent/"phase-validated.json").is_file()\n'
            'assert re.search(r"\\.\\d{5}\\.phdf$",p.name) and not p.is_symlink()\n'
            's=p.stat();assert s.st_uid==71767 and s.st_size=='+repr(old['bytes'])+' and s.st_mtime_ns=='+repr(old['mtime_ns'])+'\n'
            'h=hashlib.sha256()\nwith p.open("rb") as f:\n'
            ' for c in iter(lambda:f.read(8388608),b""):h.update(c)\n'
            'assert h.hexdigest()=='+repr(old['sha256'])+'\n'
            's2=p.stat();assert (s.st_size,s.st_mtime_ns)==(s2.st_size,s2.st_mtime_ns)\n'
            '(p.parent/(p.name+".archive.json")).write_text(json.dumps('+repr(receipt)+',indent=2)+"\\n")\n'
            'p.unlink()\nprint("completed field duplicate removed")\n')
        archive.remote_python(script)
        old.update(server_removed=True,cleanup_receipt=receipt)
        assert archive.save(index)
        removed+=1
    return removed


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE/'watch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        (STATE/'watch.pid').write_text(str(os.getpid())+'\n')
        sys.path.insert(0,'/home/adamdarx/archive-local/ArchiveControl/20261114/runtime/validation/hpc/binary_stall_20261103')
        import archive_qnm as worker
        archive=worker.archive
        worker.SSH=SSH+['bash --noprofile --norc -s'];archive.SSH=SSH
        worker.STAGING=STATE/'staging';worker.STAGING.mkdir(exist_ok=True)
        archive.HERE=worker.HERE=STATE
        archive.BASE='/lustre/home/2601110206/CODEX_WORKPLACE/Operations'
        archive.DEST=DEST;archive.RECORDS=RECORDS
        archive.ALLOWED={'20261116-matter-n128'}
        archive.ZSTD='/home/adamdarx/.local/pangu-host/bin/zstd'
        assert archive.REMOTE_READ_ONLY
        path=STATE/'live-archive-index.json'
        index=json.loads(path.read_text()) if path.exists() else dict(files={})
        status=dict(status='starting',pid=os.getpid(),remote_read_only_during_evolution=True,source_originals_deleted=False,
            retention='All running-job files retained: their completion scripts still scan the complete time series. Cleanup follows completed phase validation.',
            jobs={'tov':'17355476','bns':'17355477'},failed=[])
        while True:
            try:
                assert os.path.ismount('/mnt/d') and archive.archive_available()
                inventory=numerical(DISCOVER)
                status.update(status='watching',cases=inventory['states'],known_closed_files=len(inventory['entries']))
                save_status(status)
                early_removed=clean_completed_fields(archive,index,inventory['entries'])
                status.update(completed_phase_numbered_fields_removed_this_poll=early_removed,source_originals_deleted=any(e.get('server_removed') for e in index['files'].values()))
                save_status(status)
                for entry in inventory['entries']:
                    old=index['files'].get(entry['path'])
                    if old:
                        assert old['sha256']==old['numerical_validation_sha256']
                        assert (old['bytes'],old['mtime_ns'])==(entry['bytes'],entry['mtime_ns']), 'Archived source changed'
                        target=Path(old['archive'])
                        assert target.is_file() and target.stat().st_size==old['compressed_bytes'], 'Archive unavailable or changed'
                        continue
                    status.update(status='validating',current=entry['path']);save_status(status)
                    proof=numerical('print(json.dumps(validate('+repr(entry)+')))')
                    assert proof['status']=='snapshot_checks_passed'
                    token=hashlib.sha256(entry['absolute_source'].encode()).hexdigest()+'.json'
                    for base in [STATE, RECORDS]:
                        folder=base/'validation';folder.mkdir(parents=True,exist_ok=True)
                        (folder/token).write_text(json.dumps(proof,indent=2)+'\n')
                    transfer=dict(entry,numerical_validation_sha256=proof['source_sha256'],snapshot_validation='passed',run_physical_acceptance='pending_convergence')
                    status.update(status='transferring');save_status(status)
                    target=DEST/Path(entry['path']).with_name(Path(entry['path']).name+'.zst')
                    if target.exists():
                        target.rename(target.with_name(target.name+'.preserved-'+str(time.time_ns())))
                    archive.transfer(transfer,index)
                    assert index['files'][entry['path']]['sha256']==proof['source_sha256']
                    status.update(verified_files=len(index['files']),last_verified=entry['path'],status='verified')
                    save_status(status)
                removed=clean_completed_fields(archive,index,inventory['entries'])
                status['completed_phase_numbered_fields_removed_this_poll']=early_removed+removed
                status['source_originals_deleted']=any(e.get('server_removed') for e in index['files'].values())
                status.update(status='watching',verified_files=len(index['files']),failed=[]);save_status(status)
            except Exception as error:
                status.update(status='retry_pending',failed=[repr(error)]);save_status(status)
                print('ARCHIVE_RETRY',repr(error),flush=True)
            if '--once' in sys.argv: return
            time.sleep(300)


if __name__=='__main__':
    main()
