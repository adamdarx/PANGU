"""Verify fail-fast geometric evidence using an explicitly invalid fixture."""
from pathlib import Path
import argparse
import shutil
import subprocess

import h5py


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pangu',type=Path,required=True)
    parser.add_argument('--seed',type=Path,required=True)
    parser.add_argument('--workdir',type=Path,required=True)
    parser.add_argument('--chi-psi-power',type=int,choices=[-4],default=-4)
    args=parser.parse_args()
    folder=args.workdir.resolve();folder.mkdir(parents=True,exist_ok=False)
    fixture=folder/'invalid-negative-chi.rhdf'
    shutil.copyfile(args.seed,fixture)
    with h5py.File(fixture,'r+') as output:
        state=output['nr.z4c'][...];state[:,0]=-1;output['nr.z4c'][...]=state
    with (folder/'run.log').open('wb') as log:
        result=subprocess.run(['mpirun','-np','1',str(args.pangu.resolve()),'-r',str(fixture),
                               'parthenon/time/nlim=1',
                               'numerical_relativity/chi_psi_power='+str(args.chi_psi_power)],
                              cwd=folder,stdout=log,stderr=subprocess.STDOUT)
    text=(folder/'run.log').read_text()
    assert result.returncode!=0,'Negative geometric state was accepted'
    assert 'minimum conformal variable=-1' in text,'Missing conformal-variable evidence'
    determinant='-1'
    assert 'minimum spatial metric determinant='+determinant in text,'Missing spatial-metric evidence'
    print('PASS: invalid geometry aborts with evidence; failed state is not floored.')


if __name__=='__main__':main()
