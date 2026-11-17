"""Check the actual C++ initializer against archived PANGU point samples.

This is static initial-data analysis on the host, never CPU time evolution.
Inputs are byte-verified compact archives prepared by prepare_bns_reference.py.
"""
from pathlib import Path
import hashlib,json,subprocess,sys
import numpy as np
R=Path(__file__).resolve().parent;reference=Path(sys.argv[1]);table=R/'bns.dat'
assert hashlib.sha256(table.read_bytes()).hexdigest()==json.loads((R/'TABLE_PROOF.json').read_text())['sha256']
source=(R/'TableBNS/src/table_bns.cxx').read_text();body=source.split('struct Table {',1)[1].split('static Table table;',1)[0]
program='#include <algorithm>\n#include <cmath>\n#include <fstream>\n#include <iostream>\n#include <iomanip>\n#include <vector>\nstruct Table {'+body+'''
int main(int argc,char **argv){Table d;std::ifstream f(argv[1]);std::string schema;std::getline(f,schema);double residual;
f>>d.nr>>d.nl>>d.ns>>d.dr>>d.ds>>d.kappa>>d.separation>>d.surface>>residual;
d.moments.resize(d.nl);d.source.resize(d.ns);d.coefficients.resize(d.nr*d.nl);
for(auto &v:d.moments)f>>v;for(auto &v:d.source)f>>v;for(auto &v:d.coefficients)f>>v;
double x,y,z;std::cout<<std::setprecision(17);
while(std::cin>>x>>y>>z){double p=d.Psi(x,y,z),a=z-d.separation/2,b=z+d.separation/2;
double e=(d.Source(sqrt(x*x+y*y+a*a))+d.Source(sqrt(x*x+y*y+b*b)))/pow(p,8);
double rho=2*e/(1+sqrt(1+4*d.kappa*e));std::cout<<p<<" "<<std::max(rho,1e-14)<<"\\n";}}
'''
p=R/'sample_check.cxx';p.write_text(program);subprocess.run(['g++','-O2','-std=c++17',str(p),'-o',str(R/'sample_check')],check=True)
result={};rng=np.random.default_rng(20261117)
for n in [64,96]:
 p=reference/('bns_n%d_initial.npz'%n);proof=json.loads(p.with_suffix('.json').read_text());assert hashlib.sha256(p.read_bytes()).hexdigest()==proof['npz_sha256'];a=np.load(p)
 # Existing cache's legacy alpha key stores nr.adm[0]=gxx; new caches use gxx.
 rho=a['rho'];gxx=a['gxx'] if 'gxx' in a.files else a['alpha'];geo=a['geometry'];candidates=np.argwhere(rho>1e-10);stellar=candidates[rng.choice(len(candidates),min(12000,len(candidates)),replace=False)]
 random=np.column_stack([rng.integers(0,s,size=4000) for s in rho.shape]);idx=np.concatenate([stellar,random]);points=[];expected_rho=[];expected_psi=[]
 for m,k,j,i in idx:
  xyz=[geo[m,2*d]+(q+.5)*(geo[m,2*d+1]-geo[m,2*d])/rho.shape[3-d] for d,q in enumerate([i,j,k])]
  points.append(xyz);expected_rho.append(rho[m,k,j,i]);expected_psi.append(gxx[m,k,j,i]**.25)
 text='\n'.join(' '.join('%.17g'%v for v in xyz) for xyz in points)+'\n'
 values=np.loadtxt(subprocess.check_output([str(R/'sample_check'),str(table)],input=text.encode()).decode().splitlines());truth=np.array(expected_rho);psi=np.array(expected_psi)
 err=np.abs(values[:,1]-truth);relative_l1=float(err.sum()/np.abs(truth).sum());psi_error=float(np.max(abs(values[:,0]/psi-1)))
 result[str(n)]=dict(samples=len(idx),rho_relative_L1=relative_l1,rho_Linf=float(err.max()),psi_relative_Linf=psi_error,source_phdf_sha256=proof['source_phdf_sha256'])
 print(n,result[str(n)],flush=True)
 assert relative_l1<1e-6 and psi_error<1e-6
result['scope']='Static actual initializer function samples versus archived PANGU time-zero fields; no CPU evolution or independent constraint-solver claim.'
(R/'ARCHIVED_INITIAL_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
