"""Independent harmonic initial-data oracles for the ADM monopole fit."""
from pathlib import Path
import sys
import unittest

import numpy as np
from numpy.polynomial.legendre import leggauss

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reference'))
from initial_adm_monopole import fit_monopole


class InitialADMTests(unittest.TestCase):
    def samples(self):
        c,w=leggauss(24);r,c=np.meshgrid(np.linspace(30.25,89.75,121),c)
        return r.ravel(),c.ravel(),np.broadcast_to(w[:,None],r.shape).ravel()*r.ravel()**2

    def test_exact_schwarzschild_mass(self):
        r,c,w=self.samples();M=2.6548889790689447
        rows=fit_monopole(r,c,1+M/(2*r),w)
        self.assertLess(max(abs(v['ADM_mass']/M-1) for v in rows),1e-12)

    def test_two_separated_harmonic_sources(self):
        r,c,w=self.samples();m=1.1;a=12
        psi=1+m/(2*np.sqrt(r*r+a*a-2*r*a*c))+m/(2*np.sqrt(r*r+a*a+2*r*a*c))
        rows=fit_monopole(r,c,psi,w)
        self.assertLess(max(abs(v['ADM_mass']/(2*m)-1) for v in rows),1e-12)

    def test_missing_radial_coverage_rejected(self):
        r,c,w=self.samples();mask=r<60
        with self.assertRaises(ValueError):fit_monopole(r[mask],c[mask],np.ones(mask.sum()),w[mask])


if __name__=='__main__':unittest.main()
