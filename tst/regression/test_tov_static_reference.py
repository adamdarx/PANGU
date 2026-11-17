"""Physical oracles independent of the numerical static-star integrator."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

path = Path(__file__).resolve().parents[1] / 'reference/tov_static_reference.py'
spec = importlib.util.spec_from_file_location('static_tov', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class StaticTOVOracles(unittest.TestCase):
    def test_newtonian_n1_lane_emden_limit(self):
        kappa, rhoc = 100., 1e-10
        result = module.solve_tov(rhoc, kappa, rtol=1e-13)
        radius = np.sqrt(np.pi*kappa/2)
        mass = np.sqrt(2*np.pi)*kappa**1.5*rhoc
        self.assertLess(abs(result['areal_surface_radius']/radius-1), 1e-5)
        self.assertLess(abs(result['ADM_mass']/mass-1), 1e-5)

    def test_relativistic_polytropic_scale_invariance(self):
        original = module.solve_tov()
        rescaled = module.solve_tov(rhoc=1.28e-3/4, kappa=400.)
        for key in ['ADM_mass', 'baryon_mass', 'areal_surface_radius', 'isotropic_surface_radius']:
            self.assertLess(abs(rescaled[key]/original[key]/2-1), 1e-8)
        self.assertLess(abs(rescaled['central_lapse']/original['central_lapse']-1), 1e-8)


if __name__ == '__main__':
    unittest.main()
