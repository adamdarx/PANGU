"""Reconstruct a static V=0, exponential-coupling EMS exterior on an isotropic slice.

The horizon radius and scalar value are boundary data, not predicted endpoints.
This is a test fixture, separate from both the PG evolution and the pulse fixture.
It solves the same action and uses chi=psi^-4 throughout.

For ds^2=-exp(-2 delta) N dT^2+dR^2/N+R^2 dOmega^2, integrate
d(log rho)/dR=1/(R sqrt(N)). The static slice has Kij=P=0. Reflecting
rho across rho_h gives the second, time-symmetric exterior. inject() chooses a
positive precollapsed lapse; subsequent moving-puncture evolution changes the
coordinates, so coordinate profiles need not remain stationary.
"""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class StaticProfile:
    horizon_radius: float
    horizon_phi: float
    # Carrier-deck seed parameter required by ems_puncture_fixture.inject().
    # The physical ADM mass is computed below as adm_mass, not forced to mass.
    mass: float = 1.
    charge: float = .9
    alpha: float = 5.
    outer_radius: float = 4096.
    epsilon_factor: float = 1.e-8

    def __post_init__(self):
        rh, ph, Q, a = self.horizon_radius, self.horizon_phi, self.charge, self.alpha
        assert np.isfinite([rh, ph, Q, a, self.outer_radius, self.epsilon_factor]).all()
        assert rh > 0 and self.outer_radius > rh and 0 < self.epsilon_factor < 1.e-3
        self.Nprime_h = 1/rh-Q*Q*np.exp(-a*ph*ph)/rh**3
        assert self.Nprime_h > 0, 'A nondegenerate outer horizon is required'
        self.phiprime_h = -Q*Q*a*ph*np.exp(-a*ph*ph)/(rh**4*self.Nprime_h)
        self.eps = self.epsilon_factor*max(1., rh)

        def rhs(r, y):
            N, p, x, delta = y
            inverse_f = np.exp(-a*p*p)
            dN = (1-N)/r-r*N*x*x-Q*Q*inverse_f/r**3
            ddelta = -r*x*x
            dx = -Q*Q*a*p*inverse_f/(r**4*N)-(2/r+dN/N-ddelta)*x
            return [dN, x, dx, ddelta]

        self.static = solve_ivp(rhs, [rh+self.eps, self.outer_radius],
            [self.Nprime_h*self.eps, ph+self.phiprime_h*self.eps, self.phiprime_h, 0.],
            method='DOP853', rtol=2.e-13, atol=2.e-15, dense_output=True, max_step=1.)
        assert self.static.success, self.static.message
        far = self.outer_radius/np.array([2., 1.6, 1.3, 1.])
        far_state = self.static.sol(far)
        m = far*(1-far_state[0])/2
        fit = np.polynomial.polynomial.polyfit(self.outer_radius/far, m, 2)
        self.adm_mass = float(fit[0])
        self.tail_charge_squared = float(-2*fit[1]*self.outer_radius)
        self.phi_infinity_estimate = float(far_state[1, -1]+self.outer_radius*far_state[2, -1])

        # R-rh=u^2 regularizes the coordinate integral at the throat.
        def du_ds(s, y):
            u = float(y[0])
            r = rh+u*u
            if u*u <= self.eps:
                return [.5*rh*np.sqrt(self.Nprime_h)]
            N = (float(self.static.sol(r)[0]) if r <= self.outer_radius else
                 1-2*self.adm_mass/r+self.tail_charge_squared/r**2)
            assert N > 0, (r, N)
            return [.5*r*np.sqrt(N)/u]

        def end(s, y):
            return rh+y[0]*y[0]-self.outer_radius

        end.terminal = True
        end.direction = 1
        self.radial = solve_ivp(du_ds, [0., 30.], [0.], events=end, method='DOP853',
            rtol=2.e-13, atol=2.e-15, dense_output=True, max_step=.05)
        assert self.radial.success and len(self.radial.t_events[0]) == 1
        self.smax = float(self.radial.t_events[0][0])
        R = self.outer_radius
        rho_outer = .5*(R-self.adm_mass+
                       np.sqrt(R*R-2*self.adm_mass*R+self.tail_charge_squared))
        self.isotropic_horizon = float(rho_outer*np.exp(-self.smax))
        self.isotropic_outer = float(rho_outer)

    def areal(self, rho):
        rho = np.asarray(rho, dtype=float)
        assert np.isfinite(rho).all() and np.all(rho > 0)
        outside = np.maximum(rho, self.isotropic_horizon**2/rho)
        s = np.log(outside/self.isotropic_horizon)
        assert np.all(s <= self.smax+1.e-12), 'Radius exceeds the qualified static domain'
        u = self.radial.sol(s.reshape(-1))[0].reshape(rho.shape)
        return self.horizon_radius+u*u

    def evaluate(self, rho):
        rho = np.asarray(rho, dtype=float)
        R = self.areal(rho)
        p = self.static.sol(np.maximum(R, self.horizon_radius+self.eps).reshape(-1))[1].reshape(R.shape)
        near = R-self.horizon_radius < self.eps
        p = np.where(near, self.horizon_phi+self.phiprime_h*(R-self.horizon_radius), p)
        # inject() expects F=rho*psi, distinct from the Maxwell coupling f(phi).
        return np.sqrt(rho*R), p


def constraint_samples(profile):
    """Independent radial FD4 Hamiltonian check outside the horizon."""
    rows = []
    for n in [64, 96, 128]:
        h = 3.5/n
        rho = .5+np.arange(-2, n+3)*h
        f, p = profile.evaluate(rho)
        dd = (-f[4:]+16*f[3:-1]-30*f[2:-2]+16*f[1:-3]-f[:-4])/(12*h*h)
        dp = (-p[4:]+8*p[3:-1]-8*p[1:-3]+p[:-4])/(12*h)
        residual = (dd+.25*f[2:-2]*dp*dp+
                    profile.charge**2*np.exp(-profile.alpha*p[2:-2]**2)/(4*f[2:-2]**3))
        H = -8*rho[2:-2]**4/f[2:-2]**5*residual
        weight = rho[2:-2]**2
        rows.append(dict(n=n, h=h, H_RMS=float(np.sqrt(np.sum(weight*H*H)/np.sum(weight))),
                         H_max=float(np.max(np.abs(H)))))
    orders = [float(np.log(rows[i]['H_RMS']/rows[i+1]['H_RMS'])/
                    np.log(rows[i+1]['n']/rows[i]['n'])) for i in range(2)]
    return dict(samples=rows, orders=orders,
                scope='Independent radial FD4 operator; native Cartesian initialization/evolution are separate checks')
