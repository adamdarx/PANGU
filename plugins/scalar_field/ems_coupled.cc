#include "ems_coupled.h"

#include "driver/contributor.h"
#include "geometry_assembly.h"
#include "interface/update.hpp"
#include "pangu/plugin_api/scalar_field.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace pangu::plugins::scalar_field {
namespace {
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;
namespace api = pangu::plugin_api::scalar_field;
constexpr const char *label = "plugin.scalar_field";

// Action: (R - 2(d phi)^2 - 4V(phi) - f(phi) F^2)/(16 pi).
// PG line element: -a^2 dt^2 + (dr + a z dt)^2 + r^2 dOmega^2.
// State: phi, P=a^-1 phi_t-z phi_r, m=r z^2/2. Maxwell's radial
// Gauss equation gives E=Q/(r^2 f) exactly, including its time dependence.
struct Model {
  Real mass = 1, charge = 0.9, amplitude = 1.e-3, center = 10, width = 3;
  Real cfl = 0.3, dissipation = 0.05, diagnostic_interval = 1;
  bool gaussian = true, axion = false;
  bool constrained_metric = true;
  int constraint_iterations = 12;
  Real constraint_tolerance = 1.e-11;
  api::PolynomialPotential polynomial;
  api::AxionPotential axion_potential;
  api::MaxwellCoupling electromagnetic;
  KOKKOS_INLINE_FUNCTION Real V(Real phi) const {
    return axion ? axion_potential.Value(phi) : polynomial.Value(phi * phi);
  }
  KOKKOS_INLINE_FUNCTION Real DV(Real phi) const {
    return axion ? axion_potential.Gradient(phi, phi * phi)
                 : polynomial.Gradient(phi, phi * phi);
  }
  KOKKOS_INLINE_FUNCTION Real Phi0(Real r) const {
    return gaussian ? amplitude * exp(-pow((r - center) / width, 2)) : 0;
  }
  KOKKOS_INLINE_FUNCTION Real DPhi0(Real r) const {
    return -2 * (r - center) / (width * width) * Phi0(r);
  }
  KOKKOS_INLINE_FUNCTION Real MassSource(Real r, Real phi, Real P,
                                        Real dphi, Real m) const {
    const Real z = sqrt(2 * m / r);
    return r * r * (0.5 * (P * P + dphi * dphi) + z * P * dphi + V(phi)) +
           charge * charge * electromagnetic.Inverse(phi) / (2 * r * r);
  }
  KOKKOS_INLINE_FUNCTION Real InitialMassPerturbationSource(Real r,
                                                           Real dm) const {
    const Real rn = mass - charge * charge / (2 * r);
    return MassSource(r, Phi0(r), 0, DPhi0(r), rn + dm) -
           charge * charge / (2 * r * r);
  }
};

template <class Value>
KOKKOS_INLINE_FUNCTION Real D1(const Value &q, int i, int lo, int hi, Real h) {
  if (i < lo + 2) {
    if (i == lo)
      return (-25 * q(lo) + 48 * q(lo + 1) - 36 * q(lo + 2) +
              16 * q(lo + 3) - 3 * q(lo + 4)) / (12 * h);
    return (-3 * q(lo) - 10 * q(lo + 1) + 18 * q(lo + 2) -
            6 * q(lo + 3) + q(lo + 4)) / (12 * h);
  }
  if (i > hi - 2) {
    if (i == hi)
      return (25 * q(hi) - 48 * q(hi - 1) + 36 * q(hi - 2) -
              16 * q(hi - 3) + 3 * q(hi - 4)) / (12 * h);
    return (3 * q(hi) + 10 * q(hi - 1) - 18 * q(hi - 2) +
            6 * q(hi - 3) - q(hi - 4)) / (12 * h);
  }
  return (q(i - 2) - 8 * q(i - 1) + 8 * q(i + 1) - q(i + 2)) / (12 * h);
}

template <class Value>
KOKKOS_INLINE_FUNCTION Real SegmentIntegral(const Value &q, int i, int lo,
                                            int hi, Real h) {
  // Integral on [r_i,r_{i+1}] of the local cubic interpolant. This avoids
  // reducing a fourth-order PDE to second order through lapse quadrature.
  if (i == lo)
    return h * (9 * q(lo) + 19 * q(lo + 1) - 5 * q(lo + 2) + q(lo + 3)) / 24;
  if (i == hi - 1)
    return h * (9 * q(hi) + 19 * q(hi - 1) - 5 * q(hi - 2) + q(hi - 3)) / 24;
  return h * (-q(i - 1) + 13 * q(i) + 13 * q(i + 1) - q(i + 2)) / 24;
}

void Refresh(MeshData<Real> *data) {
  const auto p = data->GetParentPointer()->packages.Get(label)->Param<Model>("ems_model");
  const auto u = data->PackVariables(std::vector<std::string>{"scalar.state"});
  const auto w = data->PackVariables(std::vector<std::string>{"ems.work"});
  const auto g = data->PackVariables(std::vector<std::string>{"ems.geometry"});
  const auto e = data->PackVariables(std::vector<std::string>{"ems.electric"});
  const auto energy = data->PackVariables(std::vector<std::string>{"ems.energy"});
  const auto constraints = data->PackVariables(std::vector<std::string>{"ems.constraints"});
  const auto phi = data->PackVariables(std::vector<std::string>{"scalar.phi"});
  const auto bg = data->PackVariables(std::vector<std::string>{"scalar.background"});
  const auto ib = data->GetBoundsI(IndexDomain::interior);
  const int j = data->GetBoundsJ(IndexDomain::interior).s;
  const int k = data->GetBoundsK(IndexDomain::interior).s;
  const Real h = data->GetBlockData(0)->GetBlockPointer()->coords.Dxc<X1DIR>(k, j, ib.s);
  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "EMS scalar gradient",
      DevExecSpace(), ib.s, ib.e, KOKKOS_LAMBDA(int i) {
        const Real grad = D1([&](int ii) { return u(0, 0, k, j, ii); }, i, ib.s, ib.e, h);
        w(0, 0, k, j, i) = grad;
        w(0, 4, k, j, i) = 0;
      });
  if (p.constrained_metric) {
    // Parallel Picard integration of the nonlinear radial Einstein constraint.
    // The excision mass is advanced by its flux equation; the entire profile
    // is reconstructed at each substage. Check the final iteration change
    // independently in Timestep; insufficient convergence is a hard failure.
    for (int iteration = 0; iteration < p.constraint_iterations; ++iteration) {
      parthenon::par_for(
          parthenon::loop_pattern_flatrange_tag, "EMS radial constraint source",
          DevExecSpace(), ib.s, ib.e, KOKKOS_LAMBDA(int i) {
            const Real r = u.GetCoords(0).Xc<1>(i);
            w(0, 3, k, j, i) =
                p.MassSource(r, u(0, 0, k, j, i), u(0, 1, k, j, i),
                             w(0, 0, k, j, i), u(0, 2, k, j, i)) -
                p.charge * p.charge / (2 * r * r);
          });
      Kokkos::parallel_scan(
          "EMS radial Einstein constraint prefix",
          Kokkos::RangePolicy<DevExecSpace>(0, ib.e - ib.s + 1),
          KOKKOS_LAMBDA(int n, Real &integral, bool final) {
            const int i = ib.s + n;
            if (n > 0)
              integral += SegmentIntegral([&](int ii) { return w(0, 3, k, j, ii); },
                                          i - 1, ib.s, ib.e, h);
            if (final) {
              if (n == 0) {
                w(0, 4, k, j, i) = 0;
              } else {
                const Real first = u.GetCoords(0).Xc<1>(ib.s);
                const Real dm0 = u(0, 2, k, j, ib.s) -
                    (p.mass - p.charge * p.charge / (2 * first));
                const Real r = u.GetCoords(0).Xc<1>(i);
                const Real next = p.mass - p.charge * p.charge / (2 * r) + dm0 + integral;
                w(0, 4, k, j, i) = next - u(0, 2, k, j, i);
                u(0, 2, k, j, i) = next;
              }
            }
          });
    }
  }
  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "EMS dynamic lapse source",
      DevExecSpace(), ib.s, ib.e, KOKKOS_LAMBDA(int i) {
        const Real r = u.GetCoords(0).Xc<1>(i);
        const Real z = sqrt(2 * u(0, 2, k, j, i) / r);
        const Real grad = w(0, 0, k, j, i);
        w(0, 1, k, j, i) = -r * u(0, 1, k, j, i) * grad / z;
        g(0, 1, k, j, i) = z;
      });
  // a_r/a=-r P phi_r/z, with a(R)=1. This constraint is solved on
  // every RK substage; a is never a frozen RN background.
  Kokkos::parallel_scan(
      "EMS nonlinear lapse constraint", Kokkos::RangePolicy<DevExecSpace>(0, ib.e - ib.s + 1),
      KOKKOS_LAMBDA(int n, Real &integral, bool final) {
        const int i = ib.e - n;
        if (i < ib.e)
          integral += SegmentIntegral([&](int ii) { return w(0, 1, k, j, ii); },
                                      i, ib.s, ib.e, h);
        if (final)
          g(0, 0, k, j, i) = exp(-integral);
      });
  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "EMS dynamic geometry and Maxwell Gauss law",
      DevExecSpace(), ib.s, ib.e, KOKKOS_LAMBDA(int i) {
        const Real r = u.GetCoords(0).Xc<1>(i);
        const Real field = u(0, 0, k, j, i), P = u(0, 1, k, j, i);
        const Real m = u(0, 2, k, j, i), z = g(0, 1, k, j, i);
        const Real a = g(0, 0, k, j, i), grad = w(0, 0, k, j, i);
        const Real invf = p.electromagnetic.Inverse(field);
        const Real electric = p.charge * invf / (r * r);
        const Real scalar_rho = 0.5 * (P * P + grad * grad) + p.V(field);
        const Real electric_rho = p.charge * p.charge * invf / (2 * r * r * r * r);
        g(0, 2, k, j, i) = 1 - z * z;
        g(0, 3, k, j, i) = m;
        e(0, 0, k, j, i) = electric;
        e(0, 1, k, j, i) = r * r * electric / invf;
        energy(0, 0, k, j, i) = scalar_rho;
        energy(0, 1, k, j, i) = electric_rho;
        energy(0, 2, k, j, i) = scalar_rho + electric_rho;
        energy(0, 3, k, j, i) = -P * grad;
        phi(0, 0, k, j, i) = field;
        bg(0, 0, k, j, i) = r;
        bg(0, 1, k, j, i) = 1 - z * z;
        bg(0, 2, k, j, i) = 0; // GB invariant is not used in this EMS formulation.
        bg(0, 3, k, j, i) = 0;
        bg(0, 4, k, j, i) = -2 * electric * electric;
        w(0, 2, k, j, i) = r * r * a * (grad + z * P);
      });
  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "EMS independent radial constraint residuals",
      DevExecSpace(), ib.s, ib.e, KOKKOS_LAMBDA(int i) {
        const Real r = u.GetCoords(0).Xc<1>(i);
        const Real mr = D1([&](int ii) { return u(0, 2, k, j, ii); }, i, ib.s, ib.e, h);
        const Real ar = D1([&](int ii) { return g(0, 0, k, j, ii); }, i, ib.s, ib.e, h);
        constraints(0, 0, k, j, i) =
            mr - p.MassSource(r, u(0, 0, k, j, i), u(0, 1, k, j, i),
                              w(0, 0, k, j, i), u(0, 2, k, j, i));
        constraints(0, 1, k, j, i) = ar / g(0, 0, k, j, i) - w(0, 1, k, j, i);
        constraints(0, 2, k, j, i) =
            D1([&](int ii) { return e(0, 1, k, j, ii); }, i, ib.s, ib.e, h);
        constraints(0, 3, k, j, i) = 0;
      });
}

void Initialize(Mesh *, ParameterInput *, MeshData<Real> *data) {
  PARTHENON_REQUIRE(data->NumBlocks() == 1,
                    "spherical_ems currently requires one radial MeshBlock");
  auto *mesh = data->GetParentPointer();
  const auto p = mesh->packages.Get(label)->Param<Model>("ems_model");
  const auto u = data->PackVariables(std::vector<std::string>{"scalar.state"});
  const auto ib = data->GetBoundsI(IndexDomain::interior);
  const int j = data->GetBoundsJ(IndexDomain::interior).s;
  const int k = data->GetBoundsK(IndexDomain::interior).s;
  const Real h = data->GetBlockData(0)->GetBlockPointer()->coords.Dxc<X1DIR>(k, j, ib.s);
  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "EMS scalar initial data", DevExecSpace(),
      ib.s, ib.e, KOKKOS_LAMBDA(int i) {
        const Real r = u.GetCoords(0).Xc<1>(i);
        u(0, 0, k, j, i) = p.Phi0(r);
        u(0, 1, k, j, i) = 0;
      });
  // Integrate the NONLINEAR Hamiltonian constraint, not an RN metric with
  // scalar energy added afterwards. Subtract its analytic electrovac part
  // during quadrature so an exactly scalar-free RN solution stays exact.
  Kokkos::parallel_for(
      "EMS constraint-consistent initial geometry", Kokkos::RangePolicy<DevExecSpace>(0, 1),
      KOKKOS_LAMBDA(int) {
        Real dm = 0;
        for (int i = ib.s; i <= ib.e; ++i) {
          const Real r = u.GetCoords(0).Xc<1>(i);
          if (i > ib.s) {
            const Real x = r - h;
            const Real k1 = p.InitialMassPerturbationSource(x, dm);
            const Real k2 = p.InitialMassPerturbationSource(x + h / 2, dm + h * k1 / 2);
            const Real k3 = p.InitialMassPerturbationSource(x + h / 2, dm + h * k2 / 2);
            const Real k4 = p.InitialMassPerturbationSource(r, dm + h * k3);
            dm += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6;
          }
          u(0, 2, k, j, i) = p.mass - p.charge * p.charge / (2 * r) + dm;
        }
      });
  Refresh(data);
}

TaskStatus RHS(MeshData<Real> *data) {
  const auto p = data->GetParentPointer()->packages.Get(label)->Param<Model>("ems_model");
  const auto u = data->PackVariables(std::vector<std::string>{"scalar.state"});
  const auto rhs = data->PackVariables(std::vector<std::string>{"scalar.rhs"});
  const auto g = data->PackVariables(std::vector<std::string>{"ems.geometry"});
  const auto w = data->PackVariables(std::vector<std::string>{"ems.work"});
  const auto ib = data->GetBoundsI(IndexDomain::interior);
  const int j = data->GetBoundsJ(IndexDomain::interior).s;
  const int k = data->GetBoundsK(IndexDomain::interior).s;
  const Real h = data->GetBlockData(0)->GetBlockPointer()->coords.Dxc<X1DIR>(k, j, ib.s);
  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "EMS fully nonlinear RHS", DevExecSpace(),
      ib.s, ib.e, KOKKOS_LAMBDA(int i) {
        const Real r = u.GetCoords(0).Xc<1>(i);
        const Real field = u(0, 0, k, j, i), P = u(0, 1, k, j, i);
        const Real a = g(0, 0, k, j, i), z = g(0, 1, k, j, i);
        const Real grad = w(0, 0, k, j, i);
        const Real flux_gradient =
            D1([&](int ii) { return w(0, 2, k, j, ii); }, i, ib.s, ib.e, h);
        const Real F2_electrovac = -2 * p.charge * p.charge / (r * r * r * r);
        const Real force = p.DV(field) + p.electromagnetic.Source(field, F2_electrovac);
        rhs(0, 0, k, j, i) = a * (P + z * grad);
        rhs(0, 1, k, j, i) = flux_gradient / (r * r) - a * force;
        rhs(0, 2, k, j, i) = r * r * a * (P + z * grad) * (z * P + grad);
        if (i > ib.e - 2) {
          const Real c = a * (1 - z);
          const Real Pr = D1([&](int ii) { return u(0, 1, k, j, ii); }, i, ib.s, ib.e, h);
          rhs(0, 0, k, j, i) = -c * (grad + field / r);
          rhs(0, 1, k, j, i) = -c * (Pr + P / r);
        }
        // Sixth-derivative KO filter of the scalar variables only. Never
        // dissipate the electrovac mass profile or clip scalar amplitudes.
        if (i >= ib.s + 3 && i <= ib.e - 3)
          for (int n = 0; n < 2; ++n) {
            const Real d6 = u(0, n, k, j, i - 3) - 6 * u(0, n, k, j, i - 2) +
                            15 * u(0, n, k, j, i - 1) - 20 * u(0, n, k, j, i) +
                            15 * u(0, n, k, j, i + 1) - 6 * u(0, n, k, j, i + 2) +
                            u(0, n, k, j, i + 3);
            rhs(0, n, k, j, i) += p.dissipation * d6 / (64 * h);
          }
      });
  return TaskStatus::complete;
}

TaskStatus RKUpdate(MeshData<Real> *data, int stage, Real dt) {
  const auto u = data->PackVariables(std::vector<std::string>{"scalar.state"});
  const auto rhs = data->PackVariables(std::vector<std::string>{"scalar.rhs"});
  const auto initial = data->PackVariables(std::vector<std::string>{"scalar.initial"});
  const auto sum = data->PackVariables(std::vector<std::string>{"scalar.rk_sum"});
  const auto ib = data->GetBoundsI(IndexDomain::interior);
  const int j = data->GetBoundsJ(IndexDomain::interior).s;
  const int k = data->GetBoundsK(IndexDomain::interior).s;
  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "EMS classical RK4", DevExecSpace(),
      ib.s, ib.e, KOKKOS_LAMBDA(int i) {
        for (int n = 0; n < 3; ++n) {
          if (stage == 1) {
            initial(0, n, k, j, i) = u(0, n, k, j, i);
            sum(0, n, k, j, i) = 0;
          }
          sum(0, n, k, j, i) +=
              (stage == 1 || stage == 4 ? 1.0 / 6 : 1.0 / 3) * rhs(0, n, k, j, i);
          u(0, n, k, j, i) = initial(0, n, k, j, i) +
              dt * (stage == 4 ? sum(0, n, k, j, i)
                              : (stage < 3 ? 0.5 : 1) * rhs(0, n, k, j, i));
        }
      });
  return TaskStatus::complete;
}

Real Timestep(MeshData<Real> *data) {
  const auto p = data->GetParentPointer()->packages.Get(label)->Param<Model>("ems_model");
  const auto u = data->PackVariables(std::vector<std::string>{"scalar.state"});
  const auto g = data->PackVariables(std::vector<std::string>{"ems.geometry"});
  const auto w = data->PackVariables(std::vector<std::string>{"ems.work"});
  const auto ib = data->GetBoundsI(IndexDomain::interior);
  const int j = data->GetBoundsJ(IndexDomain::interior).s;
  const int k = data->GetBoundsK(IndexDomain::interior).s;
  const Real h = data->GetBlockData(0)->GetBlockPointer()->coords.Dxc<X1DIR>(k, j, ib.s);
  Real rate = 0;
  parthenon::par_reduce(
      parthenon::loop_pattern_flatrange_tag, "EMS causal timestep and state health",
      DevExecSpace(), ib.s, ib.e,
      KOKKOS_LAMBDA(int i, Real &local) {
        const Real a = g(0, 0, k, j, i), z = g(0, 1, k, j, i);
        const Real r = u.GetCoords(0).Xc<1>(i), phi = u(0, 0, k, j, i);
        const Real rho2 = phi * phi;
        const Real invf = p.electromagnetic.Inverse(phi);
        const Real freq2 = p.polynomial.mass_squared + p.axion_potential.mass_squared +
            6 * fabs(p.polynomial.quartic) * rho2 +
            5 * fabs(p.polynomial.sextic) * rho2 * rho2 +
            2 * fabs(p.electromagnetic.alpha) * p.charge * p.charge * invf *
                (1 + 2 * fabs(p.electromagnetic.alpha) * rho2) / (r * r * r * r);
        const bool healthy = isfinite(a) && a > 0 && isfinite(z) && z > 0 &&
            isfinite(phi) && isfinite(u(0, 1, k, j, i)) &&
            isfinite(freq2) && isfinite(invf) && invf > 0 &&
            (i != ib.s || z > 1) &&
            fabs(w(0, 4, k, j, i)) <=
                p.constraint_tolerance * (1 + fabs(u(0, 2, k, j, i)));
        local = healthy ? fmax(local, a * ((1 + z) / h + sqrt(freq2))) : INFINITY;
      }, Kokkos::Max<Real>(rate));
  PARTHENON_REQUIRE(std::isfinite(rate) && rate > 0,
                    "EMS state invalid, inner boundary untrapped, or radial constraint iteration unconverged");
  return p.cfl / rate;
}

TaskStatus EvolutionResidual(MeshData<Real> *data, Real dt) {
  const auto u = data->PackVariables(std::vector<std::string>{"scalar.state"});
  const auto initial = data->PackVariables(std::vector<std::string>{"scalar.initial"});
  const auto sum = data->PackVariables(std::vector<std::string>{"scalar.rk_sum"});
  const auto constraints = data->PackVariables(std::vector<std::string>{"ems.constraints"});
  const auto ib = data->GetBoundsI(IndexDomain::interior);
  const int j = data->GetBoundsJ(IndexDomain::interior).s;
  const int k = data->GetBoundsK(IndexDomain::interior).s;
  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "EMS independent temporal Einstein residual",
      DevExecSpace(), ib.s, ib.e, KOKKOS_LAMBDA(int i) {
        constraints(0, 3, k, j, i) =
            (u(0, 2, k, j, i) - initial(0, 2, k, j, i)) / dt - sum(0, 2, k, j, i);
      });
  return TaskStatus::complete;
}

void Diagnostics(Mesh *mesh, Real time, bool force = false) {
  auto package = mesh->packages.Get(label);
  const auto p = package->Param<Model>("ems_model");
  if (!force && time - package->Param<Real>("ems_last_diagnostic") <
                    p.diagnostic_interval * (1 - 1.e-10))
    return;
  auto &data = mesh->mesh_data.Get();
  const auto u = data->PackVariables(std::vector<std::string>{"scalar.state"});
  const auto g = data->PackVariables(std::vector<std::string>{"ems.geometry"});
  const auto c = data->PackVariables(std::vector<std::string>{"ems.constraints"});
  const auto e = data->PackVariables(std::vector<std::string>{"ems.electric"});
  const auto w = data->PackVariables(std::vector<std::string>{"ems.work"});
  const auto ib = data->GetBoundsI(IndexDomain::interior);
  const int j = data->GetBoundsJ(IndexDomain::interior).s;
  const int k = data->GetBoundsK(IndexDomain::interior).s;
  Kokkos::View<Real **> snapshot("EMS diagnostics snapshot", ib.e - ib.s + 1, 11);
  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "EMS diagnostics copy", DevExecSpace(),
      ib.s, ib.e, KOKKOS_LAMBDA(int i) {
        const int n = i - ib.s;
        snapshot(n, 0) = u.GetCoords(0).Xc<1>(i);
        snapshot(n, 1) = u(0, 0, k, j, i);
        snapshot(n, 2) = u(0, 2, k, j, i);
        snapshot(n, 3) = g(0, 0, k, j, i);
        snapshot(n, 4) = g(0, 2, k, j, i);
        snapshot(n, 5) = c(0, 0, k, j, i);
        snapshot(n, 6) = c(0, 1, k, j, i);
        snapshot(n, 7) = e(0, 1, k, j, i) - p.charge;
        const Real z = g(0, 1, k, j, i), P = u(0, 1, k, j, i);
        const Real grad = w(0, 0, k, j, i), r = snapshot(n, 0);
        snapshot(n, 8) = r * r * g(0, 0, k, j, i) *
            (P + z * grad) * (z * P + grad);
        snapshot(n, 9) = g(0, 1, k, j, i);
        snapshot(n, 10) = c(0, 3, k, j, i);
      });
  const auto s = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), snapshot);
  const int n = static_cast<int>(s.extent(0));
  Real rh = 0, phih = 0, amin = 1, maxmass = 0, maxlapse = 0, charge_error = 0;
  for (int i = 0; i < n; ++i) {
    PARTHENON_REQUIRE(std::isfinite(s(i, 1)) && std::isfinite(s(i, 2)) &&
                          std::isfinite(s(i, 3)) && s(i, 3) > 0,
                      "nonfinite EMS diagnostics");
    amin = std::min(amin, s(i, 3));
    charge_error = std::max(charge_error, fabs(s(i, 7)));
    if (i + 1 < n && s(i, 4) <= 0 && s(i + 1, 4) > 0) {
      const int start = std::max(0, std::min(n - 4, i - 1));
      const auto interpolate = [&](Real r, int component) {
        Real value = 0;
        for (int a = start; a < start + 4; ++a) {
          Real weight = 1;
          for (int b = start; b < start + 4; ++b)
            if (a != b)
              weight *= (r - s(b, 0)) / (s(a, 0) - s(b, 0));
          value += weight * s(a, component);
        }
        return value;
      };
      Real left = s(i, 0), right = s(i + 1, 0);
      for (int step = 0; step < 40; ++step) {
        const Real middle = (left + right) / 2;
        if (interpolate(middle, 4) <= 0)
          left = middle;
        else
          right = middle;
      }
      rh = (left + right) / 2;
      phih = interpolate(rh, 1);
    }
  }
  PARTHENON_REQUIRE(rh > 0, "EMS apparent horizon is outside the radial domain");
  Real constraint2 = 0, time_constraint = 0;
  int samples = 0;
  for (int i = 3; i + 3 < n; ++i)
    if (s(i, 0) > rh && s(i, 0) < 50) {
      maxmass = std::max(maxmass, fabs(s(i, 5)));
      maxlapse = std::max(maxlapse, fabs(s(i, 6)));
      constraint2 += s(i, 5) * s(i, 5);
      time_constraint = std::max(time_constraint, fabs(s(i, 10)));
      ++samples;
    }
  const std::string path = package->Param<std::string>("ems_history_file");
  std::ofstream out(path, std::ios::app);
  PARTHENON_REQUIRE(out.good(), "cannot open EMS diagnostics history");
  out << std::setprecision(17) << time << ',' << rh << ',' << phih << ',' << rh / 2
      << ',' << amin << ',' << s(n - 1, 2) << ','
      << s(n - 1, 2) + p.charge * p.charge / (2 * s(n - 1, 0)) << ','
      << -s(n - 1, 8) << ',' << maxmass << ','
      << sqrt(constraint2 / std::max(1, samples)) << ',' << maxlapse << ','
      << charge_error << ',' << time_constraint << '\n';
  out.flush();
  PARTHENON_REQUIRE(out.good(), "cannot write EMS diagnostics history");
  package->UpdateParam<Real>("ems_last_diagnostic", time);
}

void BeforeLoop(Mesh *mesh, ParameterInput *, parthenon::SimTime &tm) {
  const auto path = mesh->packages.Get(label)->Param<std::string>("ems_history_file");
  // Retain only data preceding the restart time, so replaying an old
  // checkpoint in an existing run directory never appends time backwards.
  std::ifstream old(path);
  std::vector<std::string> retained;
  std::string line;
  if (old.good()) {
    std::getline(old, line); // header
    while (std::getline(old, line)) {
      std::istringstream in(line);
      Real t;
      if (in >> t && t < tm.time - 1.e-12)
        retained.push_back(line);
    }
  }
  old.close();
  std::ofstream out(path);
  out << "time,horizon_radius,horizon_phi,irreducible_mass,min_lapse,outer_mass,"
         "adm_mass_estimate,outgoing_luminosity,mass_constraint_max,"
         "mass_constraint_rms,lapse_constraint_max,charge_error_max,"
         "einstein_time_constraint_max\n";
  for (const auto &row : retained)
    out << row << '\n';
  out.close();
  Diagnostics(mesh, tm.time, true);
}

void PostStep(Mesh *mesh, const parthenon::SimTime &tm) {
  const Real time = tm.time + tm.dt;
  Diagnostics(mesh, time, time >= tm.tlim - 1.e-12 * std::max(1.0, tm.tlim));
}

void Configure(Mesh *mesh, parthenon::LowStorageIntegrator &rk) {
  PARTHENON_REQUIRE(mesh->ndim == 1 && mesh->nbtotal == 1 &&
                        parthenon::Globals::nranks == 1 && !mesh->multilevel,
                    "spherical_ems requires a uniform 1D mesh, one MeshBlock and one rank");
  PARTHENON_REQUIRE(rk.GetName() == "rk4", "spherical_ems requires classical rk4");
  PARTHENON_REQUIRE(mesh->mesh_bc_names[0] == "outflow" &&
                        mesh->mesh_bc_names[1] == "outflow",
                    "spherical_ems uses excision and outer radiative boundaries");
  rk.nstages = 4;
  rk.nbuffers = 2;
  rk.gam0 = {0, 0, 0, 0};
  rk.gam1 = {1, 1, 1, 1};
  rk.beta = {0.5, 0.5, 1, 1.0 / 6};
  rk.delta = {0, 0, 0, 0};
  rk.c = {0, 0.5, 0.5, 1};
  rk.buffer_name = {"base", "base", "base"};
  rk.stage_name = {"base", "base", "base", "base", "base"};
}

TaskCollection Build(driver::StageBuildContext &context) {
  auto *mesh = context.mesh;
  const auto partitions = mesh->GetDefaultBlockPartitions();
  TaskCollection collection;
  auto &tasks = collection.AddRegion(1)[0];
  auto &data = mesh->mesh_data.Add("base", partitions.front());
  const auto rhs = tasks.AddTask(TaskID(0), RHS, data.get());
  const auto update = tasks.AddTask(rhs, RKUpdate, data.get(), context.stage, context.integrator->dt);
  const auto refresh = tasks.AddTask(update, parthenon::Update::FillDerived<MeshData<Real>>, data.get());
  if (context.stage == 4) {
    const auto checked = tasks.AddTask(refresh, EvolutionResidual, data.get(), context.integrator->dt);
    tasks.AddTask(checked, parthenon::Update::EstimateTimestep<MeshData<Real>>, data.get());
  }
  return collection;
}
} // namespace

std::shared_ptr<parthenon::StateDescriptor>
InitializeCoupledEMS(plugin::InitContext &context) {
  auto &pin = context.input;
  const std::string block = "plugin/scalar_field";
  const std::set<std::string> keys{
      "backend", "potential", "fields", "ell", "bh_mass", "bh_charge", "mass",
      "quartic", "sextic", "decay_constant", "gb_eta", "em_alpha", "em_linearized",
      "amplitude", "center", "width", "cfl", "initial_data", "em_coupling",
      "dissipation", "diagnostic_interval", "history_file", "metric_evolution",
      "constraint_iterations", "constraint_tolerance"};
  for (const auto &key : pin.GetParameterNames(block))
    PARTHENON_REQUIRE(keys.count(key) != 0, "unknown spherical_ems parameter: " + key);
  PARTHENON_REQUIRE(pin.GetString("parthenon/job", "problem_id") == "scalar_field",
                    "spherical_ems requires problem_id=scalar_field");
  PARTHENON_REQUIRE(!geometry::ConfiguredGeometryIsSynchronized(),
                    "PG EMS uses its own dynamic geometry; build METRIC=cks MODE=dynamic");
  PARTHENON_REQUIRE(pin.GetOrAddInteger(block, "fields", 1) == 1 &&
                        pin.GetOrAddInteger(block, "ell", 0) == 0 &&
                        pin.GetOrAddReal(block, "gb_eta", 0) == 0 &&
                        !pin.GetOrAddBoolean(block, "em_linearized", false),
                    "full spherical EMS requires one real ell=0 field, gb_eta=0, em_linearized=false");
  Model p;
  p.mass = pin.GetOrAddReal(block, "bh_mass", 1);
  p.charge = pin.GetOrAddReal(block, "bh_charge", 0.9);
  p.amplitude = pin.GetOrAddReal(block, "amplitude", 1.e-3);
  p.center = pin.GetOrAddReal(block, "center", 10);
  p.width = pin.GetOrAddReal(block, "width", 3);
  p.cfl = pin.GetOrAddReal(block, "cfl", 0.3);
  p.dissipation = pin.GetOrAddReal(block, "dissipation", 0.05);
  p.diagnostic_interval = pin.GetOrAddReal(block, "diagnostic_interval", 1);
  const auto metric_evolution = pin.GetOrAddString(block, "metric_evolution", "constrained");
  PARTHENON_REQUIRE(metric_evolution == "constrained" || metric_evolution == "free",
                    "metric_evolution must be constrained or free");
  p.constrained_metric = metric_evolution == "constrained";
  p.constraint_iterations = pin.GetOrAddInteger(block, "constraint_iterations", 12);
  p.constraint_tolerance = pin.GetOrAddReal(block, "constraint_tolerance", 1.e-11);
  PARTHENON_REQUIRE(p.constraint_iterations >= 1 && p.constraint_iterations <= 64 &&
                        std::isfinite(p.constraint_tolerance) && p.constraint_tolerance > 0,
                    "invalid radial constraint iteration controls");
  const Real mass = pin.GetOrAddReal(block, "mass", 0);
  const Real quartic = pin.GetOrAddReal(block, "quartic", 0);
  const Real sextic = pin.GetOrAddReal(block, "sextic", 0);
  const Real fa = pin.GetOrAddReal(block, "decay_constant", 1);
  const auto potential = pin.GetOrAddString(block, "potential", "polynomial");
  PARTHENON_REQUIRE(potential == "polynomial" || potential == "axion", "unknown EMS potential");
  p.axion = potential == "axion";
  p.polynomial = {p.axion ? 0 : mass * mass, quartic, sextic};
  p.axion_potential = {p.axion ? mass * mass : 0, fa};
  PARTHENON_REQUIRE(!p.axion || (quartic == 0 && sextic == 0), "axion has no polynomial coefficients");
  const auto coupling = pin.GetOrAddString(block, "em_coupling", "exponential");
  PARTHENON_REQUIRE(coupling == "exponential" || coupling == "quadratic", "unknown EMS coupling");
  p.electromagnetic = {pin.GetOrAddReal(block, "em_alpha", 5), false,
                      coupling == "exponential" ? 0 : 1};
  const auto initial = pin.GetOrAddString(block, "initial_data", "gaussian");
  PARTHENON_REQUIRE(initial == "gaussian" || initial == "zero", "spherical_ems initial_data is gaussian or zero");
  p.gaussian = initial == "gaussian";
  for (Real value : {p.mass, p.charge, p.amplitude, p.center, p.width, p.cfl,
                     p.dissipation, p.diagnostic_interval, mass, quartic, sextic,
                     fa, p.electromagnetic.alpha})
    PARTHENON_REQUIRE(std::isfinite(value), "nonfinite EMS parameter");
  PARTHENON_REQUIRE(p.mass > 0 && fabs(p.charge) < p.mass && p.width > 0 &&
                        p.cfl > 0 && p.cfl <= 0.5 && p.dissipation >= 0 &&
                        p.dissipation <= 1 && p.diagnostic_interval > 0 &&
                        mass >= 0 && quartic >= 0 && sextic >= 0 && fa > 0 &&
                        p.electromagnetic.alpha >= 0,
                    "invalid full EMS parameters; this version requires nonnegative potentials and coupling");
  const Real rmin = pin.GetReal("parthenon/mesh", "x1min");
  const Real rmax = pin.GetReal("parthenon/mesh", "x1max");
  const Real rp = p.mass + sqrt(p.mass * p.mass - p.charge * p.charge);
  PARTHENON_REQUIRE(rmin > p.charge * p.charge / (2 * p.mass) &&
                        rmin < rp && rmax > rp &&
                        pin.GetInteger("parthenon/mesh", "nx1") >= 16 &&
                        pin.GetInteger("parthenon/meshblock", "nx1") ==
                            pin.GetInteger("parthenon/mesh", "nx1"),
                    "EMS radial mesh must cover an excised RN horizon and use one block nx1>=16");
  auto package = std::make_shared<StateDescriptor>(label);
  package->AddParam<Model>("ems_model", p);
  package->AddParam<Real>("ems_last_diagnostic", -std::numeric_limits<Real>::max(), true);
  package->AddParam<std::string>("ems_history_file",
      pin.GetOrAddString(block, "history_file", "ems_history.csv"));
  package->AddParam<driver::StageContribution>(std::string(driver::stage_contribution_key),
      {"spherical_ems", 200, [](const driver::StageBuildContext &) { return true; },
       Build, Configure, PostStep});
  const Metadata state({Metadata::Cell, Metadata::Independent, Metadata::Restart},
                       std::vector<int>{3});
  const auto derived = [](int n) {
    return Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, std::vector<int>{n});
  };
  package->AddField("scalar.state", state);
  package->AddField("scalar.rhs", derived(3));
  package->AddField("scalar.initial", derived(3));
  package->AddField("scalar.rk_sum", derived(3));
  package->AddField("scalar.phi", derived(1));
  package->AddField("scalar.background", derived(5));
  package->AddField("ems.work", derived(5));
  package->AddField("ems.geometry", derived(4));
  package->AddField("ems.electric", derived(2));
  package->AddField("ems.energy", derived(4));
  package->AddField("ems.constraints", derived(4));
  package->PostProblemGeneratorMesh = Initialize;
  package->FillDerivedMesh = Refresh;
  package->EstimateTimestepMesh = Timestep;
  package->UserWorkBeforeLoopMesh = BeforeLoop;
  return package;
}
} // namespace pangu::plugins::scalar_field
