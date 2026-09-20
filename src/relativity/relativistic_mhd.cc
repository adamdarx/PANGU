#include "relativity/relativistic_mhd.h"

// SRMHD stress-energy and Kastaun C2P expressions track AthenaK's BSD-3-Clause
// ideal_srmhd and HLLE implementation while retaining PANGU's face-centered CT.

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#include "estimator/estimator.h"
#include "geometry_assembly.h"
#include "hydro/hydro_types.h"
#include "mhd/mhd_package.h"
#include "mhd/mhd_types.h"
#include "mhd/passive_transport.h"
#include "pangu.h"
#include "reconstruct/hydro_reconstruction.h"
#include "riemann/registry.h"

namespace pangu::relativity {
using namespace parthenon::package::prelude;
using Estimator = estimator::Selected;

namespace {

// First-order flux correction writes its passive-transport channel with the LLF formula.
static_assert(riemann::first_order_fallback == riemann::Solver::llf,
              "relativistic MHD FOFC passive transport implements the LLF fallback");

constexpr int E2X1 = 0;
constexpr int E3X1 = 1;
constexpr int E1X2 = 2;
constexpr int E3X2 = 3;
constexpr int E1X3 = 4;
constexpr int E2X3 = 5;

struct RMHDFluxState {
  Real conserved[5]{};
  Real magnetic[3]{};
  Real flux[5]{};
  Real induction[3]{};
  Real lambda_plus = 0.0;
  Real lambda_minus = 0.0;
};

KOKKOS_INLINE_FUNCTION mhd::PassiveTransportCoefficients
SolveLLFPassiveTransportFromStates(const RMHDFluxState& left, const RMHDFluxState& right) {
  const Real speed = fmax(fmax(fabs(left.lambda_minus), fabs(left.lambda_plus)),
                          fmax(fabs(right.lambda_minus), fabs(right.lambda_plus)));
  return mhd::LLFPassiveTransport(left.conserved[mhd::IDN], right.conserved[mhd::IDN],
                                  left.flux[mhd::IDN], right.flux[mhd::IDN], speed);
}

KOKKOS_INLINE_FUNCTION mhd::PassiveTransportCoefficients
SolveHLLEPassiveTransportFromStates(const RMHDFluxState& left, const RMHDFluxState& right) {
  const Real lambda_left = fmin(left.lambda_minus, right.lambda_minus);
  const Real lambda_right = fmax(left.lambda_plus, right.lambda_plus);
  return mhd::HLLEPassiveTransport(left.conserved[mhd::IDN], right.conserved[mhd::IDN],
                                   left.flux[mhd::IDN], right.flux[mhd::IDN], lambda_left,
                                   lambda_right);
}

// Keep type-dependent transport access outside device lambdas.  When passive
// transport is disabled Parthenon supplies a VariablePack, which deliberately
// has no flux() member; tag dispatch ensures that invalid expression is never
// instantiated without compile-time branching inside the device lambda.
template <typename TransportPack>
KOKKOS_INLINE_FUNCTION void
WritePassiveFluxBlock(const TransportPack& transport, const int direction, const int k, const int j,
                      const int i, const mhd::PassiveTransportCoefficients& passive,
                      std::true_type) {
  transport.flux(direction + 1, 0, k, j, i) = passive.left;
  transport.flux(direction + 1, 1, k, j, i) = passive.right;
}

template <typename TransportPack>
KOKKOS_INLINE_FUNCTION void
WritePassiveFluxBlock(const TransportPack&, const int, const int, const int, const int,
                      const mhd::PassiveTransportCoefficients&, std::false_type) {}

template <typename TransportPack>
KOKKOS_INLINE_FUNCTION void
WritePassiveFluxMesh(const TransportPack& transport, const int block, const int direction,
                     const int k, const int j, const int i,
                     const mhd::PassiveTransportCoefficients& passive, std::true_type) {
  auto block_transport = transport(block);
  block_transport.flux(direction + 1, 0, k, j, i) = passive.left;
  block_transport.flux(direction + 1, 1, k, j, i) = passive.right;
}

template <typename TransportPack>
KOKKOS_INLINE_FUNCTION void
WritePassiveFluxMesh(const TransportPack&, const int, const int, const int, const int, const int,
                     const mhd::PassiveTransportCoefficients&, std::false_type) {}

// nvcc rejects first use of a captured VariablePack from a constexpr-if body
// inside an extended device lambda.  Tag dispatch keeps the estimator decision
// at compile time while making the lambda itself a plain function call.
template <typename SignalPack>
KOKKOS_INLINE_FUNCTION void StoreSignalSpeedsBlock(const SignalPack& signal_max,
                                                   const SignalPack& signal_min,
                                                   const int direction, const int k, const int j,
                                                   const int i, const Real lambda_plus,
                                                   const Real lambda_minus, std::true_type) {
  signal_max(direction, k, j, i) = fmax(0.0, lambda_plus);
  signal_min(direction, k, j, i) = fmax(0.0, -lambda_minus);
}

template <typename SignalPack>
KOKKOS_INLINE_FUNCTION void StoreSignalSpeedsBlock(const SignalPack&, const SignalPack&, const int,
                                                   const int, const int, const int, const Real,
                                                   const Real, std::false_type) {}

template <typename SignalPack>
KOKKOS_INLINE_FUNCTION void StoreSignalSpeedsMesh(const SignalPack& signal_max,
                                                  const SignalPack& signal_min, const int block,
                                                  const int direction, const int k, const int j,
                                                  const int i, const Real lambda_plus,
                                                  const Real lambda_minus, std::true_type) {
  signal_max(block, direction, k, j, i) = fmax(0.0, lambda_plus);
  signal_min(block, direction, k, j, i) = fmax(0.0, -lambda_minus);
}

template <typename SignalPack>
KOKKOS_INLINE_FUNCTION void StoreSignalSpeedsMesh(const SignalPack&, const SignalPack&, const int,
                                                  const int, const int, const int, const int,
                                                  const Real, const Real, std::false_type) {}

template <int Direction, typename Pack>
KOKKOS_INLINE_FUNCTION Real ReadBlock(const Pack& pack, const int component, const int k,
                                      const int j, const int i, const int offset) {
  if constexpr (Direction == 0)
    return pack(component, k, j, i + offset);
  if constexpr (Direction == 1)
    return pack(component, k, j + offset, i);
  return pack(component, k + offset, j, i);
}

template <int Direction, typename Pack>
KOKKOS_INLINE_FUNCTION Real ReadMesh(const Pack& pack, const int block, const int component,
                                     const int k, const int j, const int i, const int offset) {
  if constexpr (Direction == 0)
    return pack(block, component, k, j, i + offset);
  if constexpr (Direction == 1)
    return pack(block, component, k, j + offset, i);
  return pack(block, component, k + offset, j, i);
}

template <int Direction, hydro::Reconstruction Method, typename SourcePack, typename LeftPack,
          typename RightPack>
KOKKOS_INLINE_FUNCTION void ReconstructCellBlock(const SourcePack& source, const LeftPack& left,
                                                 const RightPack& right, const int component,
                                                 const int k, const int j, const int i) {
  constexpr int di = Direction == 0 ? 1 : 0;
  constexpr int dj = Direction == 1 ? 1 : 0;
  constexpr int dk = Direction == 2 ? 1 : 0;
  using Model = reconstruct::Implementation<Method>;
  if constexpr (Model::cell_edge_states) {
    // Match AthenaK's face-indexed wl/wr buffers: cell i writes its high-side
    // edge to the left buffer at face i+1 and its low-side edge to the right
    // buffer at face i.  Persisting these values before the Riemann kernel also
    // reproduces AthenaK's floating-point materialization boundary.
    Real low_edge = 0.0, high_edge = 0.0;
    Model::CellEdges(ReadBlock<Direction>(source, component, k, j, i, -1),
                     ReadBlock<Direction>(source, component, k, j, i, 0),
                     ReadBlock<Direction>(source, component, k, j, i, 1), low_edge, high_edge);
    left(component, k + dk, j + dj, i + di) = high_edge;
    right(component, k, j, i) = low_edge;
  } else {
    Real ql = 0.0, qr = 0.0;
    reconstruct::ReconstructFaceStencil<Method>(
        [&](const int offset) { return ReadBlock<Direction>(source, component, k, j, i, offset); },
        ql, qr);
    left(component, k, j, i) = ql;
    right(component, k, j, i) = qr;
  }
}

template <int Direction, hydro::Reconstruction Method, typename SourcePack, typename LeftPack,
          typename RightPack>
KOKKOS_INLINE_FUNCTION void
ReconstructCellMesh(const SourcePack& source, const LeftPack& left, const RightPack& right,
                    const int block, const int component, const int k, const int j, const int i) {
  constexpr int di = Direction == 0 ? 1 : 0;
  constexpr int dj = Direction == 1 ? 1 : 0;
  constexpr int dk = Direction == 2 ? 1 : 0;
  using Model = reconstruct::Implementation<Method>;
  if constexpr (Model::cell_edge_states) {
    Real low_edge = 0.0, high_edge = 0.0;
    Model::CellEdges(ReadMesh<Direction>(source, block, component, k, j, i, -1),
                     ReadMesh<Direction>(source, block, component, k, j, i, 0),
                     ReadMesh<Direction>(source, block, component, k, j, i, 1), low_edge,
                     high_edge);
    left(block, component, k + dk, j + dj, i + di) = high_edge;
    right(block, component, k, j, i) = low_edge;
  } else {
    Real ql = 0.0, qr = 0.0;
    reconstruct::ReconstructFaceStencil<Method>(
        [&](const int offset) {
          return ReadMesh<Direction>(source, block, component, k, j, i, offset);
        },
        ql, qr);
    left(block, component, k, j, i) = ql;
    right(block, component, k, j, i) = qr;
  }
}

KOKKOS_INLINE_FUNCTION RMHDFluxState BuildSRMHDFluxState(const MHDPrimitiveState& primitive,
                                                         const eos::RelativisticEOS& eos,
                                                         const int direction) {
  RMHDFluxState state{};
  const auto conserved = ConvertSRMHDP2C(primitive, eos);
  state.conserved[mhd::IDN] = conserved.density;
  state.conserved[mhd::IM1] = conserved.momentum[0];
  state.conserved[mhd::IM2] = conserved.momentum[1];
  state.conserved[mhd::IM3] = conserved.momentum[2];
  state.conserved[mhd::IEN] = conserved.energy;
  const auto& fluid = primitive.fluid;
  const Real u0 =
      sqrt(1.0 + fluid.u[0] * fluid.u[0] + fluid.u[1] * fluid.u[1] + fluid.u[2] * fluid.u[2]);
  const Real b0 = primitive.magnetic[0] * fluid.u[0] + primitive.magnetic[1] * fluid.u[1] +
                  primitive.magnetic[2] * fluid.u[2];
  Real b[3];
  Real bsq = -b0 * b0;
  for (int axis = 0; axis < 3; ++axis) {
    b[axis] = (primitive.magnetic[axis] + b0 * fluid.u[axis]) / u0;
    bsq += b[axis] * b[axis];
    // Keep AthenaK's evolved-induction expression rather than replacing the
    // algebraically equal value by the input primitive B.  Its roundoff enters
    // the HLLE/LLF jump term in highly magnetized states.
    state.magnetic[axis] = b[axis] * u0 - b0 * fluid.u[axis];
  }
  const Real gamma_prime = eos.gamma / (eos.gamma - 1.0);
  const Real gas_enthalpy = fluid.density + gamma_prime * fluid.pressure;
  const Real total_enthalpy = gas_enthalpy + bsq;
  const Real total_pressure = fluid.pressure + 0.5 * bsq;
  state.flux[mhd::IDN] = fluid.density * fluid.u[direction];
  for (int axis = 0; axis < 3; ++axis)
    state.flux[mhd::IM1 + axis] = total_enthalpy * fluid.u[direction] * fluid.u[axis] -
                                  b[direction] * b[axis] +
                                  (axis == direction ? total_pressure : 0.0);
  state.flux[mhd::IEN] =
      total_enthalpy * u0 * fluid.u[direction] - b0 * b[direction] - state.flux[mhd::IDN];
  for (int axis = 0; axis < 3; ++axis)
    state.induction[axis] = b[axis] * fluid.u[direction] - b[direction] * fluid.u[axis];
  // Preserve KHARMA's vchar expression tree in the Minkowski limit.  This is
  // algebraically equivalent to the former velocity-addition form, but using
  // the same quadratic is required for an ulp-level SR estimator comparison.
  const Real sound2 =
      fmin(fmax(eos.gamma * fluid.pressure / gas_enthalpy, 0.0), eos.gamma - 1.0);
  const Real alfven2 = bsq / (bsq + gas_enthalpy);
  const Real fast2 = fmin(fmax(sound2 + alfven2 - sound2 * alfven2, 0.0), 1.0);
  const Real normal_u = fluid.u[direction];
  const Real coefficient_a = u0 * u0 - (-1.0 + u0 * u0) * fast2;
  const Real coefficient_b = 2.0 * (normal_u * u0 - normal_u * u0 * fast2);
  const Real coefficient_c = normal_u * normal_u - (1.0 + normal_u * normal_u) * fast2;
  const Real discriminant =
      sqrt(fmax(coefficient_b * coefficient_b - 4.0 * coefficient_a * coefficient_c, 0.0));
  const Real vp = -(-coefficient_b + discriminant) / (2.0 * coefficient_a);
  const Real vm = -(-coefficient_b - discriminant) / (2.0 * coefficient_a);
  state.lambda_plus = fmax(vp, vm);
  state.lambda_minus = fmin(vp, vm);
  return state;
}

KOKKOS_INLINE_FUNCTION RMHDFluxState BuildGRMHDFluxState(const MHDPrimitiveState& primitive,
                                                         const eos::RelativisticEOS& eos,
                                                         const int direction,
                                                         const geometry::MetricPoint& metric) {
  RMHDFluxState state{};
  const auto& fluid = primitive.fluid;
  // AthenaK deliberately evaluates the GR Riemann state in the cyclic
  // (normal,tangent-1,tangent-2) order.  Preserve that expression tree rather
  // than using algebraically equivalent global-axis reductions: near the
  // excision surface a few ulps in the magnetically dominated state can change
  // a following C2P floor decision.
  const int vx = direction;
  const int vy = (direction + 1) % 3;
  const int vz = (direction + 2) % 3;
  const int ix = vx + 1;
  const int iy = vy + 1;
  const int iz = vz + 1;
  const Real ux = fluid.u[vx];
  const Real uy = fluid.u[vy];
  const Real uz = fluid.u[vz];
  const Real spatial_u2 = metric.lower[ix][ix] * ux * ux + metric.lower[iy][iy] * uy * uy +
                          metric.lower[iz][iz] * uz * uz + 2.0 * metric.lower[ix][iy] * ux * uy +
                          2.0 * metric.lower[ix][iz] * ux * uz +
                          2.0 * metric.lower[iy][iz] * uy * uz;
  const Real alpha = sqrt(-1.0 / metric.upper[0][0]);
  const Real lorentz = sqrt(1.0 + spatial_u2);
  Real u_upper[4]{};
  u_upper[0] = lorentz / alpha;
  u_upper[ix] = ux - alpha * lorentz * metric.upper[0][ix];
  u_upper[iy] = uy - alpha * lorentz * metric.upper[0][iy];
  u_upper[iz] = uz - alpha * lorentz * metric.upper[0][iz];
  Real u_lower[4]{};
  u_lower[0] = metric.lower[0][0] * u_upper[0] + metric.lower[0][ix] * u_upper[ix] +
               metric.lower[0][iy] * u_upper[iy] + metric.lower[0][iz] * u_upper[iz];
  u_lower[ix] = metric.lower[ix][0] * u_upper[0] + metric.lower[ix][ix] * u_upper[ix] +
                metric.lower[ix][iy] * u_upper[iy] + metric.lower[ix][iz] * u_upper[iz];
  u_lower[iy] = metric.lower[iy][0] * u_upper[0] + metric.lower[iy][ix] * u_upper[ix] +
                metric.lower[iy][iy] * u_upper[iy] + metric.lower[iy][iz] * u_upper[iz];
  u_lower[iz] = metric.lower[iz][0] * u_upper[0] + metric.lower[iz][ix] * u_upper[ix] +
                metric.lower[iz][iy] * u_upper[iy] + metric.lower[iz][iz] * u_upper[iz];
  Real b_upper[4]{};
  b_upper[0] = u_lower[ix] * primitive.magnetic[vx] + u_lower[iy] * primitive.magnetic[vy] +
               u_lower[iz] * primitive.magnetic[vz];
  b_upper[ix] = (primitive.magnetic[vx] + b_upper[0] * u_upper[ix]) / u_upper[0];
  b_upper[iy] = (primitive.magnetic[vy] + b_upper[0] * u_upper[iy]) / u_upper[0];
  b_upper[iz] = (primitive.magnetic[vz] + b_upper[0] * u_upper[iz]) / u_upper[0];
  Real b_lower[4]{};
  b_lower[0] = metric.lower[0][0] * b_upper[0] + metric.lower[0][ix] * b_upper[ix] +
               metric.lower[0][iy] * b_upper[iy] + metric.lower[0][iz] * b_upper[iz];
  b_lower[ix] = metric.lower[ix][0] * b_upper[0] + metric.lower[ix][ix] * b_upper[ix] +
                metric.lower[ix][iy] * b_upper[iy] + metric.lower[ix][iz] * b_upper[iz];
  b_lower[iy] = metric.lower[iy][0] * b_upper[0] + metric.lower[iy][ix] * b_upper[ix] +
                metric.lower[iy][iy] * b_upper[iy] + metric.lower[iy][iz] * b_upper[iz];
  b_lower[iz] = metric.lower[iz][0] * b_upper[0] + metric.lower[iz][ix] * b_upper[ix] +
                metric.lower[iz][iy] * b_upper[iy] + metric.lower[iz][iz] * b_upper[iz];
  const Real magnetic2 = b_lower[0] * b_upper[0] + b_lower[ix] * b_upper[ix] +
                         b_lower[iy] * b_upper[iy] + b_lower[iz] * b_upper[iz];

  const Real gamma_prime = eos.gamma / (eos.gamma - 1.0);
  const Real gas_enthalpy = fluid.density + gamma_prime * fluid.pressure;
  const Real total_enthalpy = gas_enthalpy + magnetic2;
  const Real total_pressure = fluid.pressure + 0.5 * magnetic2;
  state.conserved[mhd::IDN] = fluid.density * u_upper[0];
  const Real conserved_factor = total_enthalpy * u_upper[0];
  state.conserved[mhd::IM1 + vx] = conserved_factor * u_lower[ix] - b_upper[0] * b_lower[ix];
  state.conserved[mhd::IM1 + vy] = conserved_factor * u_lower[iy] - b_upper[0] * b_lower[iy];
  state.conserved[mhd::IM1 + vz] = conserved_factor * u_lower[iz] - b_upper[0] * b_lower[iz];
  state.conserved[mhd::IEN] =
      total_enthalpy * u_upper[0] * u_lower[0] - b_upper[0] * b_lower[0] + total_pressure;
  state.flux[mhd::IDN] = fluid.density * u_upper[ix];
  const Real flux_factor = total_enthalpy * u_upper[ix];
  state.flux[mhd::IM1 + vx] =
      flux_factor * u_lower[ix] - b_upper[ix] * b_lower[ix] + total_pressure;
  state.flux[mhd::IM1 + vy] = flux_factor * u_lower[iy] - b_upper[ix] * b_lower[iy];
  state.flux[mhd::IM1 + vz] = flux_factor * u_lower[iz] - b_upper[ix] * b_lower[iz];
  state.flux[mhd::IEN] = flux_factor * u_lower[0] - b_upper[ix] * b_lower[0];
  state.magnetic[vx] = b_upper[ix] * u_upper[0] - b_upper[0] * u_upper[ix];
  state.magnetic[vy] = b_upper[iy] * u_upper[0] - b_upper[0] * u_upper[iy];
  state.magnetic[vz] = b_upper[iz] * u_upper[0] - b_upper[0] * u_upper[iz];
  state.induction[vx] = 0.0;
  state.induction[vy] = b_upper[iy] * u_upper[ix] - b_upper[ix] * u_upper[iy];
  state.induction[vz] = b_upper[iz] * u_upper[ix] - b_upper[ix] * u_upper[iz];

  const Real speed_enthalpy = eos.EnthalpyDensity(fluid.density, fluid.pressure);
  const Real sound2 = eos.SoundSpeedSquared(fluid.density, fluid.pressure);
  const Real alfven2 = magnetic2 / (magnetic2 + speed_enthalpy);
  const Real fast2 = sound2 + alfven2 - sound2 * alfven2;
  const Real a = u_upper[0] * u_upper[0] - (metric.upper[0][0] + u_upper[0] * u_upper[0]) * fast2;
  const Real b =
      -2.0 * (u_upper[0] * u_upper[ix] - (metric.upper[0][ix] + u_upper[0] * u_upper[ix]) * fast2);
  const Real c =
      u_upper[ix] * u_upper[ix] - (metric.upper[ix][ix] + u_upper[ix] * u_upper[ix]) * fast2;
  const Real a1 = b / a;
  const Real a0 = c / a;
  const Real root = sqrt(fmax(a1 * a1 - 4.0 * a0, 0.0));
  state.lambda_plus = a1 >= 0.0 ? -2.0 * a0 / (a1 + root) : (-a1 + root) / 2.0;
  state.lambda_minus = a1 >= 0.0 ? (-a1 - root) / 2.0 : -2.0 * a0 / (a1 - root);
  for (int n = 0; n < 5; ++n) {
    state.conserved[n] *= metric.gdet;
    state.flux[n] *= metric.gdet;
  }
  for (int axis = 0; axis < 3; ++axis) {
    state.magnetic[axis] *= metric.gdet;
    state.induction[axis] *= metric.gdet;
  }
  return state;
}

KOKKOS_INLINE_FUNCTION RMHDFluxState SolveRMHDHLLEFromStates(const RMHDFluxState& left,
                                                             const RMHDFluxState& right) {
  const Real lambda_left = fmin(left.lambda_minus, right.lambda_minus);
  const Real lambda_right = fmax(left.lambda_plus, right.lambda_plus);
  RMHDFluxState state{};
  const Real wave_product = lambda_right * lambda_left;
  const Real inverse = 1.0 / (lambda_right - lambda_left);
  for (int n = 0; n < 5; ++n) {
    const Real jump = right.conserved[n] - left.conserved[n];
    state.flux[n] =
        (lambda_right * left.flux[n] - lambda_left * right.flux[n] + wave_product * jump) * inverse;
  }
  for (int axis = 0; axis < 3; ++axis) {
    const Real jump = right.magnetic[axis] - left.magnetic[axis];
    state.induction[axis] = (lambda_right * left.induction[axis] -
                             lambda_left * right.induction[axis] + wave_product * jump) *
                            inverse;
  }
  if (lambda_left >= 0.0)
    return left;
  if (lambda_right <= 0.0)
    return right;
  return state;
}

KOKKOS_INLINE_FUNCTION RMHDFluxState SolveRMHDLLFFromStates(const RMHDFluxState& left,
                                                            const RMHDFluxState& right) {
  const Real speed = fmax(fmax(fabs(left.lambda_minus), fabs(left.lambda_plus)),
                          fmax(fabs(right.lambda_minus), fabs(right.lambda_plus)));
  RMHDFluxState state{};
  for (int n = 0; n < 5; ++n)
    state.flux[n] =
        0.5 * (left.flux[n] + right.flux[n] - speed * (right.conserved[n] - left.conserved[n]));
  for (int axis = 0; axis < 3; ++axis)
    state.induction[axis] = 0.5 * (left.induction[axis] + right.induction[axis] -
                                   speed * (right.magnetic[axis] - left.magnetic[axis]));
  return state;
}

KOKKOS_INLINE_FUNCTION RMHDFluxState SolveGRMHDLLFFromStates(const RMHDFluxState& left,
                                                             const RMHDFluxState& right) {
  auto state = SolveRMHDLLFFromStates(left, right);
  state.flux[mhd::IEN] += state.flux[mhd::IDN];
  return state;
}

template <int Direction, riemann::Solver Solver = riemann::Solver::hlle,
          bool WritePassiveTransport = false>
KOKKOS_INLINE_FUNCTION RMHDFluxState SolveGRMHDRiemannDirect(
    const MHDPrimitiveState& left, const MHDPrimitiveState& right,
    const eos::RelativisticEOS& eos, const geometry::MetricPoint& metric,
    mhd::PassiveTransportCoefficients* passive = nullptr) {
  // The fixed-background kernel evaluates face states and the solver in one AthenaK-matched
  // expression tree, so it implements the registered relativistic MHD solvers directly.
  static_assert(riemann::Describe(Solver).relativistic_mhd &&
                    (Solver == riemann::Solver::llf || Solver == riemann::Solver::hlle),
                "Direct GRMHD Riemann solver implements the registered LLF and HLLE solvers");
  constexpr int vx = Direction;
  constexpr int vy = (Direction + 1) % 3;
  constexpr int vz = (Direction + 2) % 3;
  constexpr int ix = vx + 1;
  constexpr int iy = vy + 1;
  constexpr int iz = vz + 1;

  const Real gm1 = eos.gamma - 1.0;
  const Real gamma_prime = eos.gamma / gm1;
  const Real wl_idn = left.fluid.density;
  const Real wl_ivx = left.fluid.u[vx];
  const Real wl_ivy = left.fluid.u[vy];
  const Real wl_ivz = left.fluid.u[vz];
  const Real wl_iby = left.magnetic[vy];
  const Real wl_ibz = left.magnetic[vz];
  const Real wl_ipr = left.fluid.pressure;
  const Real wr_idn = right.fluid.density;
  const Real wr_ivx = right.fluid.u[vx];
  const Real wr_ivy = right.fluid.u[vy];
  const Real wr_ivz = right.fluid.u[vz];
  const Real wr_iby = right.magnetic[vy];
  const Real wr_ibz = right.magnetic[vz];
  const Real wr_ipr = right.fluid.pressure;
  const Real bxi = left.magnetic[vx];

  Real q = metric.lower[ix][ix] * (wl_ivx * wl_ivx) + metric.lower[iy][iy] * (wl_ivy * wl_ivy) +
           metric.lower[iz][iz] * (wl_ivz * wl_ivz) + 2.0 * metric.lower[ix][iy] * wl_ivx * wl_ivy +
           2.0 * metric.lower[ix][iz] * wl_ivx * wl_ivz +
           2.0 * metric.lower[iy][iz] * wl_ivy * wl_ivz;
  const Real alpha = sqrt(-1.0 / metric.upper[0][0]);
  Real lorentz = sqrt(1.0 + q);
  Real uul[4];
  uul[0] = lorentz / alpha;
  uul[ix] = wl_ivx - alpha * lorentz * metric.upper[0][ix];
  uul[iy] = wl_ivy - alpha * lorentz * metric.upper[0][iy];
  uul[iz] = wl_ivz - alpha * lorentz * metric.upper[0][iz];
  Real ull[4];
  ull[0] = metric.lower[0][0] * uul[0] + metric.lower[0][ix] * uul[ix] +
           metric.lower[0][iy] * uul[iy] + metric.lower[0][iz] * uul[iz];
  ull[ix] = metric.lower[ix][0] * uul[0] + metric.lower[ix][ix] * uul[ix] +
            metric.lower[ix][iy] * uul[iy] + metric.lower[ix][iz] * uul[iz];
  ull[iy] = metric.lower[iy][0] * uul[0] + metric.lower[iy][ix] * uul[ix] +
            metric.lower[iy][iy] * uul[iy] + metric.lower[iy][iz] * uul[iz];
  ull[iz] = metric.lower[iz][0] * uul[0] + metric.lower[iz][ix] * uul[ix] +
            metric.lower[iz][iy] * uul[iy] + metric.lower[iz][iz] * uul[iz];
  Real bul[4];
  bul[0] = ull[ix] * bxi + ull[iy] * wl_iby + ull[iz] * wl_ibz;
  bul[ix] = (bxi + bul[0] * uul[ix]) / uul[0];
  bul[iy] = (wl_iby + bul[0] * uul[iy]) / uul[0];
  bul[iz] = (wl_ibz + bul[0] * uul[iz]) / uul[0];
  Real bll[4];
  bll[0] = metric.lower[0][0] * bul[0] + metric.lower[0][ix] * bul[ix] +
           metric.lower[0][iy] * bul[iy] + metric.lower[0][iz] * bul[iz];
  bll[ix] = metric.lower[ix][0] * bul[0] + metric.lower[ix][ix] * bul[ix] +
            metric.lower[ix][iy] * bul[iy] + metric.lower[ix][iz] * bul[iz];
  bll[iy] = metric.lower[iy][0] * bul[0] + metric.lower[iy][ix] * bul[ix] +
            metric.lower[iy][iy] * bul[iy] + metric.lower[iy][iz] * bul[iz];
  bll[iz] = metric.lower[iz][0] * bul[0] + metric.lower[iz][ix] * bul[ix] +
            metric.lower[iz][iy] * bul[iy] + metric.lower[iz][iz] * bul[iz];
  const Real bsq_l = bll[0] * bul[0] + bll[ix] * bul[ix] + bll[iy] * bul[iy] + bll[iz] * bul[iz];

  q = metric.lower[ix][ix] * (wr_ivx * wr_ivx) + metric.lower[iy][iy] * (wr_ivy * wr_ivy) +
      metric.lower[iz][iz] * (wr_ivz * wr_ivz) + 2.0 * metric.lower[ix][iy] * wr_ivx * wr_ivy +
      2.0 * metric.lower[ix][iz] * wr_ivx * wr_ivz + 2.0 * metric.lower[iy][iz] * wr_ivy * wr_ivz;
  lorentz = sqrt(1.0 + q);
  Real uur[4];
  uur[0] = lorentz / alpha;
  uur[ix] = wr_ivx - alpha * lorentz * metric.upper[0][ix];
  uur[iy] = wr_ivy - alpha * lorentz * metric.upper[0][iy];
  uur[iz] = wr_ivz - alpha * lorentz * metric.upper[0][iz];
  Real ulr[4];
  ulr[0] = metric.lower[0][0] * uur[0] + metric.lower[0][ix] * uur[ix] +
           metric.lower[0][iy] * uur[iy] + metric.lower[0][iz] * uur[iz];
  ulr[ix] = metric.lower[ix][0] * uur[0] + metric.lower[ix][ix] * uur[ix] +
            metric.lower[ix][iy] * uur[iy] + metric.lower[ix][iz] * uur[iz];
  ulr[iy] = metric.lower[iy][0] * uur[0] + metric.lower[iy][ix] * uur[ix] +
            metric.lower[iy][iy] * uur[iy] + metric.lower[iy][iz] * uur[iz];
  ulr[iz] = metric.lower[iz][0] * uur[0] + metric.lower[iz][ix] * uur[ix] +
            metric.lower[iz][iy] * uur[iy] + metric.lower[iz][iz] * uur[iz];
  Real bur[4];
  bur[0] = ulr[ix] * bxi + ulr[iy] * wr_iby + ulr[iz] * wr_ibz;
  bur[ix] = (bxi + bur[0] * uur[ix]) / uur[0];
  bur[iy] = (wr_iby + bur[0] * uur[iy]) / uur[0];
  bur[iz] = (wr_ibz + bur[0] * uur[iz]) / uur[0];
  Real blr[4];
  blr[0] = metric.lower[0][0] * bur[0] + metric.lower[0][ix] * bur[ix] +
           metric.lower[0][iy] * bur[iy] + metric.lower[0][iz] * bur[iz];
  blr[ix] = metric.lower[ix][0] * bur[0] + metric.lower[ix][ix] * bur[ix] +
            metric.lower[ix][iy] * bur[iy] + metric.lower[ix][iz] * bur[iz];
  blr[iy] = metric.lower[iy][0] * bur[0] + metric.lower[iy][ix] * bur[ix] +
            metric.lower[iy][iy] * bur[iy] + metric.lower[iy][iz] * bur[iz];
  blr[iz] = metric.lower[iz][0] * bur[0] + metric.lower[iz][ix] * bur[ix] +
            metric.lower[iz][iy] * bur[iy] + metric.lower[iz][iz] * bur[iz];
  const Real bsq_r = blr[0] * bur[0] + blr[ix] * bur[ix] + blr[iy] * bur[iy] + blr[iz] * bur[iz];

  const Real w_l = wl_idn + eos.gamma * wl_ipr / (eos.gamma - 1.0);
  const Real cs_l = eos.gamma * wl_ipr / w_l;
  const Real va_l = bsq_l / (bsq_l + w_l);
  const Real cms_l = cs_l + va_l - cs_l * va_l;
  Real a = uul[0] * uul[0] - (metric.upper[0][0] + uul[0] * uul[0]) * cms_l;
  Real b = -2.0 * (uul[0] * uul[ix] - (metric.upper[0][ix] + uul[0] * uul[ix]) * cms_l);
  Real c = uul[ix] * uul[ix] - (metric.upper[ix][ix] + uul[ix] * uul[ix]) * cms_l;
  Real a1 = b / a;
  Real a0 = c / a;
  Real root = fmax(a1 * a1 - 4.0 * a0, 0.0);
  root = sqrt(root);
  const Real lp_l = a1 >= 0.0 ? -2.0 * a0 / (a1 + root) : (-a1 + root) / 2.0;
  const Real lm_l = a1 >= 0.0 ? (-a1 - root) / 2.0 : -2.0 * a0 / (a1 - root);

  const Real w_r = wr_idn + eos.gamma * wr_ipr / (eos.gamma - 1.0);
  const Real cs_r = eos.gamma * wr_ipr / w_r;
  const Real va_r = bsq_r / (bsq_r + w_r);
  const Real cms_r = cs_r + va_r - cs_r * va_r;
  a = uur[0] * uur[0] - (metric.upper[0][0] + uur[0] * uur[0]) * cms_r;
  b = -2.0 * (uur[0] * uur[ix] - (metric.upper[0][ix] + uur[0] * uur[ix]) * cms_r);
  c = uur[ix] * uur[ix] - (metric.upper[ix][ix] + uur[ix] * uur[ix]) * cms_r;
  a1 = b / a;
  a0 = c / a;
  root = fmax(a1 * a1 - 4.0 * a0, 0.0);
  root = sqrt(root);
  const Real lp_r = a1 >= 0.0 ? -2.0 * a0 / (a1 + root) : (-a1 + root) / 2.0;
  const Real lm_r = a1 >= 0.0 ? (-a1 - root) / 2.0 : -2.0 * a0 / (a1 - root);
  const Real lambda_l = fmin(lm_l, lm_r);
  const Real lambda_r = fmax(lp_l, lp_r);

  const Real wtot_r = wr_idn + gamma_prime * wr_ipr + bsq_r;
  const Real ptot_r = wr_ipr + 0.5 * bsq_r;
  Real qa = wtot_r * uur[0];
  const Real wtot_l = wl_idn + gamma_prime * wl_ipr + bsq_l;
  const Real ptot_l = wl_ipr + 0.5 * bsq_l;
  Real qb = wtot_l * uul[0];
  const Real du_d = wr_idn * uur[0] - wl_idn * uul[0];
  const Real du_mx = (qa * ulr[ix] - bur[0] * blr[ix]) - (qb * ull[ix] - bul[0] * bll[ix]);
  const Real du_my = (qa * ulr[iy] - bur[0] * blr[iy]) - (qb * ull[iy] - bul[0] * bll[iy]);
  const Real du_mz = (qa * ulr[iz] - bur[0] * blr[iz]) - (qb * ull[iz] - bul[0] * bll[iz]);
  const Real du_e =
      (qa * ulr[0] - bur[0] * blr[0] + ptot_r) - (qb * ull[0] - bul[0] * bll[0] + ptot_l);
  const Real du_by = (bur[iy] * uur[0] - bur[0] * uur[iy]) - (bul[iy] * uul[0] - bul[0] * uul[iy]);
  const Real du_bz = (bur[iz] * uur[0] - bur[0] * uur[iz]) - (bul[iz] * uul[0] - bul[0] * uul[iz]);

  qa = wtot_l * uul[ix];
  const Real fl_d = wl_idn * uul[ix];
  const Real fl_mx = qa * ull[ix] - bul[ix] * bll[ix] + ptot_l;
  const Real fl_my = qa * ull[iy] - bul[ix] * bll[iy];
  const Real fl_mz = qa * ull[iz] - bul[ix] * bll[iz];
  const Real fl_e = qa * ull[0] - bul[ix] * bll[0];
  const Real fl_by = bul[iy] * uul[ix] - bul[ix] * uul[iy];
  const Real fl_bz = bul[iz] * uul[ix] - bul[ix] * uul[iz];
  qa = wtot_r * uur[ix];
  const Real fr_d = wr_idn * uur[ix];
  const Real fr_mx = qa * ulr[ix] - bur[ix] * blr[ix] + ptot_r;
  const Real fr_my = qa * ulr[iy] - bur[ix] * blr[iy];
  const Real fr_mz = qa * ulr[iz] - bur[ix] * blr[iz];
  const Real fr_e = qa * ulr[0] - bur[ix] * blr[0];
  const Real fr_by = bur[iy] * uur[ix] - bur[ix] * uur[iy];
  const Real fr_bz = bur[iz] * uur[ix] - bur[ix] * uur[iz];

  if constexpr (Solver == riemann::Solver::llf) {
    // Match AthenaK's SingleStateLLF_GRMHD expression tree.  In particular,
    // form the single extremal speed from the already materialized left/right
    // wave bounds and apply the magnetic sign convention only when the face
    // electric field is stored below.
    const Real lambda = fmax(lambda_r, -lambda_l);
    RMHDFluxState result{};
    result.lambda_plus = lambda_r;
    result.lambda_minus = lambda_l;
    result.flux[mhd::IDN] = 0.5 * (fl_d + fr_d - lambda * du_d);
    result.flux[mhd::IM1 + vx] = 0.5 * (fl_mx + fr_mx - lambda * du_mx);
    result.flux[mhd::IM1 + vy] = 0.5 * (fl_my + fr_my - lambda * du_my);
    result.flux[mhd::IM1 + vz] = 0.5 * (fl_mz + fr_mz - lambda * du_mz);
    result.flux[mhd::IEN] = 0.5 * (fl_e + fr_e - lambda * du_e);
    result.induction[vy] = 0.5 * (fl_by + fr_by - lambda * du_by);
    result.induction[vz] = 0.5 * (fl_bz + fr_bz - lambda * du_bz);
    result.flux[mhd::IEN] += result.flux[mhd::IDN];
    for (int n = 0; n < 5; ++n)
      result.flux[n] *= metric.gdet;
    for (int axis = 0; axis < 3; ++axis)
      result.induction[axis] *= metric.gdet;
    // Keep the passive-channel arithmetic after the complete MHD result.  The
    // electron module must not perturb the expression tree used to form the
    // primary GRMHD flux in floor-dominated cells.
    if constexpr (WritePassiveTransport) {
      *passive =
          mhd::LLFPassiveTransport(wl_idn * uul[0] * metric.gdet, wr_idn * uur[0] * metric.gdet,
                                   fl_d * metric.gdet, fr_d * metric.gdet, lambda);
    }
    return result;
  }

  qa = lambda_r * lambda_l;
  qb = 1.0 / (lambda_r - lambda_l);
  const Real hll_d = (lambda_r * fl_d - lambda_l * fr_d + qa * du_d) * qb;
  const Real hll_mx = (lambda_r * fl_mx - lambda_l * fr_mx + qa * du_mx) * qb;
  const Real hll_my = (lambda_r * fl_my - lambda_l * fr_my + qa * du_my) * qb;
  const Real hll_mz = (lambda_r * fl_mz - lambda_l * fr_mz + qa * du_mz) * qb;
  const Real hll_e = (lambda_r * fl_e - lambda_l * fr_e + qa * du_e) * qb;
  const Real hll_by = (lambda_r * fl_by - lambda_l * fr_by + qa * du_by) * qb;
  const Real hll_bz = (lambda_r * fl_bz - lambda_l * fr_bz + qa * du_bz) * qb;

  RMHDFluxState result{};
  result.lambda_plus = lambda_r;
  result.lambda_minus = lambda_l;
  if (lambda_l >= 0.0) {
    result.flux[mhd::IDN] = fl_d;
    result.flux[mhd::IM1 + vx] = fl_mx;
    result.flux[mhd::IM1 + vy] = fl_my;
    result.flux[mhd::IM1 + vz] = fl_mz;
    result.flux[mhd::IEN] = fl_e;
    result.induction[vy] = fl_by;
    result.induction[vz] = fl_bz;
  } else if (lambda_r <= 0.0) {
    result.flux[mhd::IDN] = fr_d;
    result.flux[mhd::IM1 + vx] = fr_mx;
    result.flux[mhd::IM1 + vy] = fr_my;
    result.flux[mhd::IM1 + vz] = fr_mz;
    result.flux[mhd::IEN] = fr_e;
    result.induction[vy] = fr_by;
    result.induction[vz] = fr_bz;
  } else {
    result.flux[mhd::IDN] = hll_d;
    result.flux[mhd::IM1 + vx] = hll_mx;
    result.flux[mhd::IM1 + vy] = hll_my;
    result.flux[mhd::IM1 + vz] = hll_mz;
    result.flux[mhd::IEN] = hll_e;
    result.induction[vy] = hll_by;
    result.induction[vz] = hll_bz;
  }
  result.flux[mhd::IEN] += result.flux[mhd::IDN];
  for (int n = 0; n < 5; ++n)
    result.flux[n] *= metric.gdet;
  for (int axis = 0; axis < 3; ++axis)
    result.induction[axis] *= metric.gdet;
  if constexpr (WritePassiveTransport) {
    *passive =
        mhd::HLLEPassiveTransport(wl_idn * uul[0] * metric.gdet, wr_idn * uur[0] * metric.gdet,
                                  fl_d * metric.gdet, fr_d * metric.gdet, lambda_l, lambda_r);
  }
  return result;
}

template <int Direction, bool WritePassiveTransport, typename PrimitivePack, typename MagneticPack,
          typename ConservedPack, typename BFaceArray, typename FaceEmfPack, typename FlagPack,
          typename TransportPack>
void ReplaceFlaggedFacesBlock(MeshBlock* block, const PrimitivePack& primitive,
                              const MagneticPack& magnetic, ConservedPack& conserved,
                              const BFaceArray& bface, const FaceEmfPack& face_emf,
                              const FlagPack& flags, const HydroMode mode, const eos::RelativisticEOS& eos,
                              const IndexRange& flag_i, const IndexRange& flag_j,
                              const IndexRange& flag_k, const TransportPack& transport) {
  int il = flag_i.s, iu = flag_i.e, jl = flag_j.s, ju = flag_j.e;
  int kl = flag_k.s, ku = flag_k.e;
  if constexpr (Direction == 0)
    ++iu;
  if constexpr (Direction == 1)
    ++ju;
  if constexpr (Direction == 2)
    ++ku;
  const auto coords = block->coords;
  const auto spacetime = geometry::GetGeometry(block->packages);
  const int geometry_block = block->lid;
  block->par_for(
      "PANGU relativistic MHD FOFC face", kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const int lk = k - (Direction == 2);
        const int lj = j - (Direction == 1);
        const int li = i - (Direction == 0);
        const bool has_left = Direction == 0   ? i > flag_i.s
                              : Direction == 1 ? j > flag_j.s
                                               : k > flag_k.s;
        const bool has_right = Direction == 0   ? i <= flag_i.e
                               : Direction == 1 ? j <= flag_j.e
                                                : k <= flag_k.e;
        if (!((has_left && flags(0, lk, lj, li) > 0.5) || (has_right && flags(0, k, j, i) > 0.5)))
          return;
        MHDPrimitiveState left{}, right{};
        left.fluid = {primitive(mhd::IDN, lk, lj, li),
                      {primitive(mhd::IV1, lk, lj, li), primitive(mhd::IV2, lk, lj, li),
                       primitive(mhd::IV3, lk, lj, li)},
                      eos.PressureFromInternalEnergyDensity(
                          primitive(mhd::IPR, lk, lj, li))};
        right.fluid = {primitive(mhd::IDN, k, j, i),
                       {primitive(mhd::IV1, k, j, i), primitive(mhd::IV2, k, j, i),
                        primitive(mhd::IV3, k, j, i)},
                       eos.PressureFromInternalEnergyDensity(
                           primitive(mhd::IPR, k, j, i))};
        for (int axis = 0; axis < 3; ++axis) {
          left.magnetic[axis] = magnetic(axis, lk, lj, li);
          right.magnetic[axis] = magnetic(axis, k, j, i);
        }
        RMHDFluxState flux{};
        mhd::PassiveTransportCoefficients passive{};
        if (mode == HydroMode::sr) {
          const Real normal = bface(Direction, 0, 0, 0, k, j, i);
          left.magnetic[Direction] = normal;
          right.magnetic[Direction] = normal;
          const auto left_state = BuildSRMHDFluxState(left, eos, Direction);
          const auto right_state = BuildSRMHDFluxState(right, eos, Direction);
          flux = SolveRMHDLLFFromStates(left_state, right_state);
          if (WritePassiveTransport)
            passive = SolveLLFPassiveTransportFromStates(left_state, right_state);
        } else {
          const auto metric = spacetime.MetricAt(geometry::FaceLocation(Direction), geometry_block,
                                                 k, j, i, coords);
          const Real normal = bface(Direction, 0, 0, 0, k, j, i) / metric.gdet;
          left.magnetic[Direction] = normal;
          right.magnetic[Direction] = normal;
          flux = SolveGRMHDRiemannDirect<Direction, riemann::first_order_fallback, WritePassiveTransport>(
              left, right, eos, metric, &passive);
        }
        WritePassiveFluxBlock(transport, Direction, k, j, i, passive,
                              std::integral_constant<bool, WritePassiveTransport>{});
        for (int n = 0; n < 5; ++n)
          conserved.flux(Direction + 1, n, k, j, i) = flux.flux[n];
        const int tangent1 = (Direction + 1) % 3;
        const int tangent2 = (Direction + 2) % 3;
        const Real electric_t1 = flux.induction[tangent2];
        const Real electric_t2 = -flux.induction[tangent1];
        if (Direction == 0) {
          face_emf(E2X1, k, j, i) = electric_t1;
          face_emf(E3X1, k, j, i) = electric_t2;
        } else if (Direction == 1) {
          face_emf(E3X2, k, j, i) = electric_t1;
          face_emf(E1X2, k, j, i) = electric_t2;
        } else {
          face_emf(E1X3, k, j, i) = electric_t1;
          face_emf(E2X3, k, j, i) = electric_t2;
        }
      });
}

template <int Direction, bool WritePassiveTransport, typename PrimitivePack, typename MagneticPack,
          typename ConservedPack, typename BFacePack, typename FaceEmfPack, typename FlagPack,
          typename TransportPack>
void ReplaceFlaggedFacesMesh(MeshData<Real>* data, const PrimitivePack& primitive,
                             const MagneticPack& magnetic, ConservedPack& conserved,
                             const BFacePack& bface, const FaceEmfPack& face_emf,
                             const FlagPack& flags, const HydroMode mode, const eos::RelativisticEOS& eos,
                             const IndexRange& flag_i, const IndexRange& flag_j,
                             const IndexRange& flag_k, const TransportPack& transport) {
  using TE = parthenon::TopologicalElement;
  int il = flag_i.s, iu = flag_i.e, jl = flag_j.s, ju = flag_j.e;
  int kl = flag_k.s, ku = flag_k.e;
  if constexpr (Direction == 0)
    ++iu;
  if constexpr (Direction == 1)
    ++ju;
  if constexpr (Direction == 2)
    ++ku;
  constexpr TE face = Direction == 0 ? TE::F1 : (Direction == 1 ? TE::F2 : TE::F3);
  const auto spacetime = geometry::GetGeometry(data->GetParentPointer()->packages);
  const int geometry_block_offset = data->GetBlockData(0)->GetBlockPointer()->lid;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU packed relativistic MHD FOFC face", parthenon::DevExecSpace(), 0,
      data->NumBlocks() - 1, kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const int lk = k - (Direction == 2);
        const int lj = j - (Direction == 1);
        const int li = i - (Direction == 0);
        const bool has_left = Direction == 0   ? i > flag_i.s
                              : Direction == 1 ? j > flag_j.s
                                               : k > flag_k.s;
        const bool has_right = Direction == 0   ? i <= flag_i.e
                               : Direction == 1 ? j <= flag_j.e
                                                : k <= flag_k.e;
        if (!((has_left && flags(b, 0, lk, lj, li) > 0.5) ||
              (has_right && flags(b, 0, k, j, i) > 0.5)))
          return;
        MHDPrimitiveState left{}, right{};
        left.fluid = {primitive(b, mhd::IDN, lk, lj, li),
                      {primitive(b, mhd::IV1, lk, lj, li), primitive(b, mhd::IV2, lk, lj, li),
                       primitive(b, mhd::IV3, lk, lj, li)},
                      eos.PressureFromInternalEnergyDensity(
                          primitive(b, mhd::IPR, lk, lj, li))};
        right.fluid = {primitive(b, mhd::IDN, k, j, i),
                       {primitive(b, mhd::IV1, k, j, i), primitive(b, mhd::IV2, k, j, i),
                        primitive(b, mhd::IV3, k, j, i)},
                       eos.PressureFromInternalEnergyDensity(
                           primitive(b, mhd::IPR, k, j, i))};
        for (int axis = 0; axis < 3; ++axis) {
          left.magnetic[axis] = magnetic(b, axis, lk, lj, li);
          right.magnetic[axis] = magnetic(b, axis, k, j, i);
        }
        RMHDFluxState flux{};
        mhd::PassiveTransportCoefficients passive{};
        if (mode == HydroMode::sr) {
          const Real normal = bface(b, face, 0, k, j, i);
          left.magnetic[Direction] = normal;
          right.magnetic[Direction] = normal;
          const auto left_state = BuildSRMHDFluxState(left, eos, Direction);
          const auto right_state = BuildSRMHDFluxState(right, eos, Direction);
          flux = SolveRMHDLLFFromStates(left_state, right_state);
          if (WritePassiveTransport)
            passive = SolveLLFPassiveTransportFromStates(left_state, right_state);
        } else {
          const auto& coords = primitive.GetCoords(b);
          const auto metric = spacetime.MetricAt(geometry::FaceLocation(Direction),
                                                 geometry_block_offset + b, k, j, i, coords);
          const Real normal = bface(b, face, 0, k, j, i) / metric.gdet;
          left.magnetic[Direction] = normal;
          right.magnetic[Direction] = normal;
          flux = SolveGRMHDRiemannDirect<Direction, riemann::first_order_fallback, WritePassiveTransport>(
              left, right, eos, metric, &passive);
        }
        WritePassiveFluxMesh(transport, b, Direction, k, j, i, passive,
                             std::integral_constant<bool, WritePassiveTransport>{});
        auto block_conserved = conserved(b);
        for (int n = 0; n < 5; ++n)
          block_conserved.flux(Direction + 1, n, k, j, i) = flux.flux[n];
        const int tangent1 = (Direction + 1) % 3;
        const int tangent2 = (Direction + 2) % 3;
        const Real electric_t1 = flux.induction[tangent2];
        const Real electric_t2 = -flux.induction[tangent1];
        if (Direction == 0) {
          face_emf(b, E2X1, k, j, i) = electric_t1;
          face_emf(b, E3X1, k, j, i) = electric_t2;
        } else if (Direction == 1) {
          face_emf(b, E3X2, k, j, i) = electric_t1;
          face_emf(b, E1X2, k, j, i) = electric_t2;
        } else {
          face_emf(b, E1X3, k, j, i) = electric_t1;
          face_emf(b, E2X3, k, j, i) = electric_t2;
        }
      });
}

template <bool WritePassiveTransport, typename PrimitivePack, typename MagneticPack,
          typename ConservedPack, typename BFaceArray, typename FaceEmfPack, typename FlagPack,
          typename TransportPack>
void ReplaceAllFlaggedFacesBlock(MeshBlock* block, const PrimitivePack& primitive,
                                 const MagneticPack& magnetic, ConservedPack& conserved,
                                 const BFaceArray& bface, const FaceEmfPack& face_emf,
                                 const FlagPack& flags, const HydroMode mode, const eos::RelativisticEOS& eos,
                                 const IndexRange& flag_i, const IndexRange& flag_j,
                                 const IndexRange& flag_k, const int ndim,
                                 const TransportPack& transport) {
  ReplaceFlaggedFacesBlock<0, WritePassiveTransport>(block, primitive, magnetic, conserved, bface,
                                                     face_emf, flags, mode, eos, flag_i, flag_j,
                                                     flag_k, transport);
  if (ndim >= 2)
    ReplaceFlaggedFacesBlock<1, WritePassiveTransport>(block, primitive, magnetic, conserved, bface,
                                                       face_emf, flags, mode, eos, flag_i, flag_j,
                                                       flag_k, transport);
  if (ndim >= 3)
    ReplaceFlaggedFacesBlock<2, WritePassiveTransport>(block, primitive, magnetic, conserved, bface,
                                                       face_emf, flags, mode, eos, flag_i, flag_j,
                                                       flag_k, transport);
}

template <int Direction, HydroMode Mode, riemann::Solver Solver, hydro::Reconstruction Method,
          bool WritePassiveTransport, typename TransportPack>
void CalculateDirectionBlock(const std::shared_ptr<MeshBlockData<Real>>& data, const eos::RelativisticEOS& eos,
                             const bool excision, const TransportPack& transport) {
  auto block = data->GetBlockPointer();
  auto primitive = data->PackVariables(std::vector<std::string>{"mhd.prim"});
  auto magnetic = data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  auto reconstructed_left = data->PackVariables(std::vector<std::string>{"mhd.recon_left"});
  auto reconstructed_right = data->PackVariables(std::vector<std::string>{"mhd.recon_right"});
  auto reconstructed_b_left = data->PackVariables(std::vector<std::string>{"mhd.recon_b_left"});
  auto reconstructed_b_right = data->PackVariables(std::vector<std::string>{"mhd.recon_b_right"});
  auto conserved = data->PackVariablesAndFluxes(std::vector<std::string>{"mhd.cons"});
  auto face_emf = data->PackVariables(std::vector<std::string>{"mhd.face_emf"});
  auto bface = data->Get("mhd.b_face").data;
  parthenon::VariablePack<Real> signal_max;
  parthenon::VariablePack<Real> signal_min;
  if constexpr (Estimator::face_signal_speeds) {
    signal_max = data->PackVariables(std::vector<std::string>{"mhd.cmax"});
    signal_min = data->PackVariables(std::vector<std::string>{"mhd.cmin"});
  }
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto coords = block->coords;
  const auto spacetime = geometry::GetGeometry(block->packages);
  const int geometry_block = block->lid;
  const int ndim = block->pmy_mesh->ndim;
  const bool replace_first_order =
      block->packages.Get("mhd")->Param<bool>("fofc") || (Mode == HydroMode::gr && excision);
  int il = ib.s, iu = ib.e, jl = jb.s, ju = jb.e, kl = kb.s, ku = kb.e;
  if constexpr (Direction == 0) {
    ++iu;
    if (replace_first_order) {
      --il;
      ++iu;
    }
    if (ndim >= 2) {
      --jl;
      ++ju;
    }
    if (ndim >= 3) {
      --kl;
      ++ku;
    }
  }
  if constexpr (Direction == 1) {
    ++ju;
    if (replace_first_order) {
      --jl;
      ++ju;
    }
    --il;
    ++iu;
    if (ndim >= 3) {
      --kl;
      ++ku;
    }
  }
  if constexpr (Direction == 2) {
    ++ku;
    if (replace_first_order) {
      --kl;
      ++ku;
    }
    --jl;
    ++ju;
    --il;
    ++iu;
  }
  int recon_il = il, recon_iu = iu;
  int recon_jl = jl, recon_ju = ju;
  int recon_kl = kl, recon_ku = ku;
  // Cell-edge models write the face above the cell they visit, so the loop needs
  // the extra cell on the low side.
  if constexpr (reconstruct::Implementation<Method>::cell_edge_states) {
    if constexpr (Direction == 0)
      --recon_il;
    if constexpr (Direction == 1)
      --recon_jl;
    if constexpr (Direction == 2)
      --recon_kl;
  }
  block->par_for(
      "PANGU relativistic MHD fluid reconstruction", 0, 4, recon_kl, recon_ku, recon_jl, recon_ju,
      recon_il, recon_iu, KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
        ReconstructCellBlock<Direction, Method>(primitive, reconstructed_left, reconstructed_right,
                                                n, k, j, i);
      });
  block->par_for(
      "PANGU relativistic MHD magnetic reconstruction", 0, 2, recon_kl, recon_ku, recon_jl,
      recon_ju, recon_il, recon_iu,
      KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
        ReconstructCellBlock<Direction, Method>(magnetic, reconstructed_b_left,
                                                reconstructed_b_right, n, k, j, i);
      });
  block->par_for(
      "PANGU relativistic MHD Riemann solve", kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        MHDPrimitiveState left{}, right{};
        left.fluid = {reconstructed_left(mhd::IDN, k, j, i),
                      {reconstructed_left(mhd::IV1, k, j, i), reconstructed_left(mhd::IV2, k, j, i),
                       reconstructed_left(mhd::IV3, k, j, i)},
                      eos.PressureFromInternalEnergyDensity(
                          reconstructed_left(mhd::IPR, k, j, i))};
        right.fluid = {reconstructed_right(mhd::IDN, k, j, i),
                       {reconstructed_right(mhd::IV1, k, j, i),
                        reconstructed_right(mhd::IV2, k, j, i),
                        reconstructed_right(mhd::IV3, k, j, i)},
                       eos.PressureFromInternalEnergyDensity(
                           reconstructed_right(mhd::IPR, k, j, i))};
        for (int axis = 0; axis < 3; ++axis) {
          left.magnetic[axis] = reconstructed_b_left(axis, k, j, i);
          right.magnetic[axis] = reconstructed_b_right(axis, k, j, i);
        }
        RMHDFluxState flux{};
        Real lambda_plus = 0.0;
        Real lambda_minus = 0.0;
        mhd::PassiveTransportCoefficients passive{};
        if (Mode == HydroMode::sr) {
          const Real normal = bface(Direction, 0, 0, 0, k, j, i);
          left.magnetic[Direction] = normal;
          right.magnetic[Direction] = normal;
          const auto left_state = BuildSRMHDFluxState(left, eos, Direction);
          const auto right_state = BuildSRMHDFluxState(right, eos, Direction);
          lambda_plus = fmax(left_state.lambda_plus, right_state.lambda_plus);
          lambda_minus = fmin(left_state.lambda_minus, right_state.lambda_minus);
          if (Solver == riemann::Solver::llf) {
            flux = SolveRMHDLLFFromStates(left_state, right_state);
            if (WritePassiveTransport)
              passive = SolveLLFPassiveTransportFromStates(left_state, right_state);
          } else {
            flux = SolveRMHDHLLEFromStates(left_state, right_state);
            if (WritePassiveTransport)
              passive = SolveHLLEPassiveTransportFromStates(left_state, right_state);
          }
        } else {
          const auto metric = spacetime.MetricAt(geometry::FaceLocation(Direction), geometry_block,
                                                 k, j, i, coords);
          const Real normal = bface(Direction, 0, 0, 0, k, j, i) / metric.gdet;
          left.magnetic[Direction] = normal;
          right.magnetic[Direction] = normal;
          if (Solver == riemann::Solver::llf) {
            const auto left_state = BuildGRMHDFluxState(left, eos, Direction, metric);
            const auto right_state = BuildGRMHDFluxState(right, eos, Direction, metric);
            lambda_plus = fmax(left_state.lambda_plus, right_state.lambda_plus);
            lambda_minus = fmin(left_state.lambda_minus, right_state.lambda_minus);
            flux = SolveGRMHDLLFFromStates(left_state, right_state);
            if (WritePassiveTransport)
              passive = SolveLLFPassiveTransportFromStates(left_state, right_state);
          } else {
            flux =
                SolveGRMHDRiemannDirect<Direction, riemann::Solver::hlle, WritePassiveTransport>(
                    left, right, eos, metric, &passive);
            lambda_plus = flux.lambda_plus;
            lambda_minus = flux.lambda_minus;
          }
        }
        StoreSignalSpeedsBlock(signal_max, signal_min, Direction, k, j, i, lambda_plus,
                               lambda_minus,
                               std::integral_constant<bool,
                                                      Estimator::face_signal_speeds>{});
        WritePassiveFluxBlock(transport, Direction, k, j, i, passive,
                              std::integral_constant<bool, WritePassiveTransport>{});
        for (int n = 0; n < 5; ++n)
          conserved.flux(Direction + 1, n, k, j, i) = flux.flux[n];
        const int tangent1 = (Direction + 1) % 3;
        const int tangent2 = (Direction + 2) % 3;
        const Real electric_t1 = flux.induction[tangent2];
        const Real electric_t2 = -flux.induction[tangent1];
        if (Direction == 0) {
          face_emf(E2X1, k, j, i) = electric_t1;
          face_emf(E3X1, k, j, i) = electric_t2;
        } else if (Direction == 1) {
          face_emf(E3X2, k, j, i) = electric_t1;
          face_emf(E1X2, k, j, i) = electric_t2;
        } else {
          face_emf(E1X3, k, j, i) = electric_t1;
          face_emf(E2X3, k, j, i) = electric_t2;
        }
      });
}

template <int Direction, HydroMode Mode, riemann::Solver Solver, hydro::Reconstruction Method,
          bool WritePassiveTransport, typename TransportPack>
void CalculateDirectionMesh(MeshData<Real>* data, const eos::RelativisticEOS& eos, const bool excision,
                            const TransportPack& transport) {
  using TE = parthenon::TopologicalElement;
  auto primitive = data->PackVariables(std::vector<std::string>{"mhd.prim"});
  auto magnetic = data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  auto reconstructed_left = data->PackVariables(std::vector<std::string>{"mhd.recon_left"});
  auto reconstructed_right = data->PackVariables(std::vector<std::string>{"mhd.recon_right"});
  auto reconstructed_b_left = data->PackVariables(std::vector<std::string>{"mhd.recon_b_left"});
  auto reconstructed_b_right = data->PackVariables(std::vector<std::string>{"mhd.recon_b_right"});
  auto conserved = data->PackVariablesAndFluxes(std::vector<std::string>{"mhd.cons"});
  auto face_emf = data->PackVariables(std::vector<std::string>{"mhd.face_emf"});
  auto bface = data->PackVariables(std::vector<std::string>{"mhd.b_face"});
  parthenon::MeshBlockVarPack<Real> signal_max;
  parthenon::MeshBlockVarPack<Real> signal_min;
  if constexpr (Estimator::face_signal_speeds) {
    signal_max = data->PackVariables(std::vector<std::string>{"mhd.cmax"});
    signal_min = data->PackVariables(std::vector<std::string>{"mhd.cmin"});
  }
  const auto ib = data->GetBoundsI(IndexDomain::interior);
  const auto jb = data->GetBoundsJ(IndexDomain::interior);
  const auto kb = data->GetBoundsK(IndexDomain::interior);
  const int ndim = data->GetParentPointer()->ndim;
  const int nblocks = data->NumBlocks();
  const auto spacetime = geometry::GetGeometry(data->GetParentPointer()->packages);
  const int geometry_block_offset = data->GetBlockData(0)->GetBlockPointer()->lid;
  const bool replace_first_order =
      data->GetParentPointer()->packages.Get("mhd")->Param<bool>("fofc") ||
      (Mode == HydroMode::gr && excision);
  int il = ib.s, iu = ib.e, jl = jb.s, ju = jb.e, kl = kb.s, ku = kb.e;
  if constexpr (Direction == 0) {
    ++iu;
    if (replace_first_order) {
      --il;
      ++iu;
    }
    if (ndim >= 2) {
      --jl;
      ++ju;
    }
    if (ndim >= 3) {
      --kl;
      ++ku;
    }
  }
  if constexpr (Direction == 1) {
    ++ju;
    if (replace_first_order) {
      --jl;
      ++ju;
    }
    --il;
    ++iu;
    if (ndim >= 3) {
      --kl;
      ++ku;
    }
  }
  if constexpr (Direction == 2) {
    ++ku;
    if (replace_first_order) {
      --kl;
      ++ku;
    }
    --jl;
    ++ju;
    --il;
    ++iu;
  }
  int recon_il = il, recon_iu = iu;
  int recon_jl = jl, recon_ju = ju;
  int recon_kl = kl, recon_ku = ku;
  // Cell-edge models write the face above the cell they visit, so the loop needs
  // the extra cell on the low side.
  if constexpr (reconstruct::Implementation<Method>::cell_edge_states) {
    if constexpr (Direction == 0)
      --recon_il;
    if constexpr (Direction == 1)
      --recon_jl;
    if constexpr (Direction == 2)
      --recon_kl;
  }
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU packed relativistic MHD fluid reconstruction",
      parthenon::DevExecSpace(), 0, nblocks - 1, 0, 4, recon_kl, recon_ku, recon_jl, recon_ju,
      recon_il, recon_iu,
      KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i) {
        ReconstructCellMesh<Direction, Method>(primitive, reconstructed_left, reconstructed_right,
                                               b, n, k, j, i);
      });
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU packed relativistic MHD magnetic reconstruction",
      parthenon::DevExecSpace(), 0, nblocks - 1, 0, 2, recon_kl, recon_ku, recon_jl, recon_ju,
      recon_il, recon_iu,
      KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i) {
        ReconstructCellMesh<Direction, Method>(magnetic, reconstructed_b_left,
                                               reconstructed_b_right, b, n, k, j, i);
      });
  constexpr TE face = Direction == 0 ? TE::F1 : (Direction == 1 ? TE::F2 : TE::F3);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU packed relativistic MHD Riemann solve",
      parthenon::DevExecSpace(), 0, nblocks - 1, kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        MHDPrimitiveState left{}, right{};
        left.fluid = {reconstructed_left(b, mhd::IDN, k, j, i),
                      {reconstructed_left(b, mhd::IV1, k, j, i),
                       reconstructed_left(b, mhd::IV2, k, j, i),
                       reconstructed_left(b, mhd::IV3, k, j, i)},
                      eos.PressureFromInternalEnergyDensity(
                          reconstructed_left(b, mhd::IPR, k, j, i))};
        right.fluid = {reconstructed_right(b, mhd::IDN, k, j, i),
                       {reconstructed_right(b, mhd::IV1, k, j, i),
                        reconstructed_right(b, mhd::IV2, k, j, i),
                        reconstructed_right(b, mhd::IV3, k, j, i)},
                       eos.PressureFromInternalEnergyDensity(
                           reconstructed_right(b, mhd::IPR, k, j, i))};
        for (int axis = 0; axis < 3; ++axis) {
          left.magnetic[axis] = reconstructed_b_left(b, axis, k, j, i);
          right.magnetic[axis] = reconstructed_b_right(b, axis, k, j, i);
        }
        const auto& coords = primitive.GetCoords(b);
        RMHDFluxState flux{};
        Real lambda_plus = 0.0;
        Real lambda_minus = 0.0;
        mhd::PassiveTransportCoefficients passive{};
        if (Mode == HydroMode::sr) {
          const Real normal = bface(b, face, 0, k, j, i);
          left.magnetic[Direction] = normal;
          right.magnetic[Direction] = normal;
          const auto left_state = BuildSRMHDFluxState(left, eos, Direction);
          const auto right_state = BuildSRMHDFluxState(right, eos, Direction);
          lambda_plus = fmax(left_state.lambda_plus, right_state.lambda_plus);
          lambda_minus = fmin(left_state.lambda_minus, right_state.lambda_minus);
          if (Solver == riemann::Solver::llf) {
            flux = SolveRMHDLLFFromStates(left_state, right_state);
            if (WritePassiveTransport)
              passive = SolveLLFPassiveTransportFromStates(left_state, right_state);
          } else {
            flux = SolveRMHDHLLEFromStates(left_state, right_state);
            if (WritePassiveTransport)
              passive = SolveHLLEPassiveTransportFromStates(left_state, right_state);
          }
        } else {
          const auto metric = spacetime.MetricAt(geometry::FaceLocation(Direction),
                                                 geometry_block_offset + b, k, j, i, coords);
          const Real normal = bface(b, face, 0, k, j, i) / metric.gdet;
          left.magnetic[Direction] = normal;
          right.magnetic[Direction] = normal;
          if (Solver == riemann::Solver::llf) {
            const auto left_state = BuildGRMHDFluxState(left, eos, Direction, metric);
            const auto right_state = BuildGRMHDFluxState(right, eos, Direction, metric);
            lambda_plus = fmax(left_state.lambda_plus, right_state.lambda_plus);
            lambda_minus = fmin(left_state.lambda_minus, right_state.lambda_minus);
            flux = SolveGRMHDLLFFromStates(left_state, right_state);
            if (WritePassiveTransport)
              passive = SolveLLFPassiveTransportFromStates(left_state, right_state);
          } else {
            flux =
                SolveGRMHDRiemannDirect<Direction, riemann::Solver::hlle, WritePassiveTransport>(
                    left, right, eos, metric, &passive);
            lambda_plus = flux.lambda_plus;
            lambda_minus = flux.lambda_minus;
          }
        }
        StoreSignalSpeedsMesh(signal_max, signal_min, b, Direction, k, j, i, lambda_plus,
                              lambda_minus,
                              std::integral_constant<bool,
                                                     Estimator::face_signal_speeds>{});
        WritePassiveFluxMesh(transport, b, Direction, k, j, i, passive,
                             std::integral_constant<bool, WritePassiveTransport>{});
        auto block_conserved = conserved(b);
        for (int n = 0; n < 5; ++n)
          block_conserved.flux(Direction + 1, n, k, j, i) = flux.flux[n];
        const int tangent1 = (Direction + 1) % 3;
        const int tangent2 = (Direction + 2) % 3;
        const Real electric_t1 = flux.induction[tangent2];
        const Real electric_t2 = -flux.induction[tangent1];
        if (Direction == 0) {
          face_emf(b, E2X1, k, j, i) = electric_t1;
          face_emf(b, E3X1, k, j, i) = electric_t2;
        } else if (Direction == 1) {
          face_emf(b, E3X2, k, j, i) = electric_t1;
          face_emf(b, E1X2, k, j, i) = electric_t2;
        } else {
          face_emf(b, E1X3, k, j, i) = electric_t1;
          face_emf(b, E2X3, k, j, i) = electric_t2;
        }
      });
}
template <HydroMode Mode, riemann::Solver Solver, hydro::Reconstruction Method,
          bool WritePassiveTransport, typename TransportPack>
void CalculateFluxesSpecializedBlock(const std::shared_ptr<MeshBlockData<Real>>& data,
                                     const eos::RelativisticEOS& eos, const bool excision,
                                     const TransportPack& transport) {
  auto block = data->GetBlockPointer();
  CalculateDirectionBlock<0, Mode, Solver, Method, WritePassiveTransport>(
      data, eos, excision, transport);
  if (block->pmy_mesh->ndim >= 2)
    CalculateDirectionBlock<1, Mode, Solver, Method, WritePassiveTransport>(
        data, eos, excision, transport);
  if (block->pmy_mesh->ndim >= 3)
    CalculateDirectionBlock<2, Mode, Solver, Method, WritePassiveTransport>(
        data, eos, excision, transport);
}

template <HydroMode Mode, riemann::Solver Solver, hydro::Reconstruction Method,
          bool WritePassiveTransport, typename TransportPack>
void CalculateFluxesSpecializedMesh(MeshData<Real>* data, const eos::RelativisticEOS& eos,
                                    const bool excision, const TransportPack& transport) {
  CalculateDirectionMesh<0, Mode, Solver, Method, WritePassiveTransport>(
      data, eos, excision, transport);
  if (data->GetParentPointer()->ndim >= 2)
    CalculateDirectionMesh<1, Mode, Solver, Method, WritePassiveTransport>(
        data, eos, excision, transport);
  if (data->GetParentPointer()->ndim >= 3)
    CalculateDirectionMesh<2, Mode, Solver, Method, WritePassiveTransport>(
        data, eos, excision, transport);
}

template <HydroMode Mode, riemann::Solver Solver, bool WritePassiveTransport,
          typename TransportPack>
void DispatchMHDReconstructionBlock(const std::shared_ptr<MeshBlockData<Real>>& data,
                                    const eos::RelativisticEOS& eos, const bool excision,
                                    const hydro::Reconstruction reconstruction,
                                    const TransportPack& transport) {
  reconstruct::VisitRelativisticMHD(reconstruction, [&]<hydro::Reconstruction Method>() {
    CalculateFluxesSpecializedBlock<Mode, Solver, Method, WritePassiveTransport>(
        data, eos, excision, transport);
  });
}

template <HydroMode Mode, riemann::Solver Solver, bool WritePassiveTransport,
          typename TransportPack>
void DispatchMHDReconstructionMesh(MeshData<Real>* data, const eos::RelativisticEOS& eos, const bool excision,
                                   const hydro::Reconstruction reconstruction,
                                   const TransportPack& transport) {
  reconstruct::VisitRelativisticMHD(reconstruction, [&]<hydro::Reconstruction Method>() {
    CalculateFluxesSpecializedMesh<Mode, Solver, Method, WritePassiveTransport>(
        data, eos, excision, transport);
  });
}

template <bool WritePassiveTransport, typename TransportPack>
void DispatchMHDFluxesBlock(const std::shared_ptr<MeshBlockData<Real>>& data, const eos::RelativisticEOS& eos,
                            const bool excision,
                            const hydro::Reconstruction reconstruction, const HydroMode mode,
                            const riemann::Solver solver, const TransportPack& transport) {
  riemann::Visit<riemann::Physics::relativistic_mhd>(solver, [&]<riemann::Solver Solver>() {
    if (mode == HydroMode::sr)
      DispatchMHDReconstructionBlock<HydroMode::sr, Solver, WritePassiveTransport>(
          data, eos, excision, reconstruction, transport);
    else
      DispatchMHDReconstructionBlock<HydroMode::gr, Solver, WritePassiveTransport>(
          data, eos, excision, reconstruction, transport);
  });
}

template <bool WritePassiveTransport, typename TransportPack>
void DispatchMHDFluxesMesh(MeshData<Real>* data, const eos::RelativisticEOS& eos, const bool excision,
                           const hydro::Reconstruction reconstruction,
                           const HydroMode mode, const riemann::Solver solver,
                           const TransportPack& transport) {
  riemann::Visit<riemann::Physics::relativistic_mhd>(solver, [&]<riemann::Solver Solver>() {
    if (mode == HydroMode::sr)
      DispatchMHDReconstructionMesh<HydroMode::sr, Solver, WritePassiveTransport>(
          data, eos, excision, reconstruction, transport);
    else
      DispatchMHDReconstructionMesh<HydroMode::gr, Solver, WritePassiveTransport>(
          data, eos, excision, reconstruction, transport);
  });
}

} // namespace

TaskStatus CalculateMHDFluxesBlockTask(std::shared_ptr<MeshBlockData<Real>>& data) {
  auto block = data->GetBlockPointer();
  const auto package = block->packages.Get("mhd");
  const auto eos = pangu::eos::ReadRelativistic(*package);
  const auto solver = static_cast<riemann::Solver>(package->Param<int>("riemann"));
  const auto reconstruction =
      static_cast<hydro::Reconstruction>(package->Param<int>("reconstruction"));
  const auto mode = static_cast<HydroMode>(package->Param<int>("physics_mode"));
  const auto geometry_package = block->packages.Get("geometry");
  const bool excision = geometry_package->Param<bool>("excision");
  // The disabled path retains the original GRMHD template instance and never
  // packs or writes optional passive fields.
  const auto unused = data->PackVariables(std::vector<std::string>{"mhd.fofc"});
  DispatchMHDFluxesBlock<false>(data, eos, excision, reconstruction, mode, solver, unused);
  return mhd::BuildCornerEMFBlockTask(data);
}

TaskStatus CalculateMHDFluxesWithPassiveBlockTask(std::shared_ptr<MeshBlockData<Real>>& data,
                                                  const std::string& passive_conserved_name) {
  auto* block = data->GetBlockPointer();
  const auto package = block->packages.Get("mhd");
  const auto eos = pangu::eos::ReadRelativistic(*package);
  const auto solver = static_cast<riemann::Solver>(package->Param<int>("riemann"));
  const auto reconstruction =
      static_cast<hydro::Reconstruction>(package->Param<int>("reconstruction"));
  const auto mode = static_cast<HydroMode>(package->Param<int>("physics_mode"));
  const auto geometry_package = block->packages.Get("geometry");
  const bool excision = geometry_package->Param<bool>("excision");
  auto transport = data->PackVariablesAndFluxes(std::vector<std::string>{passive_conserved_name});
  PARTHENON_REQUIRE(transport.GetDim(4) >= 2,
                    "passive transport requires at least two conserved components");
  DispatchMHDFluxesBlock<true>(data, eos, excision, reconstruction, mode, solver, transport);
  return mhd::BuildCornerEMFBlockTask(data);
}

TaskStatus CalculateMHDFluxesMeshTask(MeshData<Real>* data) {
  auto* mesh = data->GetParentPointer();
  const auto package = mesh->packages.Get("mhd");
  const auto eos = pangu::eos::ReadRelativistic(*package);
  const auto solver = static_cast<riemann::Solver>(package->Param<int>("riemann"));
  const auto reconstruction =
      static_cast<hydro::Reconstruction>(package->Param<int>("reconstruction"));
  const auto mode = static_cast<HydroMode>(package->Param<int>("physics_mode"));
  const auto geometry_package = mesh->packages.Get("geometry");
  const bool excision = geometry_package->Param<bool>("excision");
  PARTHENON_REQUIRE(mode != HydroMode::newtonian,
                    "Packed relativistic MHD flux dispatch requires SR or GR mode");
  const auto unused = data->PackVariables(std::vector<std::string>{"mhd.fofc"});
  DispatchMHDFluxesMesh<false>(data, eos, excision, reconstruction, mode, solver, unused);
  // The packed GRMHD driver applies FOFC next.  FOFC can replace face EMFs, so
  // constructing corner EMFs here would be redundant; the driver builds them
  // once, after all block-local corrections have completed.
  return TaskStatus::complete;
}

TaskStatus CalculateMHDFluxesWithPassiveMeshTask(MeshData<Real>* data,
                                                 const std::string& passive_conserved_name) {
  auto* mesh = data->GetParentPointer();
  const auto package = mesh->packages.Get("mhd");
  const auto eos = pangu::eos::ReadRelativistic(*package);
  const auto solver = static_cast<riemann::Solver>(package->Param<int>("riemann"));
  const auto reconstruction =
      static_cast<hydro::Reconstruction>(package->Param<int>("reconstruction"));
  const auto mode = static_cast<HydroMode>(package->Param<int>("physics_mode"));
  const auto geometry_package = mesh->packages.Get("geometry");
  const bool excision = geometry_package->Param<bool>("excision");
  PARTHENON_REQUIRE(mode != HydroMode::newtonian,
                    "Packed relativistic passive flux dispatch requires SR or GR mode");
  auto transport = data->PackVariablesAndFluxes(std::vector<std::string>{passive_conserved_name});
  PARTHENON_REQUIRE(transport.GetDim(4) >= 2,
                    "passive transport requires at least two conserved components");
  DispatchMHDFluxesMesh<true>(data, eos, excision, reconstruction, mode, solver, transport);
  return TaskStatus::complete;
}

template <bool WritePassiveTransport, typename TransportPack>
TaskStatus ApplyMHDFluxCorrectionBlockImpl(std::shared_ptr<MeshBlockData<Real>>& data,
                                           std::shared_ptr<MeshBlockData<Real>>& base,
                                           const Real gam0, const Real gam1, const Real beta_dt,
                                           const bool build_corner_emf,
                                           const TransportPack& transport) {
  const Real dt = beta_dt;
  auto block = data->GetBlockPointer();
  const auto package = block->packages.Get("mhd");
  const auto mode = static_cast<HydroMode>(package->Param<int>("physics_mode"));
  if (mode == HydroMode::newtonian)
    return TaskStatus::complete;
  const bool use_fofc = package->Param<bool>("fofc");
  const auto eos = pangu::eos::ReadMagnetised(*package);
  const Real gamma_max = package->Param<Real>("gamma_max");
  const Real sigma_max = package->Param<Real>("sigma_max");
  const auto geometry_package = block->packages.Get("geometry");
  const bool excision = geometry_package->Param<bool>("excision");
  const Real excision_radius = geometry_package->Param<Real>("excision_radius");
  auto conserved = data->PackVariablesAndFluxes(std::vector<std::string>{"mhd.cons"});
  const auto base_conserved = base->PackVariables(std::vector<std::string>{"mhd.cons"});
  auto primitive = data->PackVariables(std::vector<std::string>{"mhd.prim"});
  auto bcell = data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  auto flags = data->PackVariables(std::vector<std::string>{"mhd.fofc"});
  auto face_emf = data->PackVariables(std::vector<std::string>{"mhd.face_emf"});
  auto bface = data->Get("mhd.b_face").data;
  const auto base_face = base->Get("mhd.b_face").data;
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const int ndim = block->pmy_mesh->ndim;
  const auto coords = block->coords;
  const auto spacetime = geometry::GetGeometry(block->packages);
  const int geometry_block = block->lid;
  const IndexRange flag_i{ib.s - 1, ib.e + 1};
  const IndexRange flag_j{jb.s - (ndim >= 2), jb.e + (ndim >= 2)};
  const IndexRange flag_k{kb.s - (ndim >= 3), kb.e + (ndim >= 3)};
  block->par_for(
      "PANGU relativistic MHD FOFC candidate", flag_k.s, flag_k.e, flag_j.s, flag_j.e, flag_i.s,
      flag_i.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        HydroConservedState candidate{};
        Real values[5]{};
        for (int n = 0; n < 5; ++n) {
          values[n] =
              gam0 * conserved(n, k, j, i) + gam1 * base_conserved(n, k, j, i) -
              dt * (conserved.flux(X1DIR, n, k, j, i + 1) - conserved.flux(X1DIR, n, k, j, i)) /
                  coords.Dxc<X1DIR>(k, j, i);
          if (ndim >= 2)
            values[n] -=
                dt * (conserved.flux(X2DIR, n, k, j + 1, i) - conserved.flux(X2DIR, n, k, j, i)) /
                coords.Dxc<X2DIR>(k, j, i);
          if (ndim >= 3)
            values[n] -=
                dt * (conserved.flux(X3DIR, n, k + 1, j, i) - conserved.flux(X3DIR, n, k, j, i)) /
                coords.Dxc<X3DIR>(k, j, i);
        }
        candidate = {values[mhd::IDN],
                     {values[mhd::IM1], values[mhd::IM2], values[mhd::IM3]},
                     values[mhd::IEN]};
        Real magnetic[3]{
            gam0 * bcell(0, k, j, i) +
                0.5 * gam1 * (base_face(0, 0, 0, 0, k, j, i) + base_face(0, 0, 0, 0, k, j, i + 1)),
            gam0 * bcell(1, k, j, i) +
                0.5 * gam1 *
                    (base_face(1, 0, 0, 0, k, j, i) + base_face(1, 0, 0, 0, k, j + (ndim >= 2), i)),
            gam0 * bcell(2, k, j, i) + 0.5 * gam1 *
                                           (base_face(2, 0, 0, 0, k, j, i) +
                                            base_face(2, 0, 0, 0, k + (ndim >= 3), j, i))};
        Real magnetic_update_scale = 1.0;
        if (!decltype(spacetime)::unit_determinant) {
          const auto center =
              spacetime.MetricAt(geometry::Location::cell_center, geometry_block, k, j, i, coords);
          magnetic_update_scale = 1.0 / center.gdet;
          const auto x1_left =
              spacetime.MetricAt(geometry::Location::face1, geometry_block, k, j, i, coords);
          const auto x1_right =
              spacetime.MetricAt(geometry::Location::face1, geometry_block, k, j, i + 1, coords);
          const auto x2_left =
              spacetime.MetricAt(geometry::Location::face2, geometry_block, k, j, i, coords);
          const auto x2_right = spacetime.MetricAt(geometry::Location::face2, geometry_block, k,
                                                   j + (ndim >= 2), i, coords);
          const auto x3_left =
              spacetime.MetricAt(geometry::Location::face3, geometry_block, k, j, i, coords);
          const auto x3_right = spacetime.MetricAt(geometry::Location::face3, geometry_block,
                                                   k + (ndim >= 3), j, i, coords);
          magnetic[0] =
              gam0 * bcell(0, k, j, i) + 0.5 * gam1 *
                                             (base_face(0, 0, 0, 0, k, j, i) / x1_left.gdet +
                                              base_face(0, 0, 0, 0, k, j, i + 1) / x1_right.gdet);
          magnetic[1] = gam0 * bcell(1, k, j, i) +
                        0.5 * gam1 *
                            (base_face(1, 0, 0, 0, k, j, i) / x2_left.gdet +
                             base_face(1, 0, 0, 0, k, j + (ndim >= 2), i) / x2_right.gdet);
          magnetic[2] = gam0 * bcell(2, k, j, i) +
                        0.5 * gam1 *
                            (base_face(2, 0, 0, 0, k, j, i) / x3_left.gdet +
                             base_face(2, 0, 0, 0, k + (ndim >= 3), j, i) / x3_right.gdet);
        }
        magnetic[1] += dt * magnetic_update_scale *
                       (face_emf(E3X1, k, j, i + 1) - face_emf(E3X1, k, j, i)) /
                       coords.Dxc<X1DIR>(k, j, i);
        magnetic[2] -= dt * magnetic_update_scale *
                       (face_emf(E2X1, k, j, i + 1) - face_emf(E2X1, k, j, i)) /
                       coords.Dxc<X1DIR>(k, j, i);
        if (ndim >= 2) {
          magnetic[0] -= dt * magnetic_update_scale *
                         (face_emf(E3X2, k, j + 1, i) - face_emf(E3X2, k, j, i)) /
                         coords.Dxc<X2DIR>(k, j, i);
          magnetic[2] += dt * magnetic_update_scale *
                         (face_emf(E1X2, k, j + 1, i) - face_emf(E1X2, k, j, i)) /
                         coords.Dxc<X2DIR>(k, j, i);
        }
        if (ndim >= 3) {
          magnetic[0] += dt * magnetic_update_scale *
                         (face_emf(E2X3, k + 1, j, i) - face_emf(E2X3, k, j, i)) /
                         coords.Dxc<X3DIR>(k, j, i);
          magnetic[1] -= dt * magnetic_update_scale *
                         (face_emf(E1X3, k + 1, j, i) - face_emf(E1X3, k, j, i)) /
                         coords.Dxc<X3DIR>(k, j, i);
        }
        bool bad = false;
        if (use_fofc) {
          const auto metric =
              spacetime.MetricAt(geometry::Location::cell_center, geometry_block, k, j, i, coords);
          const auto trial =
              mode == HydroMode::sr
                  ? SolveSRMHDC2P(candidate, magnetic, eos, gamma_max, sigma_max)
                  : SolveGRMHDC2P(candidate, magnetic, eos, gamma_max, sigma_max, metric);
          bad = !trial.success || trial.density_floor || trial.pressure_floor ||
                trial.lorentz_ceiling || trial.sigma_ceiling;
        }
        if (mode == HydroMode::gr && excision) {
          bad = bad || spacetime.NeedsFluxExcision(coords, k, j, i, excision_radius);
        }
        flags(0, k, j, i) = bad ? 1.0 : 0.0;
      });
  ReplaceAllFlaggedFacesBlock<WritePassiveTransport>(block, primitive, bcell, conserved, bface,
                                                     face_emf, flags, mode, eos, flag_i, flag_j,
                                                     flag_k, ndim, transport);
  return build_corner_emf ? mhd::BuildCornerEMFBlockTask(data) : TaskStatus::complete;
}

TaskStatus ApplyMHDFluxCorrectionBlockTask(std::shared_ptr<MeshBlockData<Real>>& data,
                                           std::shared_ptr<MeshBlockData<Real>>& base,
                                           const Real gam0, const Real gam1, const Real beta_dt) {
  const auto unused = data->PackVariables(std::vector<std::string>{"mhd.fofc"});
  return ApplyMHDFluxCorrectionBlockImpl<false>(data, base, gam0, gam1, beta_dt, true, unused);
}

TaskStatus ApplyMHDFluxCorrectionWithPassiveBlockTask(std::shared_ptr<MeshBlockData<Real>>& data,
                                                      std::shared_ptr<MeshBlockData<Real>>& base,
                                                      const Real gam0, const Real gam1,
                                                      const Real beta_dt,
                                                      const std::string& passive_conserved_name) {
  const auto transport =
      data->PackVariablesAndFluxes(std::vector<std::string>{passive_conserved_name});
  return ApplyMHDFluxCorrectionBlockImpl<true>(data, base, gam0, gam1, beta_dt, true, transport);
}

TaskStatus ApplyMHDFluxCorrectionNoCornerBlockTask(std::shared_ptr<MeshBlockData<Real>>& data,
                                                   std::shared_ptr<MeshBlockData<Real>>& base,
                                                   const Real gam0, const Real gam1,
                                                   const Real beta_dt) {
  const auto unused = data->PackVariables(std::vector<std::string>{"mhd.fofc"});
  return ApplyMHDFluxCorrectionBlockImpl<false>(data, base, gam0, gam1, beta_dt, false, unused);
}

template <bool WritePassiveTransport>
TaskStatus ApplyMHDFluxCorrectionMeshImpl(MeshData<Real>* data, MeshData<Real>* base,
                                          const Real gam0, const Real gam1, const Real beta_dt,
                                          const std::string& passive_conserved_name) {
  using TE = parthenon::TopologicalElement;
  const Real dt = beta_dt;
  auto* mesh = data->GetParentPointer();
  const auto package = mesh->packages.Get("mhd");
  const auto mode = static_cast<HydroMode>(package->Param<int>("physics_mode"));
  if (mode == HydroMode::newtonian)
    return TaskStatus::complete;
  const bool use_fofc = package->Param<bool>("fofc");
  const auto eos = pangu::eos::ReadMagnetised(*package);
  const Real gamma_max = package->Param<Real>("gamma_max");
  const Real sigma_max = package->Param<Real>("sigma_max");
  const auto geometry_package = mesh->packages.Get("geometry");
  const bool excision = geometry_package->Param<bool>("excision");
  const Real excision_radius = geometry_package->Param<Real>("excision_radius");
  // With neither first-order correction nor geometric excision enabled there
  // are no faces to replace.  Avoid entering the compatibility MeshBlock
  // replacement path at all; this is the standard MKS/static KHARMA-style
  // configuration.
  if (!use_fofc && !(mode == HydroMode::gr && excision))
    return TaskStatus::complete;
  auto conserved = data->PackVariablesAndFluxes(std::vector<std::string>{"mhd.cons"});
  const auto base_conserved = base->PackVariables(std::vector<std::string>{"mhd.cons"});
  auto primitive = data->PackVariables(std::vector<std::string>{"mhd.prim"});
  auto bcell = data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  auto flags = data->PackVariables(std::vector<std::string>{"mhd.fofc"});
  auto face_emf = data->PackVariables(std::vector<std::string>{"mhd.face_emf"});
  auto bface = data->PackVariables(std::vector<std::string>{"mhd.b_face"});
  const auto base_face = base->PackVariables(std::vector<std::string>{"mhd.b_face"});
  const auto ib = data->GetBoundsI(IndexDomain::interior);
  const auto jb = data->GetBoundsJ(IndexDomain::interior);
  const auto kb = data->GetBoundsK(IndexDomain::interior);
  const int ndim = mesh->ndim;
  const auto spacetime = geometry::GetGeometry(mesh->packages);
  const int geometry_block_offset = data->GetBlockData(0)->GetBlockPointer()->lid;
  const IndexRange flag_i{ib.s - 1, ib.e + 1};
  const IndexRange flag_j{jb.s - (ndim >= 2), jb.e + (ndim >= 2)};
  const IndexRange flag_k{kb.s - (ndim >= 3), kb.e + (ndim >= 3)};
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU packed relativistic MHD FOFC candidate",
      parthenon::DevExecSpace(), 0, data->NumBlocks() - 1, flag_k.s, flag_k.e, flag_j.s, flag_j.e,
      flag_i.s, flag_i.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto& coords = conserved.GetCoords(b);
        const auto block_conserved = conserved(b);
        HydroConservedState candidate{};
        Real values[5]{};
        for (int n = 0; n < 5; ++n) {
          values[n] = gam0 * conserved(b, n, k, j, i) + gam1 * base_conserved(b, n, k, j, i) -
                      dt *
                          (block_conserved.flux(X1DIR, n, k, j, i + 1) -
                           block_conserved.flux(X1DIR, n, k, j, i)) /
                          coords.Dxc<X1DIR>(k, j, i);
          if (ndim >= 2)
            values[n] -= dt *
                         (block_conserved.flux(X2DIR, n, k, j + 1, i) -
                          block_conserved.flux(X2DIR, n, k, j, i)) /
                         coords.Dxc<X2DIR>(k, j, i);
          if (ndim >= 3)
            values[n] -= dt *
                         (block_conserved.flux(X3DIR, n, k + 1, j, i) -
                          block_conserved.flux(X3DIR, n, k, j, i)) /
                         coords.Dxc<X3DIR>(k, j, i);
        }
        candidate = {values[mhd::IDN],
                     {values[mhd::IM1], values[mhd::IM2], values[mhd::IM3]},
                     values[mhd::IEN]};
        Real magnetic[3]{
            gam0 * bcell(b, 0, k, j, i) +
                0.5 * gam1 *
                    (base_face(b, TE::F1, 0, k, j, i) + base_face(b, TE::F1, 0, k, j, i + 1)),
            gam0 * bcell(b, 1, k, j, i) + 0.5 * gam1 *
                                              (base_face(b, TE::F2, 0, k, j, i) +
                                               base_face(b, TE::F2, 0, k, j + (ndim >= 2), i)),
            gam0 * bcell(b, 2, k, j, i) + 0.5 * gam1 *
                                              (base_face(b, TE::F3, 0, k, j, i) +
                                               base_face(b, TE::F3, 0, k + (ndim >= 3), j, i))};
        Real magnetic_update_scale = 1.0;
        if (!decltype(spacetime)::unit_determinant) {
          const auto center = spacetime.MetricAt(geometry::Location::cell_center,
                                                 geometry_block_offset + b, k, j, i, coords);
          magnetic_update_scale = 1.0 / center.gdet;
          const auto x1_left = spacetime.MetricAt(geometry::Location::face1,
                                                  geometry_block_offset + b, k, j, i, coords);
          const auto x1_right = spacetime.MetricAt(geometry::Location::face1,
                                                   geometry_block_offset + b, k, j, i + 1, coords);
          const auto x2_left = spacetime.MetricAt(geometry::Location::face2,
                                                  geometry_block_offset + b, k, j, i, coords);
          const auto x2_right = spacetime.MetricAt(
              geometry::Location::face2, geometry_block_offset + b, k, j + (ndim >= 2), i, coords);
          const auto x3_left = spacetime.MetricAt(geometry::Location::face3,
                                                  geometry_block_offset + b, k, j, i, coords);
          const auto x3_right = spacetime.MetricAt(
              geometry::Location::face3, geometry_block_offset + b, k + (ndim >= 3), j, i, coords);
          magnetic[0] = gam0 * bcell(b, 0, k, j, i) +
                        0.5 * gam1 *
                            (base_face(b, TE::F1, 0, k, j, i) / x1_left.gdet +
                             base_face(b, TE::F1, 0, k, j, i + 1) / x1_right.gdet);
          magnetic[1] = gam0 * bcell(b, 1, k, j, i) +
                        0.5 * gam1 *
                            (base_face(b, TE::F2, 0, k, j, i) / x2_left.gdet +
                             base_face(b, TE::F2, 0, k, j + (ndim >= 2), i) / x2_right.gdet);
          magnetic[2] = gam0 * bcell(b, 2, k, j, i) +
                        0.5 * gam1 *
                            (base_face(b, TE::F3, 0, k, j, i) / x3_left.gdet +
                             base_face(b, TE::F3, 0, k + (ndim >= 3), j, i) / x3_right.gdet);
        }
        magnetic[1] += dt * magnetic_update_scale *
                       (face_emf(b, E3X1, k, j, i + 1) - face_emf(b, E3X1, k, j, i)) /
                       coords.Dxc<X1DIR>(k, j, i);
        magnetic[2] -= dt * magnetic_update_scale *
                       (face_emf(b, E2X1, k, j, i + 1) - face_emf(b, E2X1, k, j, i)) /
                       coords.Dxc<X1DIR>(k, j, i);
        if (ndim >= 2) {
          magnetic[0] -= dt * magnetic_update_scale *
                         (face_emf(b, E3X2, k, j + 1, i) - face_emf(b, E3X2, k, j, i)) /
                         coords.Dxc<X2DIR>(k, j, i);
          magnetic[2] += dt * magnetic_update_scale *
                         (face_emf(b, E1X2, k, j + 1, i) - face_emf(b, E1X2, k, j, i)) /
                         coords.Dxc<X2DIR>(k, j, i);
        }
        if (ndim >= 3) {
          magnetic[0] += dt * magnetic_update_scale *
                         (face_emf(b, E2X3, k + 1, j, i) - face_emf(b, E2X3, k, j, i)) /
                         coords.Dxc<X3DIR>(k, j, i);
          magnetic[1] -= dt * magnetic_update_scale *
                         (face_emf(b, E1X3, k + 1, j, i) - face_emf(b, E1X3, k, j, i)) /
                         coords.Dxc<X3DIR>(k, j, i);
        }
        bool bad = false;
        if (use_fofc) {
          const auto metric = spacetime.MetricAt(geometry::Location::cell_center,
                                                 geometry_block_offset + b, k, j, i, coords);
          const auto trial =
              mode == HydroMode::sr
                  ? SolveSRMHDC2P(candidate, magnetic, eos, gamma_max, sigma_max)
                  : SolveGRMHDC2P(candidate, magnetic, eos, gamma_max, sigma_max, metric);
          bad = !trial.success || trial.density_floor || trial.pressure_floor ||
                trial.lorentz_ceiling || trial.sigma_ceiling;
        }
        if (mode == HydroMode::gr && excision)
          bad = bad || spacetime.NeedsFluxExcision(coords, k, j, i, excision_radius);
        flags(b, 0, k, j, i) = bad ? 1.0 : 0.0;
      });
  // Keep the corrected-face arithmetic in its original VariablePack shape.
  // The expensive candidate inversion is pack-wide above; these lightweight
  // block kernels preserve bitwise identity at the excision transition.
  for (int b = 0; b < data->NumBlocks(); ++b) {
    auto block_data = data->GetBlockData(b);
    auto* block = block_data->GetBlockPointer();
    auto block_primitive = block_data->PackVariables(std::vector<std::string>{"mhd.prim"});
    auto block_bcell = block_data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
    auto block_conserved = block_data->PackVariablesAndFluxes(std::vector<std::string>{"mhd.cons"});
    auto block_bface = block_data->Get("mhd.b_face").data;
    auto block_face_emf = block_data->PackVariables(std::vector<std::string>{"mhd.face_emf"});
    auto block_flags = block_data->PackVariables(std::vector<std::string>{"mhd.fofc"});
    if constexpr (WritePassiveTransport) {
      auto block_transport =
          block_data->PackVariablesAndFluxes(std::vector<std::string>{passive_conserved_name});
      ReplaceAllFlaggedFacesBlock<true>(block, block_primitive, block_bcell, block_conserved,
                                        block_bface, block_face_emf, block_flags, mode, eos,
                                        flag_i, flag_j, flag_k, ndim, block_transport);
    } else {
      ReplaceAllFlaggedFacesBlock<false>(block, block_primitive, block_bcell, block_conserved,
                                         block_bface, block_face_emf, block_flags, mode, eos,
                                         flag_i, flag_j, flag_k, ndim, block_flags);
    }
  }
  return TaskStatus::complete;
}

TaskStatus ApplyMHDFluxCorrectionMeshTask(MeshData<Real>* data, MeshData<Real>* base,
                                          const Real gam0, const Real gam1, const Real beta_dt) {
  return ApplyMHDFluxCorrectionMeshImpl<false>(data, base, gam0, gam1, beta_dt, "");
}

TaskStatus ApplyMHDFluxCorrectionWithPassiveMeshTask(MeshData<Real>* data, MeshData<Real>* base,
                                                     const Real gam0, const Real gam1,
                                                     const Real beta_dt,
                                                     const std::string& passive_conserved_name) {
  return ApplyMHDFluxCorrectionMeshImpl<true>(data, base, gam0, gam1, beta_dt,
                                              passive_conserved_name);
}

void MHDConservedToPrimitiveBlock(MeshBlockData<Real>* data) {
  auto block = data->GetBlockPointer();
  const auto package = block->packages.Get("mhd");
  const auto mode = static_cast<HydroMode>(package->Param<int>("physics_mode"));
  const auto geometry_package = block->packages.Get("geometry");
  const bool excision = geometry_package->Param<bool>("excision");
  const Real excision_radius = geometry_package->Param<Real>("excision_radius");
  const auto eos = pangu::eos::ReadMagnetised(*package);
  const Real configured_excision_density = geometry_package->Param<Real>("excision_density");
  const Real configured_excision_pressure = geometry_package->Param<Real>("excision_pressure");
  const Real excision_density =
      configured_excision_density > 0.0 ? configured_excision_density : eos.density_floor;
  const Real excision_pressure =
      configured_excision_pressure > 0.0 ? configured_excision_pressure : eos.pressure_floor;
  const Real gamma_max = package->Param<Real>("gamma_max");
  const Real sigma_max = package->Param<Real>("sigma_max");
  auto conserved = data->PackVariables(std::vector<std::string>{"mhd.cons"});
  auto primitive = data->PackVariables(std::vector<std::string>{"mhd.prim"});
  auto bcell = data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  auto divb = data->PackVariables(std::vector<std::string>{"mhd.divb"});
  auto flags = data->PackVariables(std::vector<std::string>{"mhd.fofc"});
  auto bface = data->Get("mhd.b_face").data;
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::entire);
  const auto coords = block->coords;
  const auto spacetime = geometry::GetGeometry(block->packages);
  const int geometry_block = block->lid;
  const int ndim = block->pmy_mesh->ndim;
  if (mode == HydroMode::sr) {
    // Keep SR and GR inversion in separate device kernels.  A runtime mode
    // branch makes the SR kernel carry all metric, excision, and GR repair
    // code, increasing register pressure in the dominant SRMHD kernel.
    block->par_for(
        "PANGU SR MHD C2P", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          Real magnetic[3]{
              0.5 * (bface(0, 0, 0, 0, k, j, i) + bface(0, 0, 0, 0, k, j, i + 1)),
              0.5 * (bface(1, 0, 0, 0, k, j, i) + bface(1, 0, 0, 0, k, j + (ndim >= 2), i)),
              0.5 * (bface(2, 0, 0, 0, k, j, i) + bface(2, 0, 0, 0, k + (ndim >= 3), j, i))};
          HydroConservedState state{conserved(mhd::IDN, k, j, i),
                                    {conserved(mhd::IM1, k, j, i), conserved(mhd::IM2, k, j, i),
                                     conserved(mhd::IM3, k, j, i)},
                                    conserved(mhd::IEN, k, j, i)};
          auto result = SolveSRMHDC2P(state, magnetic, eos, gamma_max, sigma_max);
          primitive(mhd::IDN, k, j, i) = result.primitive.fluid.density;
          primitive(mhd::IV1, k, j, i) = result.primitive.fluid.u[0];
          primitive(mhd::IV2, k, j, i) = result.primitive.fluid.u[1];
          primitive(mhd::IV3, k, j, i) = result.primitive.fluid.u[2];
          primitive(mhd::IPR, k, j, i) = result.internal_energy;
          for (int axis = 0; axis < 3; ++axis)
            bcell(axis, k, j, i) = magnetic[axis];
          Real divergence = (bface(0, 0, 0, 0, k, j, i + 1) - bface(0, 0, 0, 0, k, j, i)) /
                            coords.Dxc<X1DIR>(k, j, i);
          if (ndim >= 2)
            divergence += (bface(1, 0, 0, 0, k, j + 1, i) - bface(1, 0, 0, 0, k, j, i)) /
                          coords.Dxc<X2DIR>(k, j, i);
          if (ndim >= 3)
            divergence += (bface(2, 0, 0, 0, k + 1, j, i) - bface(2, 0, 0, 0, k, j, i)) /
                          coords.Dxc<X3DIR>(k, j, i);
          divb(0, k, j, i) = divergence;
          const bool repaired = !result.success || result.density_floor || result.pressure_floor ||
                                result.lorentz_ceiling || result.sigma_ceiling;
          flags(0, k, j, i) = repaired ? 1.0 : 0.0;
          if (repaired) {
            const auto fixed =
                ConvertSRMHDP2CWithInternalEnergy(result.primitive, result.internal_energy, eos);
            conserved(mhd::IDN, k, j, i) = fixed.density;
            conserved(mhd::IM1, k, j, i) = fixed.momentum[0];
            conserved(mhd::IM2, k, j, i) = fixed.momentum[1];
            conserved(mhd::IM3, k, j, i) = fixed.momentum[2];
            conserved(mhd::IEN, k, j, i) = fixed.energy;
          }
        });
    return;
  }
  // AthenaK's GR inversion uses a one-dimensional RangePolicy.  Flatten this
  // kernel identically: besides reducing launch-policy overhead, it keeps the
  // highly magnetized C2P expression tree in the same CUDA compilation shape.
  const int ni = ib.e - ib.s + 1;
  const int nji = (jb.e - jb.s + 1) * ni;
  const int nkji = (kb.e - kb.s + 1) * nji;
  block->par_for(
      "PANGU GR MHD C2P", 0, nkji - 1, KOKKOS_LAMBDA(const int idx) {
        const int k = idx / nji + kb.s;
        const int j = (idx % nji) / ni + jb.s;
        const int i = idx % ni + ib.s;
        Real magnetic[3]{};
        if (decltype(spacetime)::unit_determinant) {
          magnetic[0] = 0.5 * (bface(0, 0, 0, 0, k, j, i) + bface(0, 0, 0, 0, k, j, i + 1));
          magnetic[1] =
              0.5 * (bface(1, 0, 0, 0, k, j, i) + bface(1, 0, 0, 0, k, j + (ndim >= 2), i));
          magnetic[2] =
              0.5 * (bface(2, 0, 0, 0, k, j, i) + bface(2, 0, 0, 0, k + (ndim >= 3), j, i));
        } else {
          const auto metric_x1_left =
              spacetime.MetricAt(geometry::Location::face1, geometry_block, k, j, i, coords);
          const auto metric_x1_right =
              spacetime.MetricAt(geometry::Location::face1, geometry_block, k, j, i + 1, coords);
          const auto metric_x2_left =
              spacetime.MetricAt(geometry::Location::face2, geometry_block, k, j, i, coords);
          const auto metric_x2_right = spacetime.MetricAt(geometry::Location::face2, geometry_block,
                                                          k, j + (ndim >= 2), i, coords);
          const auto metric_x3_left =
              spacetime.MetricAt(geometry::Location::face3, geometry_block, k, j, i, coords);
          const auto metric_x3_right = spacetime.MetricAt(geometry::Location::face3, geometry_block,
                                                          k + (ndim >= 3), j, i, coords);
          magnetic[0] = 0.5 * (bface(0, 0, 0, 0, k, j, i) / metric_x1_left.gdet +
                               bface(0, 0, 0, 0, k, j, i + 1) / metric_x1_right.gdet);
          magnetic[1] = 0.5 * (bface(1, 0, 0, 0, k, j, i) / metric_x2_left.gdet +
                               bface(1, 0, 0, 0, k, j + (ndim >= 2), i) / metric_x2_right.gdet);
          magnetic[2] = 0.5 * (bface(2, 0, 0, 0, k, j, i) / metric_x3_left.gdet +
                               bface(2, 0, 0, 0, k + (ndim >= 3), j, i) / metric_x3_right.gdet);
        }
        HydroConservedState state{conserved(mhd::IDN, k, j, i),
                                  {conserved(mhd::IM1, k, j, i), conserved(mhd::IM2, k, j, i),
                                   conserved(mhd::IM3, k, j, i)},
                                  conserved(mhd::IEN, k, j, i)};
        const Real x = coords.Xc<1>(i);
        const Real y = coords.Xc<2>(j);
        const Real z = coords.Xc<3>(k);
        const auto metric =
            spacetime.MetricAt(geometry::Location::cell_center, geometry_block, k, j, i, coords);
        auto result = SolveGRMHDC2P(state, magnetic, eos, gamma_max, sigma_max, metric);
        if (excision && spacetime.IsExcised(x, y, z, excision_radius)) {
          result.primitive.fluid = {excision_density, {0.0, 0.0, 0.0}, excision_pressure};
          result.internal_energy = eos.InternalEnergyDensity(excision_pressure);
          result.density_floor = true;
          result.pressure_floor = true;
        }
        primitive(mhd::IDN, k, j, i) = result.primitive.fluid.density;
        primitive(mhd::IV1, k, j, i) = result.primitive.fluid.u[0];
        primitive(mhd::IV2, k, j, i) = result.primitive.fluid.u[1];
        primitive(mhd::IV3, k, j, i) = result.primitive.fluid.u[2];
        primitive(mhd::IPR, k, j, i) = result.internal_energy;
        for (int axis = 0; axis < 3; ++axis)
          bcell(axis, k, j, i) = magnetic[axis];
        Real divergence = (bface(0, 0, 0, 0, k, j, i + 1) - bface(0, 0, 0, 0, k, j, i)) /
                          coords.Dxc<X1DIR>(k, j, i);
        if (ndim >= 2)
          divergence += (bface(1, 0, 0, 0, k, j + 1, i) - bface(1, 0, 0, 0, k, j, i)) /
                        coords.Dxc<X2DIR>(k, j, i);
        if (ndim >= 3)
          divergence += (bface(2, 0, 0, 0, k + 1, j, i) - bface(2, 0, 0, 0, k, j, i)) /
                        coords.Dxc<X3DIR>(k, j, i);
        divb(0, k, j, i) = divergence / metric.gdet;
        const bool repaired = !result.success || result.density_floor || result.pressure_floor ||
                              result.lorentz_ceiling || result.sigma_ceiling;
        flags(0, k, j, i) = repaired ? 1.0 : 0.0;
        if (repaired) {
          const auto fixed = ConvertGRMHDP2CWithInternalEnergy(
              result.primitive, result.internal_energy, eos, metric);
          conserved(mhd::IDN, k, j, i) = fixed.density;
          conserved(mhd::IM1, k, j, i) = fixed.momentum[0];
          conserved(mhd::IM2, k, j, i) = fixed.momentum[1];
          conserved(mhd::IM3, k, j, i) = fixed.momentum[2];
          conserved(mhd::IEN, k, j, i) = fixed.energy;
        }
      });
}

void MHDConservedToPrimitiveMesh(MeshData<Real>* data) {
  auto* mesh = data->GetParentPointer();
  const auto package = mesh->packages.Get("mhd");
  const auto mode = static_cast<HydroMode>(package->Param<int>("physics_mode"));
  if (mode != HydroMode::gr) {
    for (int b = 0; b < data->NumBlocks(); ++b)
      MHDConservedToPrimitiveBlock(data->GetBlockData(b).get());
    return;
  }
  const auto geometry_package = mesh->packages.Get("geometry");
  const bool excision = geometry_package->Param<bool>("excision");
  const Real excision_radius = geometry_package->Param<Real>("excision_radius");
  const auto eos = pangu::eos::ReadMagnetised(*package);
  const Real configured_excision_density = geometry_package->Param<Real>("excision_density");
  const Real configured_excision_pressure = geometry_package->Param<Real>("excision_pressure");
  const Real excision_density =
      configured_excision_density > 0.0 ? configured_excision_density : eos.density_floor;
  const Real excision_pressure =
      configured_excision_pressure > 0.0 ? configured_excision_pressure : eos.pressure_floor;
  const Real gamma_max = package->Param<Real>("gamma_max");
  const Real sigma_max = package->Param<Real>("sigma_max");
  const int ndim = mesh->ndim;
  const auto spacetime = geometry::GetGeometry(mesh->packages);
  const int geometry_block_offset = data->GetBlockData(0)->GetBlockPointer()->lid;

  auto bcell = data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  auto divb = data->PackVariables(std::vector<std::string>{"mhd.divb"});
  auto bface = data->PackVariables(std::vector<std::string>{"mhd.b_face"});
  const auto ib = data->GetBoundsI(IndexDomain::entire);
  const auto jb = data->GetBoundsJ(IndexDomain::entire);
  const auto kb = data->GetBoundsK(IndexDomain::entire);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU packed GR MHD magnetic derived", parthenon::DevExecSpace(), 0,
      data->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto& coords = bcell.GetCoords(b);
        const int geometry_block = geometry_block_offset + b;
        Real magnetic[3]{};
        if (decltype(spacetime)::unit_determinant) {
          magnetic[0] =
              0.5 * (bface(b, parthenon::TopologicalElement::F1, 0, k, j, i) +
                     bface(b, parthenon::TopologicalElement::F1, 0, k, j, i + 1));
          magnetic[1] =
              0.5 * (bface(b, parthenon::TopologicalElement::F2, 0, k, j, i) +
                     bface(b, parthenon::TopologicalElement::F2, 0, k, j + (ndim >= 2), i));
          magnetic[2] =
              0.5 * (bface(b, parthenon::TopologicalElement::F3, 0, k, j, i) +
                     bface(b, parthenon::TopologicalElement::F3, 0, k + (ndim >= 3), j, i));
        } else {
          const auto metric_x1_left =
              spacetime.MetricAt(geometry::Location::face1, geometry_block, k, j, i, coords);
          const auto metric_x1_right =
              spacetime.MetricAt(geometry::Location::face1, geometry_block, k, j, i + 1, coords);
          const auto metric_x2_left =
              spacetime.MetricAt(geometry::Location::face2, geometry_block, k, j, i, coords);
          const auto metric_x2_right = spacetime.MetricAt(
              geometry::Location::face2, geometry_block, k, j + (ndim >= 2), i, coords);
          const auto metric_x3_left =
              spacetime.MetricAt(geometry::Location::face3, geometry_block, k, j, i, coords);
          const auto metric_x3_right = spacetime.MetricAt(
              geometry::Location::face3, geometry_block, k + (ndim >= 3), j, i, coords);
          magnetic[0] =
              0.5 * (bface(b, parthenon::TopologicalElement::F1, 0, k, j, i) /
                         metric_x1_left.gdet +
                     bface(b, parthenon::TopologicalElement::F1, 0, k, j, i + 1) /
                         metric_x1_right.gdet);
          magnetic[1] =
              0.5 * (bface(b, parthenon::TopologicalElement::F2, 0, k, j, i) /
                         metric_x2_left.gdet +
                     bface(b, parthenon::TopologicalElement::F2, 0, k, j + (ndim >= 2), i) /
                         metric_x2_right.gdet);
          magnetic[2] =
              0.5 * (bface(b, parthenon::TopologicalElement::F3, 0, k, j, i) /
                         metric_x3_left.gdet +
                     bface(b, parthenon::TopologicalElement::F3, 0, k + (ndim >= 3), j, i) /
                         metric_x3_right.gdet);
        }
        for (int axis = 0; axis < 3; ++axis)
          bcell(b, axis, k, j, i) = magnetic[axis];
        Real divergence =
            (bface(b, parthenon::TopologicalElement::F1, 0, k, j, i + 1) -
             bface(b, parthenon::TopologicalElement::F1, 0, k, j, i)) /
            coords.Dxc<X1DIR>(k, j, i);
        if (ndim >= 2)
          divergence +=
              (bface(b, parthenon::TopologicalElement::F2, 0, k, j + 1, i) -
               bface(b, parthenon::TopologicalElement::F2, 0, k, j, i)) /
              coords.Dxc<X2DIR>(k, j, i);
        if (ndim >= 3)
          divergence +=
              (bface(b, parthenon::TopologicalElement::F3, 0, k + 1, j, i) -
               bface(b, parthenon::TopologicalElement::F3, 0, k, j, i)) /
              coords.Dxc<X3DIR>(k, j, i);
        if (decltype(spacetime)::unit_determinant) {
          divb(b, 0, k, j, i) = divergence;
        } else {
          const auto metric = spacetime.MetricAt(geometry::Location::cell_center, geometry_block,
                                                 k, j, i, coords);
          divb(b, 0, k, j, i) = divergence / metric.gdet;
        }
      });

  auto conserved = data->PackVariables(std::vector<std::string>{"mhd.cons"});
  auto primitive = data->PackVariables(std::vector<std::string>{"mhd.prim"});
  auto flags = data->PackVariables(std::vector<std::string>{"mhd.fofc"});
  const int ni = ib.e - ib.s + 1;
  const int nji = (jb.e - jb.s + 1) * ni;
  const int nkji = (kb.e - kb.s + 1) * nji;
  const int total = data->NumBlocks() * nkji;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU packed GR MHD inversion", parthenon::DevExecSpace(), 0,
      total - 1, KOKKOS_LAMBDA(const int idx) {
        const int b = idx / nkji;
        const int local = idx % nkji;
        const int k = local / nji + kb.s;
        const int j = (local % nji) / ni + jb.s;
        const int i = local % ni + ib.s;
        Real magnetic[3]{bcell(b, 0, k, j, i), bcell(b, 1, k, j, i), bcell(b, 2, k, j, i)};
        HydroConservedState state{conserved(b, mhd::IDN, k, j, i),
                                  {conserved(b, mhd::IM1, k, j, i), conserved(b, mhd::IM2, k, j, i),
                                   conserved(b, mhd::IM3, k, j, i)},
                                  conserved(b, mhd::IEN, k, j, i)};
        const auto& coords = conserved.GetCoords(b);
        const Real x = coords.Xc<1>(i);
        const Real y = coords.Xc<2>(j);
        const Real z = coords.Xc<3>(k);
        const auto metric = spacetime.MetricAt(geometry::Location::cell_center,
                                               geometry_block_offset + b, k, j, i, coords);
        auto result = SolveGRMHDC2P(state, magnetic, eos, gamma_max, sigma_max, metric);
        if (excision && spacetime.IsExcised(x, y, z, excision_radius)) {
          result.primitive.fluid = {excision_density, {0.0, 0.0, 0.0}, excision_pressure};
          result.internal_energy = eos.InternalEnergyDensity(excision_pressure);
          result.density_floor = true;
          result.pressure_floor = true;
        }
        primitive(b, mhd::IDN, k, j, i) = result.primitive.fluid.density;
        primitive(b, mhd::IV1, k, j, i) = result.primitive.fluid.u[0];
        primitive(b, mhd::IV2, k, j, i) = result.primitive.fluid.u[1];
        primitive(b, mhd::IV3, k, j, i) = result.primitive.fluid.u[2];
        primitive(b, mhd::IPR, k, j, i) = result.internal_energy;
        const bool repaired = !result.success || result.density_floor || result.pressure_floor ||
                              result.lorentz_ceiling || result.sigma_ceiling;
        flags(b, 0, k, j, i) = repaired ? 1.0 : 0.0;
        if (repaired) {
          const auto fixed = ConvertGRMHDP2CWithInternalEnergy(
              result.primitive, result.internal_energy, eos, metric);
          conserved(b, mhd::IDN, k, j, i) = fixed.density;
          conserved(b, mhd::IM1, k, j, i) = fixed.momentum[0];
          conserved(b, mhd::IM2, k, j, i) = fixed.momentum[1];
          conserved(b, mhd::IM3, k, j, i) = fixed.momentum[2];
          conserved(b, mhd::IEN, k, j, i) = fixed.energy;
        }
      });
}

Real EstimateMHDTimestepBlock(MeshBlockData<Real>* data) {
  auto block = data->GetBlockPointer();
  const auto package = block->packages.Get("mhd");
  const auto mode = static_cast<HydroMode>(package->Param<int>("physics_mode"));
  const Real cfl = package->Param<Real>("cfl");
  const auto eos = pangu::eos::ReadRelativistic(*package);
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto coords = block->coords;
  const auto spacetime = geometry::GetGeometry(block->packages);
  const int geometry_block = block->lid;
  const int ndim = block->pmy_mesh->ndim;
  if constexpr (Estimator::face_signal_speeds) {
    const auto signal_max = data->PackVariables(std::vector<std::string>{"mhd.cmax"});
    const auto signal_min = data->PackVariables(std::vector<std::string>{"mhd.cmin"});
    Real minimum = std::numeric_limits<Real>::max();
    ParReduce(
        "PANGU face-speed MHD timestep", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i, Real& local) {
          Real estimate = Estimator::Start();
          for (int direction = 0; direction < ndim; ++direction) {
            const Real speed = fmax(signal_max(direction, k, j, i), signal_min(direction, k, j, i));
            if (isfinite(speed) && speed > 0.0)
              estimate = Estimator::Add(estimate, speed, coords.Dxc(direction + 1, k, j, i));
          }
          local = fmin(local, Estimator::Finish(estimate));
        },
        Kokkos::Min<Real>(minimum));
    if (minimum < std::numeric_limits<Real>::max())
      return cfl * minimum;
    // Before the first Riemann solve the face cache is zero; seed the
    // timestep once from cell-centred characteristics below.
  }
  if (!Estimator::characteristic_gr_speeds && mode == HydroMode::gr)
    return cfl * estimator::UnitSpeedTimestep(coords, kb.s, jb.s, ib.s, ndim);
  auto primitive = data->PackVariables(std::vector<std::string>{"mhd.prim"});
  auto magnetic = data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  Real minimum = std::numeric_limits<Real>::max();
  ParReduce(
      "PANGU relativistic MHD timestep", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, Real& local) {
        MHDPrimitiveState state{
            {primitive(mhd::IDN, k, j, i),
             {primitive(mhd::IV1, k, j, i), primitive(mhd::IV2, k, j, i),
              primitive(mhd::IV3, k, j, i)},
             eos.PressureFromInternalEnergyDensity(primitive(mhd::IPR, k, j, i))},
            {magnetic(0, k, j, i), magnetic(1, k, j, i), magnetic(2, k, j, i)}};
        Real estimate = Estimator::Start();
        for (int direction = 0; direction < ndim; ++direction) {
          Real speed;
          if (mode == HydroMode::sr) {
            const auto physical = BuildSRMHDFluxState(state, eos, direction);
            speed = fmax(fabs(physical.lambda_plus), fabs(physical.lambda_minus));
          } else {
            const auto metric = spacetime.MetricAt(geometry::Location::cell_center,
                                                   geometry_block, k, j, i, coords);
            const auto physical = BuildGRMHDFluxState(state, eos, direction, metric);
            speed = fmax(fabs(physical.lambda_plus), fabs(physical.lambda_minus));
            // Fast characteristics must remain inside the coordinate light
            // cone.  This guard only protects the timestep estimator from
            // a near-horizon quadratic roundoff; it does not alter fluxes.
            Real light_plus = 0.0, light_minus = 0.0;
            ComputeCoordinateLightSpeeds(metric, direction, light_plus, light_minus);
            speed = fmin(speed, fmax(fabs(light_plus), fabs(light_minus)));
          }
          estimate = Estimator::Add(estimate, speed, coords.Dxc(direction + 1, k, j, i));
        }
        local = fmin(local, Estimator::Finish(estimate));
      },
      Kokkos::Min<Real>(minimum));
  return cfl * minimum;
}

Real EstimateMHDTimestepMesh(MeshData<Real>* data) {
  auto* mesh = data->GetParentPointer();
  const auto package = mesh->packages.Get("mhd");
  const auto mode = static_cast<HydroMode>(package->Param<int>("physics_mode"));
  const Real cfl = package->Param<Real>("cfl");
  const auto eos = pangu::eos::ReadRelativistic(*package);
  const auto ib = data->GetBoundsI(IndexDomain::interior);
  const auto jb = data->GetBoundsJ(IndexDomain::interior);
  const auto kb = data->GetBoundsK(IndexDomain::interior);
  const int ndim = mesh->ndim;
  if constexpr (Estimator::face_signal_speeds) {
    const auto signal_max = data->PackVariables(std::vector<std::string>{"mhd.cmax"});
    const auto signal_min = data->PackVariables(std::vector<std::string>{"mhd.cmin"});
    Real minimum = std::numeric_limits<Real>::max();
    ParReduce(
        "PANGU packed face-speed MHD timestep", 0, data->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e,
        ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real& local) {
          const auto& coords = signal_max.GetCoords(b);
          Real estimate = Estimator::Start();
          for (int direction = 0; direction < ndim; ++direction) {
            const Real speed = fmax(signal_max(b, direction, k, j, i),
                                    signal_min(b, direction, k, j, i));
            if (isfinite(speed) && speed > 0.0)
              estimate = Estimator::Add(estimate, speed, coords.Dxc(direction + 1, k, j, i));
          }
          local = fmin(local, Estimator::Finish(estimate));
        },
        Kokkos::Min<Real>(minimum));
    if (minimum < std::numeric_limits<Real>::max())
      return cfl * minimum;
    // Initial allocation has no face characteristics yet.  Seed the first
    // timestep from the same estimator evaluated at cell centres.
  }
  auto primitive = data->PackVariables(std::vector<std::string>{"mhd.prim"});
  auto magnetic = data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  const auto spacetime = geometry::GetGeometry(mesh->packages);
  const int geometry_block_offset = data->GetBlockData(0)->GetBlockPointer()->lid;
  Real minimum = std::numeric_limits<Real>::max();
  ParReduce(
      "PANGU packed relativistic MHD timestep", 0, data->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e,
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real& local) {
        const auto& coords = primitive.GetCoords(b);
        if (!Estimator::characteristic_gr_speeds && mode == HydroMode::gr) {
          local = fmin(local, estimator::UnitSpeedTimestep(coords, k, j, i, ndim));
          return;
        }
        MHDPrimitiveState state{
            {primitive(b, mhd::IDN, k, j, i),
             {primitive(b, mhd::IV1, k, j, i), primitive(b, mhd::IV2, k, j, i),
              primitive(b, mhd::IV3, k, j, i)},
             eos.PressureFromInternalEnergyDensity(
                 primitive(b, mhd::IPR, k, j, i))},
            {magnetic(b, 0, k, j, i), magnetic(b, 1, k, j, i), magnetic(b, 2, k, j, i)}};
        Real estimate = Estimator::Start();
        for (int direction = 0; direction < ndim; ++direction) {
          Real speed;
          if (mode == HydroMode::sr) {
            const auto physical = BuildSRMHDFluxState(state, eos, direction);
            speed = fmax(fabs(physical.lambda_plus), fabs(physical.lambda_minus));
          } else {
            const auto metric = spacetime.MetricAt(geometry::Location::cell_center,
                                                   geometry_block_offset + b, k, j, i, coords);
            const auto physical = BuildGRMHDFluxState(state, eos, direction, metric);
            speed = fmax(fabs(physical.lambda_plus), fabs(physical.lambda_minus));
            // Keep the timestep signal speed inside the coordinate light
            // cone; the flux solver retains its original characteristic
            // values unchanged.
            Real light_plus = 0.0, light_minus = 0.0;
            ComputeCoordinateLightSpeeds(metric, direction, light_plus, light_minus);
            speed = fmin(speed, fmax(fabs(light_plus), fabs(light_minus)));
          }
          estimate = Estimator::Add(estimate, speed, coords.Dxc(direction + 1, k, j, i));
        }
        local = fmin(local, Estimator::Finish(estimate));
      },
      Kokkos::Min<Real>(minimum));
  return cfl * minimum;
}

TaskStatus ApplyMHDSourcesBlockTask(std::shared_ptr<MeshBlockData<Real>>& data, const Real dt) {
  auto block = data->GetBlockPointer();
  const auto package = block->packages.Get("mhd");
  if (static_cast<HydroMode>(package->Param<int>("physics_mode")) != HydroMode::gr)
    return TaskStatus::complete;
  const auto eos = pangu::eos::ReadRelativistic(*package);
  auto primitive = data->PackVariables(std::vector<std::string>{"mhd.prim"});
  auto magnetic = data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  auto conserved = data->PackVariables(std::vector<std::string>{"mhd.cons"});
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto coords = block->coords;
  const auto spacetime = geometry::GetGeometry(block->packages);
  const int geometry_block = block->lid;
  block->par_for(
      "PANGU fixed-GRMHD geometric source", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const auto metric =
            spacetime.MetricAt(geometry::Location::cell_center, geometry_block, k, j, i, coords);
        MHDPrimitiveState state{{primitive(mhd::IDN, k, j, i),
                                 {primitive(mhd::IV1, k, j, i), primitive(mhd::IV2, k, j, i),
                                  primitive(mhd::IV3, k, j, i)},
                                 eos.PressureFromInternalEnergyDensity(
                                     primitive(mhd::IPR, k, j, i))},
                                {magnetic(0, k, j, i), magnetic(1, k, j, i), magnetic(2, k, j, i)}};
        const Real uu1 = state.fluid.u[0], uu2 = state.fluid.u[1], uu3 = state.fluid.u[2];
        const Real spatial_u2 =
            metric.lower[1][1] * uu1 * uu1 + 2.0 * metric.lower[1][2] * uu1 * uu2 +
            2.0 * metric.lower[1][3] * uu1 * uu3 + metric.lower[2][2] * uu2 * uu2 +
            2.0 * metric.lower[2][3] * uu2 * uu3 + metric.lower[3][3] * uu3 * uu3;
        const Real alpha = sqrt(-1.0 / metric.upper[0][0]);
        const Real lorentz = sqrt(1.0 + spatial_u2);
        const Real u0 = lorentz / alpha;
        const Real u1 = uu1 - alpha * lorentz * metric.upper[0][1];
        const Real u2 = uu2 - alpha * lorentz * metric.upper[0][2];
        const Real u3 = uu3 - alpha * lorentz * metric.upper[0][3];
        const Real u_1 = metric.lower[1][0] * u0 + metric.lower[1][1] * u1 +
                         metric.lower[1][2] * u2 + metric.lower[1][3] * u3;
        const Real u_2 = metric.lower[2][0] * u0 + metric.lower[2][1] * u1 +
                         metric.lower[2][2] * u2 + metric.lower[2][3] * u3;
        const Real u_3 = metric.lower[3][0] * u0 + metric.lower[3][1] * u1 +
                         metric.lower[3][2] * u2 + metric.lower[3][3] * u3;
        const Real bb1 = state.magnetic[0], bb2 = state.magnetic[1], bb3 = state.magnetic[2];
        const Real b0 = u_1 * bb1 + u_2 * bb2 + u_3 * bb3;
        const Real b1 = (bb1 + b0 * u1) / u0;
        const Real b2 = (bb2 + b0 * u2) / u0;
        const Real b3 = (bb3 + b0 * u3) / u0;
        const Real b_0 = metric.lower[0][0] * b0 + metric.lower[0][1] * b1 +
                         metric.lower[0][2] * b2 + metric.lower[0][3] * b3;
        const Real b_1 = metric.lower[1][0] * b0 + metric.lower[1][1] * b1 +
                         metric.lower[1][2] * b2 + metric.lower[1][3] * b3;
        const Real b_2 = metric.lower[2][0] * b0 + metric.lower[2][1] * b1 +
                         metric.lower[2][2] * b2 + metric.lower[2][3] * b3;
        const Real b_3 = metric.lower[3][0] * b0 + metric.lower[3][1] * b1 +
                         metric.lower[3][2] * b2 + metric.lower[3][3] * b3;
        const Real magnetic2 = b_0 * b0 + b_1 * b1 + b_2 * b2 + b_3 * b3;
        const Real gamma_prime = eos.gamma / (eos.gamma - 1.0);
        const Real total_enthalpy =
            state.fluid.density + gamma_prime * state.fluid.pressure + magnetic2;
        const Real total_pressure = state.fluid.pressure + 0.5 * magnetic2;
        Real stress[4][4]{};
        stress[0][0] = total_enthalpy * u0 * u0 + total_pressure * metric.upper[0][0] - b0 * b0;
        stress[0][1] = total_enthalpy * u0 * u1 + total_pressure * metric.upper[0][1] - b0 * b1;
        stress[0][2] = total_enthalpy * u0 * u2 + total_pressure * metric.upper[0][2] - b0 * b2;
        stress[0][3] = total_enthalpy * u0 * u3 + total_pressure * metric.upper[0][3] - b0 * b3;
        stress[1][1] = total_enthalpy * u1 * u1 + total_pressure * metric.upper[1][1] - b1 * b1;
        stress[1][2] = total_enthalpy * u1 * u2 + total_pressure * metric.upper[1][2] - b1 * b2;
        stress[1][3] = total_enthalpy * u1 * u3 + total_pressure * metric.upper[1][3] - b1 * b3;
        stress[2][2] = total_enthalpy * u2 * u2 + total_pressure * metric.upper[2][2] - b2 * b2;
        stress[2][3] = total_enthalpy * u2 * u3 + total_pressure * metric.upper[2][3] - b2 * b3;
        stress[3][3] = total_enthalpy * u3 * u3 + total_pressure * metric.upper[3][3] - b3 * b3;
        const auto derivatives = spacetime.DerivativesAt(geometry_block, k, j, i, coords);
        // Keep the three source contractions explicitly separated, matching
        // AthenaK's CoordSrcTerms kernel.  A loop is algebraically identical,
        // but the compiler can schedule its reductions differently; one ulp
        // is enough to perturb a floor-dominated GRMHD inversion.
        Real source1 = 0.0, source2 = 0.0, source3 = 0.0;
        source1 += 0.5 * derivatives.lower[0][0][0] * stress[0][0];
        source1 += derivatives.lower[0][0][1] * stress[0][1];
        source1 += derivatives.lower[0][0][2] * stress[0][2];
        source1 += derivatives.lower[0][0][3] * stress[0][3];
        source1 += 0.5 * derivatives.lower[0][1][1] * stress[1][1];
        source1 += derivatives.lower[0][1][2] * stress[1][2];
        source1 += derivatives.lower[0][1][3] * stress[1][3];
        source1 += 0.5 * derivatives.lower[0][2][2] * stress[2][2];
        source1 += derivatives.lower[0][2][3] * stress[2][3];
        source1 += 0.5 * derivatives.lower[0][3][3] * stress[3][3];
        source2 += 0.5 * derivatives.lower[1][0][0] * stress[0][0];
        source2 += derivatives.lower[1][0][1] * stress[0][1];
        source2 += derivatives.lower[1][0][2] * stress[0][2];
        source2 += derivatives.lower[1][0][3] * stress[0][3];
        source2 += 0.5 * derivatives.lower[1][1][1] * stress[1][1];
        source2 += derivatives.lower[1][1][2] * stress[1][2];
        source2 += derivatives.lower[1][1][3] * stress[1][3];
        source2 += 0.5 * derivatives.lower[1][2][2] * stress[2][2];
        source2 += derivatives.lower[1][2][3] * stress[2][3];
        source2 += 0.5 * derivatives.lower[1][3][3] * stress[3][3];
        source3 += 0.5 * derivatives.lower[2][0][0] * stress[0][0];
        source3 += derivatives.lower[2][0][1] * stress[0][1];
        source3 += derivatives.lower[2][0][2] * stress[0][2];
        source3 += derivatives.lower[2][0][3] * stress[0][3];
        source3 += 0.5 * derivatives.lower[2][1][1] * stress[1][1];
        source3 += derivatives.lower[2][1][2] * stress[1][2];
        source3 += derivatives.lower[2][1][3] * stress[1][3];
        source3 += 0.5 * derivatives.lower[2][2][2] * stress[2][2];
        source3 += derivatives.lower[2][2][3] * stress[2][3];
        source3 += 0.5 * derivatives.lower[2][3][3] * stress[3][3];
        conserved(mhd::IM1, k, j, i) += dt * metric.gdet * source1;
        conserved(mhd::IM2, k, j, i) += dt * metric.gdet * source2;
        conserved(mhd::IM3, k, j, i) += dt * metric.gdet * source3;
      });
  return TaskStatus::complete;
}

TaskStatus ApplyMHDSourcesMeshTask(MeshData<Real>* data, const Real dt) {
  auto* mesh = data->GetParentPointer();
  const auto package = mesh->packages.Get("mhd");
  if (static_cast<HydroMode>(package->Param<int>("physics_mode")) != HydroMode::gr)
    return TaskStatus::complete;
  const auto eos = pangu::eos::ReadRelativistic(*package);
  auto primitive = data->PackVariables(std::vector<std::string>{"mhd.prim"});
  auto magnetic = data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  auto conserved = data->PackVariables(std::vector<std::string>{"mhd.cons"});
  const auto ib = data->GetBoundsI(IndexDomain::interior);
  const auto jb = data->GetBoundsJ(IndexDomain::interior);
  const auto kb = data->GetBoundsK(IndexDomain::interior);
  const auto spacetime = geometry::GetGeometry(mesh->packages);
  const int geometry_block_offset = data->GetBlockData(0)->GetBlockPointer()->lid;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU packed fixed-GRMHD geometric source", parthenon::DevExecSpace(),
      0, data->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto& coords = primitive.GetCoords(b);
        const auto metric = spacetime.MetricAt(geometry::Location::cell_center,
                                               geometry_block_offset + b, k, j, i, coords);
        MHDPrimitiveState state{
            {primitive(b, mhd::IDN, k, j, i),
             {primitive(b, mhd::IV1, k, j, i), primitive(b, mhd::IV2, k, j, i),
              primitive(b, mhd::IV3, k, j, i)},
             eos.PressureFromInternalEnergyDensity(
                 primitive(b, mhd::IPR, k, j, i))},
            {magnetic(b, 0, k, j, i), magnetic(b, 1, k, j, i), magnetic(b, 2, k, j, i)}};
        const Real uu1 = state.fluid.u[0], uu2 = state.fluid.u[1], uu3 = state.fluid.u[2];
        const Real spatial_u2 =
            metric.lower[1][1] * uu1 * uu1 + 2.0 * metric.lower[1][2] * uu1 * uu2 +
            2.0 * metric.lower[1][3] * uu1 * uu3 + metric.lower[2][2] * uu2 * uu2 +
            2.0 * metric.lower[2][3] * uu2 * uu3 + metric.lower[3][3] * uu3 * uu3;
        const Real alpha = sqrt(-1.0 / metric.upper[0][0]);
        const Real lorentz = sqrt(1.0 + spatial_u2);
        const Real u0 = lorentz / alpha;
        const Real u1 = uu1 - alpha * lorentz * metric.upper[0][1];
        const Real u2 = uu2 - alpha * lorentz * metric.upper[0][2];
        const Real u3 = uu3 - alpha * lorentz * metric.upper[0][3];
        const Real u_1 = metric.lower[1][0] * u0 + metric.lower[1][1] * u1 +
                         metric.lower[1][2] * u2 + metric.lower[1][3] * u3;
        const Real u_2 = metric.lower[2][0] * u0 + metric.lower[2][1] * u1 +
                         metric.lower[2][2] * u2 + metric.lower[2][3] * u3;
        const Real u_3 = metric.lower[3][0] * u0 + metric.lower[3][1] * u1 +
                         metric.lower[3][2] * u2 + metric.lower[3][3] * u3;
        const Real bb1 = state.magnetic[0], bb2 = state.magnetic[1], bb3 = state.magnetic[2];
        const Real b0 = u_1 * bb1 + u_2 * bb2 + u_3 * bb3;
        const Real b1 = (bb1 + b0 * u1) / u0;
        const Real b2 = (bb2 + b0 * u2) / u0;
        const Real b3 = (bb3 + b0 * u3) / u0;
        const Real b_0 = metric.lower[0][0] * b0 + metric.lower[0][1] * b1 +
                         metric.lower[0][2] * b2 + metric.lower[0][3] * b3;
        const Real b_1 = metric.lower[1][0] * b0 + metric.lower[1][1] * b1 +
                         metric.lower[1][2] * b2 + metric.lower[1][3] * b3;
        const Real b_2 = metric.lower[2][0] * b0 + metric.lower[2][1] * b1 +
                         metric.lower[2][2] * b2 + metric.lower[2][3] * b3;
        const Real b_3 = metric.lower[3][0] * b0 + metric.lower[3][1] * b1 +
                         metric.lower[3][2] * b2 + metric.lower[3][3] * b3;
        const Real magnetic2 = b_0 * b0 + b_1 * b1 + b_2 * b2 + b_3 * b3;
        const Real gamma_prime = eos.gamma / (eos.gamma - 1.0);
        const Real total_enthalpy =
            state.fluid.density + gamma_prime * state.fluid.pressure + magnetic2;
        const Real total_pressure = state.fluid.pressure + 0.5 * magnetic2;
        Real stress[4][4]{};
        stress[0][0] = total_enthalpy * u0 * u0 + total_pressure * metric.upper[0][0] - b0 * b0;
        stress[0][1] = total_enthalpy * u0 * u1 + total_pressure * metric.upper[0][1] - b0 * b1;
        stress[0][2] = total_enthalpy * u0 * u2 + total_pressure * metric.upper[0][2] - b0 * b2;
        stress[0][3] = total_enthalpy * u0 * u3 + total_pressure * metric.upper[0][3] - b0 * b3;
        stress[1][1] = total_enthalpy * u1 * u1 + total_pressure * metric.upper[1][1] - b1 * b1;
        stress[1][2] = total_enthalpy * u1 * u2 + total_pressure * metric.upper[1][2] - b1 * b2;
        stress[1][3] = total_enthalpy * u1 * u3 + total_pressure * metric.upper[1][3] - b1 * b3;
        stress[2][2] = total_enthalpy * u2 * u2 + total_pressure * metric.upper[2][2] - b2 * b2;
        stress[2][3] = total_enthalpy * u2 * u3 + total_pressure * metric.upper[2][3] - b2 * b3;
        stress[3][3] = total_enthalpy * u3 * u3 + total_pressure * metric.upper[3][3] - b3 * b3;
        const auto derivatives =
            spacetime.DerivativesAt(geometry_block_offset + b, k, j, i, coords);
        Real source1 = 0.0, source2 = 0.0, source3 = 0.0;
        source1 += 0.5 * derivatives.lower[0][0][0] * stress[0][0];
        source1 += derivatives.lower[0][0][1] * stress[0][1];
        source1 += derivatives.lower[0][0][2] * stress[0][2];
        source1 += derivatives.lower[0][0][3] * stress[0][3];
        source1 += 0.5 * derivatives.lower[0][1][1] * stress[1][1];
        source1 += derivatives.lower[0][1][2] * stress[1][2];
        source1 += derivatives.lower[0][1][3] * stress[1][3];
        source1 += 0.5 * derivatives.lower[0][2][2] * stress[2][2];
        source1 += derivatives.lower[0][2][3] * stress[2][3];
        source1 += 0.5 * derivatives.lower[0][3][3] * stress[3][3];
        source2 += 0.5 * derivatives.lower[1][0][0] * stress[0][0];
        source2 += derivatives.lower[1][0][1] * stress[0][1];
        source2 += derivatives.lower[1][0][2] * stress[0][2];
        source2 += derivatives.lower[1][0][3] * stress[0][3];
        source2 += 0.5 * derivatives.lower[1][1][1] * stress[1][1];
        source2 += derivatives.lower[1][1][2] * stress[1][2];
        source2 += derivatives.lower[1][1][3] * stress[1][3];
        source2 += 0.5 * derivatives.lower[1][2][2] * stress[2][2];
        source2 += derivatives.lower[1][2][3] * stress[2][3];
        source2 += 0.5 * derivatives.lower[1][3][3] * stress[3][3];
        source3 += 0.5 * derivatives.lower[2][0][0] * stress[0][0];
        source3 += derivatives.lower[2][0][1] * stress[0][1];
        source3 += derivatives.lower[2][0][2] * stress[0][2];
        source3 += derivatives.lower[2][0][3] * stress[0][3];
        source3 += 0.5 * derivatives.lower[2][1][1] * stress[1][1];
        source3 += derivatives.lower[2][1][2] * stress[1][2];
        source3 += derivatives.lower[2][1][3] * stress[1][3];
        source3 += 0.5 * derivatives.lower[2][2][2] * stress[2][2];
        source3 += derivatives.lower[2][2][3] * stress[2][3];
        source3 += 0.5 * derivatives.lower[2][3][3] * stress[3][3];
        conserved(b, mhd::IM1, k, j, i) += dt * metric.gdet * source1;
        conserved(b, mhd::IM2, k, j, i) += dt * metric.gdet * source2;
        conserved(b, mhd::IM3, k, j, i) += dt * metric.gdet * source3;
      });
  return TaskStatus::complete;
}

} // namespace pangu::relativity
