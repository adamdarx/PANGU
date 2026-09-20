#include "plugin.h"

#include <limits>
#include <string>
#include <vector>

#include "mhd/mhd_types.h"
#include "pangu/plugin_api/diffusion.h"

namespace pangu::plugins::ohmic {
using namespace parthenon::package::prelude;

namespace {

template <typename FaceArray>
KOKKOS_INLINE_FUNCTION Real EdgeJ1(const FaceArray& b, const int k, const int j, const int i,
                                   const Real dx2, const Real dx3, const int ndim) {
  Real value = 0.0;
  if (ndim >= 2)
    value += (b(2, 0, 0, 0, k, j, i) - b(2, 0, 0, 0, k, j - 1, i)) / dx2;
  if (ndim >= 3)
    value -= (b(1, 0, 0, 0, k, j, i) - b(1, 0, 0, 0, k - 1, j, i)) / dx3;
  return value;
}

template <typename FaceArray>
KOKKOS_INLINE_FUNCTION Real EdgeJ2(const FaceArray& b, const int k, const int j, const int i,
                                   const Real dx1, const Real dx3, const int ndim) {
  Real value = -(b(2, 0, 0, 0, k, j, i) - b(2, 0, 0, 0, k, j, i - 1)) / dx1;
  if (ndim >= 3)
    value += (b(0, 0, 0, 0, k, j, i) - b(0, 0, 0, 0, k - 1, j, i)) / dx3;
  return value;
}

template <typename FaceArray>
KOKKOS_INLINE_FUNCTION Real EdgeJ3(const FaceArray& b, const int k, const int j, const int i,
                                   const Real dx1, const Real dx2, const int ndim) {
  Real value = (b(1, 0, 0, 0, k, j, i) - b(1, 0, 0, 0, k, j, i - 1)) / dx1;
  if (ndim >= 2)
    value -= (b(0, 0, 0, 0, k, j, i) - b(0, 0, 0, 0, k, j - 1, i)) / dx2;
  return value;
}

// Interpolate B to the same staggered edge as the requested current component.
// These are the AthenaK/Athena++ diagonal/off-diagonal averages, not a shared
// cell-corner approximation: J1, J2, and J3 live at three distinct edge locations.
template <int Axis, typename FaceArray, typename CellPack>
KOKKOS_INLINE_FUNCTION void EdgeMagnetic(const FaceArray& b, const CellPack& bc, const int k,
                                         const int j, const int i, const int ndim,
                                         Real magnetic[3]) {
  if constexpr (Axis == 0) {
    if (ndim == 2) {
      magnetic[0] = 0.5 * (bc(0, k, j, i) + bc(0, k, j - 1, i));
      magnetic[1] = b(1, 0, 0, 0, k, j, i);
      magnetic[2] = 0.5 * (bc(2, k, j, i) + bc(2, k, j - 1, i));
    } else {
      magnetic[0] = 0.25 * (bc(0, k, j, i) + bc(0, k - 1, j, i) + bc(0, k, j - 1, i) +
                            bc(0, k - 1, j - 1, i));
      magnetic[1] = 0.5 * (b(1, 0, 0, 0, k, j, i) + b(1, 0, 0, 0, k - 1, j, i));
      magnetic[2] = 0.5 * (b(2, 0, 0, 0, k, j, i) + b(2, 0, 0, 0, k, j - 1, i));
    }
  } else if constexpr (Axis == 1) {
    if (ndim <= 2) {
      magnetic[0] = b(0, 0, 0, 0, k, j, i);
      magnetic[1] = 0.5 * (bc(1, k, j, i) + bc(1, k, j, i - 1));
      magnetic[2] = 0.5 * (bc(2, k, j, i) + bc(2, k, j, i - 1));
    } else {
      magnetic[0] = 0.5 * (b(0, 0, 0, 0, k, j, i) + b(0, 0, 0, 0, k - 1, j, i));
      magnetic[1] = 0.25 * (bc(1, k, j, i) + bc(1, k - 1, j, i) + bc(1, k, j, i - 1) +
                            bc(1, k - 1, j, i - 1));
      magnetic[2] = 0.5 * (b(2, 0, 0, 0, k, j, i) + b(2, 0, 0, 0, k, j, i - 1));
    }
  } else {
    if (ndim == 1) {
      magnetic[0] = b(0, 0, 0, 0, k, j, i);
      magnetic[1] = 0.5 * (bc(1, k, j, i) + bc(1, k, j, i - 1));
      magnetic[2] = 0.5 * (bc(2, k, j, i) + bc(2, k, j, i - 1));
    } else {
      magnetic[0] = 0.5 * (b(0, 0, 0, 0, k, j, i) + b(0, 0, 0, 0, k, j - 1, i));
      magnetic[1] = 0.5 * (b(1, 0, 0, 0, k, j, i) + b(1, 0, 0, 0, k, j, i - 1));
      magnetic[2] = 0.25 * (bc(2, k, j, i) + bc(2, k, j - 1, i) + bc(2, k, j, i - 1) +
                            bc(2, k, j - 1, i - 1));
    }
  }
}

template <int Axis, typename FaceArray>
KOKKOS_INLINE_FUNCTION void EdgeCurrent(const FaceArray& b, const int k, const int j, const int i,
                                        const Real dx1, const Real dx2, const Real dx3,
                                        const int ndim, Real current[3]) {
  if constexpr (Axis == 0) {
    current[0] = EdgeJ1(b, k, j, i, dx2, dx3, ndim);
    current[1] =
        0.25 *
        (EdgeJ2(b, k, j - 1, i, dx1, dx3, ndim) + EdgeJ2(b, k, j - 1, i + 1, dx1, dx3, ndim) +
         EdgeJ2(b, k, j, i, dx1, dx3, ndim) + EdgeJ2(b, k, j, i + 1, dx1, dx3, ndim));
    if (ndim == 2) {
      current[2] =
          0.5 * (EdgeJ3(b, k, j, i, dx1, dx2, ndim) + EdgeJ3(b, k, j, i + 1, dx1, dx2, ndim));
    } else {
      current[2] =
          0.25 *
          (EdgeJ3(b, k - 1, j, i, dx1, dx2, ndim) + EdgeJ3(b, k - 1, j, i + 1, dx1, dx2, ndim) +
           EdgeJ3(b, k, j, i, dx1, dx2, ndim) + EdgeJ3(b, k, j, i + 1, dx1, dx2, ndim));
    }
  } else if constexpr (Axis == 1) {
    current[1] = EdgeJ2(b, k, j, i, dx1, dx3, ndim);
    if (ndim == 1) {
      current[0] = 0.0;
      current[2] = EdgeJ3(b, k, j, i, dx1, dx2, ndim);
    } else {
      current[0] =
          0.25 *
          (EdgeJ1(b, k, j, i - 1, dx2, dx3, ndim) + EdgeJ1(b, k, j, i, dx2, dx3, ndim) +
           EdgeJ1(b, k, j + 1, i - 1, dx2, dx3, ndim) + EdgeJ1(b, k, j + 1, i, dx2, dx3, ndim));
      if (ndim == 2) {
        current[2] =
            0.5 * (EdgeJ3(b, k, j, i, dx1, dx2, ndim) + EdgeJ3(b, k, j + 1, i, dx1, dx2, ndim));
      } else {
        current[2] =
            0.25 *
            (EdgeJ3(b, k - 1, j, i, dx1, dx2, ndim) + EdgeJ3(b, k - 1, j + 1, i, dx1, dx2, ndim) +
             EdgeJ3(b, k, j, i, dx1, dx2, ndim) + EdgeJ3(b, k, j + 1, i, dx1, dx2, ndim));
      }
    }
  } else {
    current[2] = EdgeJ3(b, k, j, i, dx1, dx2, ndim);
    if (ndim == 1) {
      current[0] = 0.0;
      current[1] = EdgeJ2(b, k, j, i, dx1, dx3, ndim);
    } else if (ndim == 2) {
      current[0] =
          0.5 * (EdgeJ1(b, k, j, i - 1, dx2, dx3, ndim) + EdgeJ1(b, k, j, i, dx2, dx3, ndim));
      current[1] =
          0.5 * (EdgeJ2(b, k, j - 1, i, dx1, dx3, ndim) + EdgeJ2(b, k, j, i, dx1, dx3, ndim));
    } else {
      current[0] =
          0.25 *
          (EdgeJ1(b, k, j, i - 1, dx2, dx3, ndim) + EdgeJ1(b, k, j, i, dx2, dx3, ndim) +
           EdgeJ1(b, k + 1, j, i - 1, dx2, dx3, ndim) + EdgeJ1(b, k + 1, j, i, dx2, dx3, ndim));
      current[1] =
          0.25 *
          (EdgeJ2(b, k, j - 1, i, dx1, dx3, ndim) + EdgeJ2(b, k, j, i, dx1, dx3, ndim) +
           EdgeJ2(b, k + 1, j - 1, i, dx1, dx3, ndim) + EdgeJ2(b, k + 1, j, i, dx1, dx3, ndim));
    }
  }
}

template <int Axis, typename FaceArray, typename CellPack>
KOKKOS_INLINE_FUNCTION Real NonidealEMF(const FaceArray& b, const CellPack& bc, const int k,
                                        const int j, const int i, const Real dx1, const Real dx2,
                                        const Real dx3, const int ndim, const Real eta_ohm,
                                        const Real eta_ad) {
  Real current[3]{};
  EdgeCurrent<Axis>(b, k, j, i, dx1, dx2, dx3, ndim, current);
  Real value = eta_ohm * current[Axis];
  if (eta_ad != 0.0) {
    Real magnetic[3]{};
    EdgeMagnetic<Axis>(b, bc, k, j, i, ndim, magnetic);
    const Real bsq =
        magnetic[0] * magnetic[0] + magnetic[1] * magnetic[1] + magnetic[2] * magnetic[2];
    const Real jdotb =
        current[0] * magnetic[0] + current[1] * magnetic[1] + current[2] * magnetic[2];
    value += eta_ad * (bsq * current[Axis] - jdotb * magnetic[Axis]);
  }
  return value;
}

// Component of (eta_ohm + eta_ad B^2) J at an edge.  AthenaK uses this
// expression for the conservative non-ideal Poynting flux; the parallel
// -(J.B)B term vanishes in E x B before edge-to-face averaging.
template <int Axis, typename FaceArray, typename CellPack>
KOKKOS_INLINE_FUNCTION Real WeightedCurrent(const FaceArray& b, const CellPack& bc, const int k,
                                            const int j, const int i, const Real dx1,
                                            const Real dx2, const Real dx3, const int ndim,
                                            const Real eta_ohm, const Real eta_ad) {
  Real current = 0.0;
  if constexpr (Axis == 0)
    current = EdgeJ1(b, k, j, i, dx2, dx3, ndim);
  else if constexpr (Axis == 1)
    current = EdgeJ2(b, k, j, i, dx1, dx3, ndim);
  else
    current = EdgeJ3(b, k, j, i, dx1, dx2, ndim);
  if (eta_ad == 0.0)
    return eta_ohm * current;
  Real magnetic[3]{};
  EdgeMagnetic<Axis>(b, bc, k, j, i, ndim, magnetic);
  const Real bsq =
      magnetic[0] * magnetic[0] + magnetic[1] * magnetic[1] + magnetic[2] * magnetic[2];
  return (eta_ohm + eta_ad * bsq) * current;
}


TaskStatus AddFluxes(plugin_api::DiffusionOperator::BlockData& data) {
  auto block = data->GetBlockPointer();
  const auto package = block->packages.Get("plugin.ohmic");
  const Real eta_ohm = package->Param<Real>("eta");
  constexpr Real eta_ad = 0.0;
  const bool ideal = block->packages.Get("mhd")->Param<int>("eos_mode") == 0;
  auto b = data->Get("mhd.b_face").data;
  auto bcell = data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  auto conserved = data->PackVariablesAndFluxes(std::vector<std::string>{"mhd.cons"});
  auto edge = data->Get("bnd_flux::mhd.b_face").data;
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto coords = block->coords;
  const int ndim = block->pmy_mesh->ndim;
  if (ndim == 1) {
    block->par_for(
        "PANGU 1D nonideal edge EMF", ib.s, ib.e + 1, KOKKOS_LAMBDA(const int i) {
          const Real dx1 = coords.Dxc<X1DIR>(kb.s, jb.s, i);
          const Real e2 =
              NonidealEMF<1>(b, bcell, kb.s, jb.s, i, dx1, 1.0, 1.0, ndim, eta_ohm, eta_ad);
          const Real e3 =
              NonidealEMF<2>(b, bcell, kb.s, jb.s, i, dx1, 1.0, 1.0, ndim, eta_ohm, eta_ad);
          edge(1, 0, 0, 0, kb.s, jb.s, i) += e2;
          edge(2, 0, 0, 0, kb.s, jb.s, i) += e3;
        });
  } else if (ndim == 2) {
    block->par_for(
        "PANGU 2D nonideal edge EMF", jb.s, jb.e + 1, ib.s, ib.e + 1,
        KOKKOS_LAMBDA(const int j, const int i) {
          const Real dx1 = coords.Dxc<X1DIR>(kb.s, j, i);
          const Real dx2 = coords.Dxc<X2DIR>(kb.s, j, i);
          const Real e1 =
              NonidealEMF<0>(b, bcell, kb.s, j, i, dx1, dx2, 1.0, ndim, eta_ohm, eta_ad);
          const Real e2 =
              NonidealEMF<1>(b, bcell, kb.s, j, i, dx1, dx2, 1.0, ndim, eta_ohm, eta_ad);
          const Real e3 =
              NonidealEMF<2>(b, bcell, kb.s, j, i, dx1, dx2, 1.0, ndim, eta_ohm, eta_ad);
          edge(0, 0, 0, 0, kb.s, j, i) += e1;
          edge(1, 0, 0, 0, kb.s, j, i) += e2;
          edge(2, 0, 0, 0, kb.s, j, i) += e3;
        });
  } else {
    block->par_for(
        "PANGU 3D nonideal edge EMF", kb.s, kb.e + 1, jb.s, jb.e + 1, ib.s, ib.e + 1,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const Real dx1 = coords.Dxc<X1DIR>(k, j, i);
          const Real dx2 = coords.Dxc<X2DIR>(k, j, i);
          const Real dx3 = coords.Dxc<X3DIR>(k, j, i);
          edge(0, 0, 0, 0, k, j, i) +=
              NonidealEMF<0>(b, bcell, k, j, i, dx1, dx2, dx3, ndim, eta_ohm, eta_ad);
          edge(1, 0, 0, 0, k, j, i) +=
              NonidealEMF<1>(b, bcell, k, j, i, dx1, dx2, dx3, ndim, eta_ohm, eta_ad);
          edge(2, 0, 0, 0, k, j, i) +=
              NonidealEMF<2>(b, bcell, k, j, i, dx1, dx2, dx3, ndim, eta_ohm, eta_ad);
        });
  }

  // Conservative non-ideal Poynting flux.  Edge values are formed with the
  // same staggered B interpolation as the EMF and then averaged onto each
  // face exactly as in AthenaK's constant-resistivity/ambipolar kernels.
  if (ideal) {
    block->par_for(
        "PANGU nonideal x1 Poynting flux", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e + 1,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const Real dx1 = coords.Dxc<X1DIR>(k, j, i);
          const Real dx2 = ndim >= 2 ? coords.Dxc<X2DIR>(k, j, i) : 1.0;
          const Real dx3 = ndim >= 3 ? coords.Dxc<X3DIR>(k, j, i) : 1.0;
          Real q2 = WeightedCurrent<1>(b, bcell, k, j, i, dx1, dx2, dx3, ndim, eta_ohm, eta_ad);
          Real q3 = WeightedCurrent<2>(b, bcell, k, j, i, dx1, dx2, dx3, ndim, eta_ohm, eta_ad);
          if (ndim >= 3)
            q2 = 0.5 * (q2 + WeightedCurrent<1>(b, bcell, k + 1, j, i, dx1, dx2, dx3, ndim, eta_ohm,
                                                eta_ad));
          if (ndim >= 2)
            q3 = 0.5 * (q3 + WeightedCurrent<2>(b, bcell, k, j + 1, i, dx1, dx2, dx3, ndim, eta_ohm,
                                                eta_ad));
          const Real b2f = 0.5 * (bcell(1, k, j, i - 1) + bcell(1, k, j, i));
          const Real b3f = 0.5 * (bcell(2, k, j, i - 1) + bcell(2, k, j, i));
          conserved.flux(X1DIR, mhd::IEN, k, j, i) += q2 * b3f - q3 * b2f;
        });
    if (ndim >= 2) {
      block->par_for(
          "PANGU nonideal x2 Poynting flux", kb.s, kb.e, jb.s, jb.e + 1, ib.s, ib.e,
          KOKKOS_LAMBDA(const int k, const int j, const int i) {
            const Real dx1 = coords.Dxc<X1DIR>(k, j, i);
            const Real dx2 = coords.Dxc<X2DIR>(k, j, i);
            const Real dx3 = ndim >= 3 ? coords.Dxc<X3DIR>(k, j, i) : 1.0;
            Real q3 =
                0.5 *
                (WeightedCurrent<2>(b, bcell, k, j, i, dx1, dx2, dx3, ndim, eta_ohm, eta_ad) +
                 WeightedCurrent<2>(b, bcell, k, j, i + 1, dx1, dx2, dx3, ndim, eta_ohm, eta_ad));
            Real q1 = WeightedCurrent<0>(b, bcell, k, j, i, dx1, dx2, dx3, ndim, eta_ohm, eta_ad);
            if (ndim >= 3)
              q1 = 0.5 * (q1 + WeightedCurrent<0>(b, bcell, k + 1, j, i, dx1, dx2, dx3, ndim,
                                                  eta_ohm, eta_ad));
            const Real b1f = 0.5 * (bcell(0, k, j - 1, i) + bcell(0, k, j, i));
            const Real b3f = 0.5 * (bcell(2, k, j - 1, i) + bcell(2, k, j, i));
            conserved.flux(X2DIR, mhd::IEN, k, j, i) += q3 * b1f - q1 * b3f;
          });
    }
    if (ndim >= 3) {
      block->par_for(
          "PANGU nonideal x3 Poynting flux", kb.s, kb.e + 1, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int k, const int j, const int i) {
            const Real dx1 = coords.Dxc<X1DIR>(k, j, i);
            const Real dx2 = coords.Dxc<X2DIR>(k, j, i);
            const Real dx3 = coords.Dxc<X3DIR>(k, j, i);
            const Real q1 =
                0.5 *
                (WeightedCurrent<0>(b, bcell, k, j, i, dx1, dx2, dx3, ndim, eta_ohm, eta_ad) +
                 WeightedCurrent<0>(b, bcell, k, j + 1, i, dx1, dx2, dx3, ndim, eta_ohm, eta_ad));
            const Real q2 =
                0.5 *
                (WeightedCurrent<1>(b, bcell, k, j, i, dx1, dx2, dx3, ndim, eta_ohm, eta_ad) +
                 WeightedCurrent<1>(b, bcell, k, j, i + 1, dx1, dx2, dx3, ndim, eta_ohm, eta_ad));
            const Real b1f = 0.5 * (bcell(0, k - 1, j, i) + bcell(0, k, j, i));
            const Real b2f = 0.5 * (bcell(1, k - 1, j, i) + bcell(1, k, j, i));
            conserved.flux(X3DIR, mhd::IEN, k, j, i) += q1 * b2f - q2 * b1f;
          });
    }
  }
  return TaskStatus::complete;
}

Real EstimateTimestep(MeshBlockData<Real>* data) {
  auto block = data->GetBlockPointer();
  const auto package = block->packages.Get("plugin.ohmic");
  const Real eta = package->Param<Real>("eta");
  if (eta == 0.0)
    return std::numeric_limits<Real>::max();
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto coordinates = block->coords;
  const int dimensions = block->pmy_mesh->ndim;
  Real timestep = std::numeric_limits<Real>::max();
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "PANGU Ohmic timestep", parthenon::DevExecSpace(),
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, Real& local) {
        Real inverse_spacing_squared =
            1.0 / (coordinates.Dxc<X1DIR>(k, j, i) * coordinates.Dxc<X1DIR>(k, j, i));
        if (dimensions >= 2)
          inverse_spacing_squared +=
              1.0 / (coordinates.Dxc<X2DIR>(k, j, i) * coordinates.Dxc<X2DIR>(k, j, i));
        if (dimensions >= 3)
          inverse_spacing_squared +=
              1.0 / (coordinates.Dxc<X3DIR>(k, j, i) * coordinates.Dxc<X3DIR>(k, j, i));
        local = fmin(local, 0.5 / (eta * inverse_spacing_squared));
      }, Kokkos::Min<Real>(timestep));
  return package->Param<Real>("cfl") * timestep;
}

} // namespace

std::shared_ptr<StateDescriptor> Plugin::Initialize(plugin::InitContext& context) {
  auto package = std::make_shared<StateDescriptor>("plugin.ohmic");
  const Real eta = context.input.GetOrAddReal("plugin/ohmic", "eta", 0.0);
  const Real cfl = context.input.GetOrAddReal("plugin/ohmic", "cfl", 0.5);
  PARTHENON_REQUIRE(eta >= 0.0, "plugin/ohmic/eta must be non-negative");
  PARTHENON_REQUIRE(cfl > 0.0 && cfl <= 1.0, "plugin/ohmic/cfl must be in (0,1]");
  package->AddParam<Real>("eta", eta);
  package->AddParam<Real>("cfl", cfl);
  package->AddParam<plugin_api::DiffusionOperator>(
      std::string(plugin_api::diffusion_operator_key),
      plugin_api::DiffusionOperator{"ohmic", AddFluxes});
  package->EstimateTimestepBlock = EstimateTimestep;
  return package;
}

} // namespace pangu::plugins::ohmic
