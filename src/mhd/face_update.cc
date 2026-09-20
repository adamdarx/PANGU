#include "mhd/mhd_package.h"

#include <string>
#include <vector>

namespace pangu::mhd {
using namespace parthenon::package::prelude;

TaskStatus UpdateFaceFieldsMeshTask(MeshData<Real>* current, MeshData<Real>* base, const Real gam0,
                                    const Real gam1, const Real beta_dt, MeshData<Real>* next) {
  // Preserve AthenaK's CT assignment order in one kernel.  Splitting the RK
  // average, curl construction, and update into three materialized kernels is
  // algebraically equivalent, but the second RK stage can cross a GRMHD floor
  // branch after only a few ulps of magnetic-field roundoff.
  using TE = parthenon::TopologicalElement;
  auto x = current->PackVariablesAndFluxes(std::vector<std::string>{"mhd.b_face"});
  auto y = base->PackVariables(std::vector<std::string>{"mhd.b_face"});
  auto z = next->PackVariables(std::vector<std::string>{"mhd.b_face"});
  auto edge_emf = current->PackVariables(std::vector<std::string>{"mhd.edge_emf"});
  const auto ib = current->GetBoundsI(IndexDomain::interior);
  const auto jb = current->GetBoundsJ(IndexDomain::interior);
  const auto kb = current->GetBoundsK(IndexDomain::interior);
  const int ndim = current->GetMeshPointer()->ndim;
  const int nblocks = current->NumBlocks();
  if (ndim == 1) {
    // In one dimension Parthenon's VariableFluxPack does not retain all of the
    // collapsed transverse topological views used by CT.  The native face and
    // edge variables do, so update the three components directly from those
    // views.  This is the same storage path used to construct the 1-D corner
    // EMFs in mhd_package.cpp.
    for (int m = 0; m < nblocks; ++m) {
      auto current_block = current->GetBlockData(m);
      auto base_block = base->GetBlockData(m);
      auto next_block = next->GetBlockData(m);
      auto block = current_block->GetBlockPointer();
      auto current_face = current_block->Get("mhd.b_face").data;
      auto base_face = base_block->Get("mhd.b_face").data;
      auto next_face = next_block->Get("mhd.b_face").data;
      auto edge = current_block->Get("bnd_flux::mhd.b_face").data;
      const auto coords = block->coords;
      block->par_for(
          "PANGU CT B1 1D", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e + 1,
          KOKKOS_LAMBDA(const int k, const int j, const int i) {
            next_face(0, 0, 0, 0, k, j, i) =
                gam0 * current_face(0, 0, 0, 0, k, j, i) + gam1 * base_face(0, 0, 0, 0, k, j, i);
          });
      block->par_for(
          "PANGU CT B2 1D", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int k, const int j, const int i) {
            next_face(1, 0, 0, 0, k, j, i) =
                gam0 * current_face(1, 0, 0, 0, k, j, i) + gam1 * base_face(1, 0, 0, 0, k, j, i) +
                beta_dt * (edge(2, 0, 0, 0, k, j, i + 1) - edge(2, 0, 0, 0, k, j, i)) /
                    coords.Dxc<parthenon::X1DIR>(k, j, i);
          });
      block->par_for(
          "PANGU CT B3 1D", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int k, const int j, const int i) {
            next_face(2, 0, 0, 0, k, j, i) =
                gam0 * current_face(2, 0, 0, 0, k, j, i) + gam1 * base_face(2, 0, 0, 0, k, j, i) -
                beta_dt * (edge(1, 0, 0, 0, k, j, i + 1) - edge(1, 0, 0, 0, k, j, i)) /
                    coords.Dxc<parthenon::X1DIR>(k, j, i);
          });
    }
    return TaskStatus::complete;
  }
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU packed CT B1", parthenon::DevExecSpace(), 0, nblocks - 1, kb.s,
      kb.e, jb.s, jb.e, ib.s, ib.e + 1,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        const auto& coords = x.GetCoords(m);
        z(m, TE::F1, 0, k, j, i) =
            gam0 * x(m, TE::F1, 0, k, j, i) + gam1 * y(m, TE::F1, 0, k, j, i);
        if (ndim >= 2)
          z(m, TE::F1, 0, k, j, i) -=
              beta_dt * (edge_emf(m, TE::E3, 0, k, j + 1, i) - edge_emf(m, TE::E3, 0, k, j, i)) /
              coords.Dxc<parthenon::X2DIR>(k, j, i);
        if (ndim >= 3)
          z(m, TE::F1, 0, k, j, i) +=
              beta_dt * (edge_emf(m, TE::E2, 0, k + 1, j, i) - edge_emf(m, TE::E2, 0, k, j, i)) /
              coords.Dxc<parthenon::X3DIR>(k, j, i);
      });
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU packed CT B2", parthenon::DevExecSpace(), 0, nblocks - 1, kb.s,
      kb.e, jb.s, jb.e + (ndim >= 2), ib.s, ib.e,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        const auto& coords = x.GetCoords(m);
        z(m, TE::F2, 0, k, j, i) =
            gam0 * x(m, TE::F2, 0, k, j, i) + gam1 * y(m, TE::F2, 0, k, j, i);
        z(m, TE::F2, 0, k, j, i) +=
            beta_dt * (edge_emf(m, TE::E3, 0, k, j, i + 1) - edge_emf(m, TE::E3, 0, k, j, i)) /
            coords.Dxc<parthenon::X1DIR>(k, j, i);
        if (ndim >= 3)
          z(m, TE::F2, 0, k, j, i) -=
              beta_dt * (edge_emf(m, TE::E1, 0, k + 1, j, i) - edge_emf(m, TE::E1, 0, k, j, i)) /
              coords.Dxc<parthenon::X3DIR>(k, j, i);
      });
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PANGU packed CT B3", parthenon::DevExecSpace(), 0, nblocks - 1, kb.s,
      kb.e + (ndim >= 3), jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        const auto& coords = x.GetCoords(m);
        z(m, TE::F3, 0, k, j, i) =
            gam0 * x(m, TE::F3, 0, k, j, i) + gam1 * y(m, TE::F3, 0, k, j, i);
        if (ndim >= 3) {
          z(m, TE::F3, 0, k, j, i) -=
              beta_dt * (edge_emf(m, TE::E2, 0, k, j, i + 1) - edge_emf(m, TE::E2, 0, k, j, i)) /
              coords.Dxc<parthenon::X1DIR>(k, j, i);
          z(m, TE::F3, 0, k, j, i) +=
              beta_dt * (edge_emf(m, TE::E1, 0, k, j + 1, i) - edge_emf(m, TE::E1, 0, k, j, i)) /
              coords.Dxc<parthenon::X2DIR>(k, j, i);
        } else {
          z(m, TE::F3, 0, k, j, i) -=
              beta_dt * (edge_emf(m, TE::E2, 0, k, j, i + 1) - edge_emf(m, TE::E2, 0, k, j, i)) /
              coords.Dxc<parthenon::X1DIR>(k, j, i);
          z(m, TE::F3, 0, k, j, i) +=
              beta_dt * (edge_emf(m, TE::E1, 0, k, j + 1, i) - edge_emf(m, TE::E1, 0, k, j, i)) /
              coords.Dxc<parthenon::X2DIR>(k, j, i);
        }
      });
  return TaskStatus::complete;
}

} // namespace pangu::mhd
