// Analytic AMR-halo regression: constants and tensor quartics must survive
// restriction/prolongation at coarse/fine faces, edges and corners.
#include "bvals/comms/bvals_in_one.hpp"
#include "parthenon_manager.hpp"
#include "z4c/evolution/amr_refinement.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <parthenon/parthenon.hpp>
using namespace parthenon;
KOKKOS_INLINE_FUNCTION Real F(Real x, Real y, Real z, int epoch) {
  if (epoch < 2)
    return 1 + epoch;
  return 1 + epoch + .03 * x - .02 * y + .01 * z +
         .005 * (x * x + 2 * y * y + 3 * z * z) + .002 * x * y * z +
         .0001 * (x * x * x * x + 2 * y * y * y * y + 3 * z * z * z * z +
                  x * x * y * y * z * z);
}
Packages_t Packages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t p;
  auto pkg = std::make_shared<StateDescriptor>("probe");
  Metadata m({Metadata::Cell, Metadata::Independent, Metadata::FillGhost},
             std::vector<int>{1});
  m.RegisterRefinementOps<pangu::nr::amr::ProlongateZ4cHighOrder,
                          pangu::nr::amr::RestrictZ4cHighOrder>();
  pkg->AddField("probe", m);
  p.Add(pkg);
  return p;
}
void Fill(Mesh *mesh, int epoch) {
  for (const auto &part : mesh->GetDefaultBlockPartitions()) {
    auto &md = mesh->mesh_data.Add("base", part);
    auto q = md->PackVariables(std::vector<std::string>{"probe"});
    auto pmb = md->GetBlockData(0)->GetBlockPointer();
    auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
    auto ie = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
    auto je = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
    auto ke = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
    par_for(
        DEFAULT_LOOP_PATTERN, "reset analytic interior, poison ghosts",
        DevExecSpace(), 0, q.GetDim(5) - 1, ke.s, ke.e, je.s, je.e, ie.s, ie.e,
        KOKKOS_LAMBDA(int b, int k, int j, int i) {
          auto c = q.GetCoords(b);
          q(b, 0, k, j, i) = (i >= ib.s && i <= ib.e && j >= jb.s &&
                              j <= jb.e && k >= kb.s && k <= kb.e)
                                 ? F(c.Xc<X1DIR>(k, j, i), c.Xc<X2DIR>(k, j, i),
                                     c.Xc<X3DIR>(k, j, i), epoch)
                                 : NAN;
        });
  }
  Kokkos::fence();
}
void Exchange(Mesh *mesh) {
  TaskCollection coll;
  auto parts = mesh->GetDefaultBlockPartitions();
  auto &region = coll.AddRegion(parts.size());
  for (size_t j = 0; j < parts.size(); ++j) {
    auto &md = mesh->mesh_data.Add("base", parts[j]);
    auto &tasks = region[j];
    auto r =
        tasks.AddTask(TaskID(0), StartReceiveBoundBufs<BoundaryType::any>, md);
    AddBoundaryExchangeTasks(r, tasks, md, mesh->multilevel);
  }
  PARTHENON_REQUIRE(coll.Execute() == TaskListStatus::complete,
                    "probe exchange incomplete");
  Kokkos::fence();
}
int Check(Mesh *mesh, int epoch, int pass) {
  long bad = 0, total = 0;
  Real err = 0;
  int peak_b = 0, peak_i = 0, peak_j = 0, peak_k = 0;
  Real peak_v = 0, peak_exact = 0;
  for (auto pmb : mesh->block_list) {
    auto md = pmb->meshblock_data.Get();
    auto h = md->Get("probe").data.GetHostMirrorAndCopy();
    auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
    auto ie = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
    auto je = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
    auto ke = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
    for (int k = ke.s; k <= ke.e; ++k)
      for (int j = je.s; j <= je.e; ++j)
        for (int i = ie.s; i <= ie.e; ++i) {
          if (i >= ib.s && i <= ib.e && j >= jb.s && j <= jb.e && k >= kb.s &&
              k <= kb.e)
            continue;
          Real x = pmb->coords.Xc<X1DIR>(k, j, i),
               y = pmb->coords.Xc<X2DIR>(k, j, i),
               z = pmb->coords.Xc<X3DIR>(k, j, i);
          if (x <= mesh->mesh_size.xmin(X1DIR) + .5 ||
              x >= mesh->mesh_size.xmax(X1DIR) - .5 ||
              y <= mesh->mesh_size.xmin(X2DIR) + .5 ||
              y >= mesh->mesh_size.xmax(X2DIR) - .5 ||
              z <= mesh->mesh_size.xmin(X3DIR) + .5 ||
              z >= mesh->mesh_size.xmax(X3DIR) - .5)
            continue;
          ++total;
          Real v = h(0, k, j, i);
          if (!std::isfinite(v))
            ++bad;
          else if (std::abs(v - F(x, y, z, epoch)) > err) {
            err = std::abs(v - F(x, y, z, epoch));
            peak_b = pmb->gid;
            peak_i = i;
            peak_j = j;
            peak_k = k;
            peak_v = v;
            peak_exact = F(x, y, z, epoch);
          }
        }
  }
  if (err > 1e-8)
    std::cout << "PROBE_PEAK rank=" << Globals::my_rank << " block=" << peak_b
              << " ijk=" << peak_i << "," << peak_j << "," << peak_k
              << " value=" << peak_v << " exact=" << peak_exact << std::endl;
  long gb, gt;
  Real ge;
#ifdef MPI_PARALLEL
  MPI_Allreduce(&bad, &gb, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&total, &gt, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&err, &ge, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#else
  gb = bad;
  gt = total;
  ge = err;
#endif
  if (Globals::my_rank == 0)
    std::cout << "PROBE epoch=" << epoch << " exchange=" << pass
              << " ghost_cells=" << gt << " nonfinite=" << gb
              << " max_error=" << ge << std::endl;
  return gb > 0 || gt == 0 || ge > 2e-11;
}
int main(int argc, char **argv) {
  ParthenonManager pm;
  pm.app_input->ProcessPackages = Packages;
  auto status = pm.ParthenonInitEnv(argc, argv);
  if (status != ParthenonStatus::ok)
    return 1;
  pm.ParthenonInitPackagesAndMesh();
  int failures = 0;
  for (int e = 0; e < 3; ++e) {
    Fill(pm.pmesh.get(), e);
    for (int pass = 1; pass <= 3; ++pass) {
      Exchange(pm.pmesh.get());
      failures += Check(pm.pmesh.get(), e, pass);
    }
  }
  pm.ParthenonFinalize();
  return failures ? 1 : 0;
}
