#include "plugin.h"
#include "ems_coupled.h"
#include "driver/contributor.h"
#include "driver/task_assembly.h"
#include "geometry_assembly.h"
#include "interface/update.hpp"
#include "pangu/plugin_api/scalar_field.h"
#include <cmath>
#include <limits>
#include <set>
#include <string>
#include <vector>

namespace pangu::plugins::scalar_field {
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;
namespace api = pangu::plugin_api::scalar_field;
namespace {
constexpr const char *package_name = "plugin.scalar_field";
struct Parameters {
  bool radial = true, axion = false, em_linearized = false;
  int fields = 1, ell = 0, initial = 0;
  Real mass = 1, charge = 0, scalar_mass = 0, quartic = 0, sextic = 0;
  Real decay_constant = 1, eta = 0, em_alpha = 0, coupling_alpha = 1;
  int coupling_kind = 0, em_kind = 0;
  Real amplitude = 1.e-3, center = 20, width = 3;
  Real wave_number = 1, cfl = 0.4;
};

void InitializeState(Mesh *, ParameterInput *, MeshData<Real> *data) {
  const auto p = data->GetParentPointer()
                     ->packages.Get(package_name)
                     ->Param<Parameters>("parameters");
  const auto state =
      data->PackVariables(std::vector<std::string>{"scalar.state"});
  const auto background =
      data->PackVariables(std::vector<std::string>{"scalar.background"});
  const auto ib = data->GetBoundsI(IndexDomain::entire),
             jb = data->GetBoundsJ(IndexDomain::entire),
             kb = data->GetBoundsK(IndexDomain::entire);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "scalar initialize", DevExecSpace(), 0,
      data->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(int b, int k, int j, int i) {
        const auto &c = state.GetCoords(b);
        const Real x = c.Xc<1>(i);
        const auto bg = p.radial ? api::RNTortoise(x, p.mass, p.charge)
                                 : api::RNPoint{1, 1, 0, 0, 0};
        background(b, 0, k, j, i) = bg.radius;
        background(b, 1, k, j, i) = bg.redshift;
        background(b, 2, k, j, i) = bg.gauss_bonnet;
        background(b, 3, k, j, i) = bg.geometric_potential;
        background(b, 4, k, j, i) = bg.maxwell_squared;
        for (int a = 0; a < p.fields; ++a) {
          Real u = 0, v = 0;
          if (p.initial == 1 && a == 0)
            u = p.amplitude * exp(-pow((x - p.center) / p.width, 2));
          if (p.initial == 2 && a == 0)
            u = bg.radius * api::LinearHair(bg.radius, p.mass, p.eta);
          if (p.initial == 3) {
            const Real theta = p.wave_number * x;
            const Real a2 = p.amplitude * p.amplitude;
            const Real omega = sqrt(
                p.wave_number * p.wave_number + p.scalar_mass * p.scalar_mass +
                (p.fields == 2 ? 2 * p.quartic * a2 + p.sextic * a2 * a2 : 0));
            u = p.amplitude * (p.fields == 1
                                   ? sin(theta)
                                   : (a == 0 ? cos(theta) : sin(theta)));
            v = p.amplitude * omega *
                (p.fields == 1 ? -cos(theta)
                               : (a == 0 ? sin(theta) : -cos(theta)));
          }
          if (p.initial == 4 && a == 0)
            u = p.amplitude;
          state(b, 2 * a, k, j, i) = u;
          state(b, 2 * a + 1, k, j, i) = v;
        }
      });
}
void FillPhi(MeshData<Real> *data) {
  const auto p = data->GetParentPointer()
                     ->packages.Get(package_name)
                     ->Param<Parameters>("parameters");
  const auto u = data->PackVariables(std::vector<std::string>{"scalar.state"});
  const auto bg =
      data->PackVariables(std::vector<std::string>{"scalar.background"});
  const auto phi = data->PackVariables(std::vector<std::string>{"scalar.phi"});
  const auto ib = data->GetBoundsI(IndexDomain::entire),
             jb = data->GetBoundsJ(IndexDomain::entire),
             kb = data->GetBoundsK(IndexDomain::entire);
  // This callback also executes during restart initialization, before timestep
  // estimation. Rebuild derived coefficients without altering checkpoint state.
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "scalar physical field", DevExecSpace(), 0,
      data->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(int b, int k, int j, int i) {
        const auto q = p.radial ? api::RNTortoise(u.GetCoords(b).Xc<1>(i),
                                                  p.mass, p.charge)
                                : api::RNPoint{1, 1, 0, 0, 0};
        bg(b, 0, k, j, i) = q.radius;
        bg(b, 1, k, j, i) = q.redshift;
        bg(b, 2, k, j, i) = q.gauss_bonnet;
        bg(b, 3, k, j, i) = q.geometric_potential;
        bg(b, 4, k, j, i) = q.maxwell_squared;
        for (int a = 0; a < p.fields; ++a)
          phi(b, a, k, j, i) = u(b, 2 * a, k, j, i) / q.radius;
      });
}

TaskStatus RHS(MeshData<Real> *data) {
  auto *mesh = data->GetParentPointer();
  const auto p =
      mesh->packages.Get(package_name)->Param<Parameters>("parameters");
  const auto u = data->PackVariables(std::vector<std::string>{"scalar.state"});
  const auto rhs = data->PackVariables(std::vector<std::string>{"scalar.rhs"});
  const auto bg =
      data->PackVariables(std::vector<std::string>{"scalar.background"});
  const Real xmin = mesh->mesh_size.xmin(X1DIR),
             xmax = mesh->mesh_size.xmax(X1DIR);
  const int ndim = mesh->ndim;
  const auto ib = data->GetBoundsI(IndexDomain::interior),
             jb = data->GetBoundsJ(IndexDomain::interior),
             kb = data->GetBoundsK(IndexDomain::interior);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "scalar Klein Gordon RHS", DevExecSpace(), 0,
      data->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(int b, int k, int j, int i) {
        const auto &c = u.GetCoords(b);
        const Real dx = c.Dxc<X1DIR>(k, j, i);
        const Real r = bg(b, 0, k, j, i), f = bg(b, 1, k, j, i);
        Real rho2 = 0;
        for (int a = 0; a < p.fields; ++a) {
          const Real v = u(b, 2 * a, k, j, i) / r;
          rho2 += v * v;
        }
        const api::Invariants inv{bg(b, 2, k, j, i), bg(b, 4, k, j, i)};
        const api::Coupling coupling{p.coupling_kind, p.coupling_alpha};
        const api::Force<api::PolynomialPotential> polynomial{
            {p.scalar_mass * p.scalar_mass, p.quartic, p.sextic},
            coupling,
            p.eta,
            {p.em_alpha, p.em_linearized, p.em_kind}};
        const api::Force<api::AxionPotential> axion{
            {p.scalar_mass * p.scalar_mass, p.decay_constant},
            coupling,
            p.eta,
            {p.em_alpha, p.em_linearized, p.em_kind}};
        for (int a = 0; a < p.fields; ++a) {
          const int n = 2 * a;
          const Real value = u(b, n, k, j, i), v = u(b, n + 1, k, j, i);
          rhs(b, n, k, j, i) = v;
          // Pure incoming characteristic at the inner horizon end and outgoing
          // at infinity. One-sided fourth-order closure never reads boundary
          // ghosts.
          const bool left = p.radial && c.Xc<1>(i) < xmin + 2 * dx;
          const bool right = p.radial && c.Xc<1>(i) > xmax - 2 * dx;
          if (left || right) {
            const int s = left ? 1 : -1;
            const Real d = (-25 * v + 48 * u(b, n + 1, k, j, i + s) -
                            36 * u(b, n + 1, k, j, i + 2 * s) +
                            16 * u(b, n + 1, k, j, i + 3 * s) -
                            3 * u(b, n + 1, k, j, i + 4 * s)) /
                           (12 * dx * s);
            rhs(b, n + 1, k, j, i) = left ? d : -d;
          } else {
            Real lap = (-u(b, n, k, j, i + 2) + 16 * u(b, n, k, j, i + 1) -
                        30 * value + 16 * u(b, n, k, j, i - 1) -
                        u(b, n, k, j, i - 2)) /
                       (12 * dx * dx);
            if (ndim > 1) {
              const Real h = c.Dxc<X2DIR>(k, j, i);
              lap += (-u(b, n, k, j + 2, i) + 16 * u(b, n, k, j + 1, i) -
                      30 * value + 16 * u(b, n, k, j - 1, i) -
                      u(b, n, k, j - 2, i)) /
                     (12 * h * h);
            }
            if (ndim > 2) {
              const Real h = c.Dxc<X3DIR>(k, j, i);
              lap += (-u(b, n, k + 2, j, i) + 16 * u(b, n, k + 1, j, i) -
                      30 * value + 16 * u(b, n, k - 1, j, i) -
                      u(b, n, k - 2, j, i)) /
                     (12 * h * h);
            }
            const Real force = p.axion ? axion(value / r, rho2, inv)
                                       : polynomial(value / r, rho2, inv);
            rhs(b, n + 1, k, j, i) = lap - bg(b, 3, k, j, i) * value -
                                     f * p.ell * (p.ell + 1) / (r * r) * value -
                                     f * r * force;
          }
        }
      });
  return TaskStatus::complete;
}
TaskStatus RKUpdate(MeshData<Real> *data, int stage, Real dt) {
  const auto u = data->PackVariables(std::vector<std::string>{"scalar.state"});
  const auto rhs = data->PackVariables(std::vector<std::string>{"scalar.rhs"});
  const auto initial =
      data->PackVariables(std::vector<std::string>{"scalar.initial"});
  const auto sum =
      data->PackVariables(std::vector<std::string>{"scalar.rk_sum"});
  const auto ib = data->GetBoundsI(IndexDomain::interior),
             jb = data->GetBoundsJ(IndexDomain::interior),
             kb = data->GetBoundsK(IndexDomain::interior);
  const Real weight = (stage == 1 || stage == 4) ? 1.0 / 6 : 1.0 / 3;
  const Real alpha = stage < 3 ? 0.5 : 1;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "scalar classical RK4", DevExecSpace(), 0,
      data->NumBlocks() - 1, 0, u.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(int b, int n, int k, int j, int i) {
        if (stage == 1) {
          initial(b, n, k, j, i) = u(b, n, k, j, i);
          sum(b, n, k, j, i) = 0;
        }
        sum(b, n, k, j, i) += weight * rhs(b, n, k, j, i);
        u(b, n, k, j, i) =
            initial(b, n, k, j, i) +
            dt * (stage == 4 ? sum(b, n, k, j, i) : alpha * rhs(b, n, k, j, i));
      });
  return TaskStatus::complete;
}
Real Timestep(MeshBlockData<Real> *data) {
  const auto p = data->GetBlockPointer()
                     ->packages.Get(package_name)
                     ->Param<Parameters>("parameters");
  const auto &c = data->GetBlockPointer()->coords;
  const auto ib = data->GetBoundsI(IndexDomain::interior),
             jb = data->GetBoundsJ(IndexDomain::interior),
             kb = data->GetBoundsK(IndexDomain::interior);
  const int ndim = data->GetBlockPointer()->pmy_mesh->ndim;
  Real rate2 = 0;
  for (int d = 0; d < ndim; ++d) {
    Real h = d == 0 ? c.Dxc<X1DIR>(kb.s, jb.s, ib.s)
                    : (d == 1 ? c.Dxc<X2DIR>(kb.s, jb.s, ib.s)
                              : c.Dxc<X3DIR>(kb.s, jb.s, ib.s));
    rate2 += 1 / (h * h);
  }
  // The nonlinear local oscillation frequency also constrains the explicit
  // step.
  const auto state =
      data->PackVariables(std::vector<std::string>{"scalar.state"});
  const auto bg =
      data->PackVariables(std::vector<std::string>{"scalar.background"});
  Real vmax = 0;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "scalar source timestep",
      DevExecSpace(), kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(int k, int j, int i, Real &local) {
        const Real r = bg(0, k, j, i);
        Real rho2 = 0;
        for (int a = 0; a < p.fields; ++a) {
          Real v = state(2 * a, k, j, i) / r;
          rho2 += v * v;
        }
        Real bound = p.scalar_mass * p.scalar_mass +
                     6 * fabs(p.quartic) * rho2 +
                     5 * fabs(p.sextic) * rho2 * rho2;
        if (p.eta != 0 && p.coupling_kind == 1)
          bound += fabs(p.eta * bg(2, k, j, i));
        if (p.eta != 0 && p.coupling_kind == 2)
          bound += fabs(p.eta * bg(2, k, j, i)) * 2 * fabs(p.coupling_alpha) *
                   (1 + 2 * fabs(p.coupling_alpha) * rho2) *
                   exp(p.coupling_alpha * rho2);
        bound +=
            fabs(p.em_alpha * bg(4, k, j, i)) *
            (p.em_kind == 0 && p.em_alpha < 0 && !p.em_linearized
                 ? exp(-p.em_alpha * rho2) * (1 + 2 * fabs(p.em_alpha) * rho2)
                 : 1);
        local =
            (!isfinite(rho2) || !isfinite(bound))
                ? INFINITY
                : fmax(local, fabs(bg(3, k, j, i)) + bg(1, k, j, i) * bound);
      },
      Kokkos::Max<Real>(vmax));
  PARTHENON_REQUIRE(
      std::isfinite(vmax) && vmax >= 0,
      "scalar field or force became nonfinite; evolution stopped");
  return p.cfl / sqrt(rate2 + vmax);
}
Real MeshTimestep(MeshData<Real> *data) {
  // Parthenon's mesh callback does not fall back to EstimateTimestepBlock.
  // Refresh the CFL vote after every completed RK step, including restart.
  Real dt = std::numeric_limits<Real>::max();
  for (int b = 0; b < data->NumBlocks(); ++b)
    dt = std::min(dt, Timestep(data->GetBlockData(b).get()));
  return dt;
}
void Configure(Mesh *mesh, parthenon::LowStorageIntegrator &rk) {
  const auto p =
      mesh->packages.Get(package_name)->Param<Parameters>("parameters");
  PARTHENON_REQUIRE(
      !mesh->multilevel,
      "scalar_field v0.1 uses uniform grids; AMR validation pending");
  PARTHENON_REQUIRE(!p.radial || mesh->ndim == 1,
                    "radial_tortoise requires a 1D computational mesh");
  PARTHENON_REQUIRE(parthenon::Globals::nghost >= 2,
                    "scalar fourth-order derivatives require nghost>=2");
  PARTHENON_REQUIRE(rk.GetName() == "rk4",
                    "scalar_field requires time/integrator=rk4");
  if (p.radial) {
    PARTHENON_REQUIRE(mesh->mesh_bc_names[0] == "outflow" &&
                          mesh->mesh_bc_names[1] == "outflow",
                      "radial_tortoise requires outflow x boundaries "
                      "(radiative active-cell closure)");
    PARTHENON_REQUIRE(mesh->block_list.front()
                                  ->cellbounds.GetBoundsI(IndexDomain::interior)
                                  .e -
                              mesh->block_list.front()
                                  ->cellbounds.GetBoundsI(IndexDomain::interior)
                                  .s +
                              1 >=
                          8,
                      "scalar block nx1>=8");
  } else {
    for (int d = 0; d < 2 * mesh->ndim; ++d)
      PARTHENON_REQUIRE(
          mesh->mesh_bc_names[d] == "periodic",
          "cartesian_flat v0.1 requires periodic active directions");
  }
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
  auto *rk = context.integrator;
  const auto partitions = mesh->GetDefaultBlockPartitions();
  TaskCollection collection;
  TaskID none(0);
  auto &region = collection.AddRegion(partitions.size());
  for (std::size_t p = 0; p < partitions.size(); ++p) {
    auto &tasks = region[p];
    auto &data = mesh->mesh_data.Add("base", partitions[p]);
    const auto receive = tasks.AddTask(
        none, parthenon::StartReceiveBoundBufs<parthenon::BoundaryType::any>,
        data);
    const auto rhs = tasks.AddTask(none, RHS, data.get());
    const auto update =
        tasks.AddTask(rhs, RKUpdate, data.get(), context.stage, rk->dt);
    const auto boundaries = parthenon::AddBoundaryExchangeTasks(
        update | receive, tasks, data, false);
    const auto derived = tasks.AddTask(
        boundaries, parthenon::Update::FillDerived<MeshData<Real>>, data.get());
    if (context.stage == 4)
      tasks.AddTask(derived,
                    parthenon::Update::EstimateTimestep<MeshData<Real>>,
                    data.get());
  }
  return collection;
}
} // namespace
std::shared_ptr<parthenon::StateDescriptor>
Plugin::Initialize(plugin::InitContext &context) {
  auto &pin = context.input;
  const std::string block = "plugin/scalar_field";
  if (pin.GetOrAddString(block, "backend", "radial_tortoise") ==
      "spherical_ems")
    return InitializeCoupledEMS(context);
  const std::set<std::string> keys{
      "backend",      "potential",      "fields",        "ell",
      "bh_mass",      "bh_charge",      "mass",          "quartic",
      "sextic",       "decay_constant", "gb_eta",        "gb_coupling",
      "gb_alpha",     "em_alpha",       "em_linearized", "amplitude",
      "center",       "width",          "wave_number",   "cfl",
      "initial_data", "em_coupling"};
  for (const auto &key : pin.GetParameterNames(block))
    PARTHENON_REQUIRE(keys.count(key) != 0, "unknown scalar parameter: " + key);
  PARTHENON_REQUIRE(pin.GetString("parthenon/job", "problem_id") ==
                        "scalar_field",
                    "scalar_field v0.1 is a standalone test-field problem, not "
                    "a coupled Z4c matter source");
  PARTHENON_REQUIRE(!geometry::ConfiguredGeometryIsSynchronized(),
                    "scalar test-field backend requires a fixed-background "
                    "build (METRIC=cks MODE=dynamic)");
  Parameters p;
  const auto backend = pin.GetOrAddString(block, "backend", "radial_tortoise");
  PARTHENON_REQUIRE(backend == "radial_tortoise" || backend == "cartesian_flat",
                    "unknown scalar backend");
  p.radial = backend == "radial_tortoise";
  const auto potential = pin.GetOrAddString(block, "potential", "polynomial");
  PARTHENON_REQUIRE(potential == "polynomial" || potential == "axion",
                    "unknown scalar potential");
  p.axion = potential == "axion";
  p.fields = pin.GetOrAddInteger(block, "fields", 1);
  p.ell = pin.GetOrAddInteger(block, "ell", 0);
  p.mass = pin.GetOrAddReal(block, "bh_mass", 1);
  p.charge = pin.GetOrAddReal(block, "bh_charge", 0);
  p.scalar_mass = pin.GetOrAddReal(block, "mass", 0);
  p.quartic = pin.GetOrAddReal(block, "quartic", 0);
  p.sextic = pin.GetOrAddReal(block, "sextic", 0);
  p.decay_constant = pin.GetOrAddReal(block, "decay_constant", 1);
  p.eta = pin.GetOrAddReal(block, "gb_eta", 0);
  p.em_alpha = pin.GetOrAddReal(block, "em_alpha", 0);
  const auto em = pin.GetOrAddString(block, "em_coupling", "exponential");
  PARTHENON_REQUIRE(em == "exponential" || em == "quadratic",
                    "unknown EMS coupling");
  p.em_kind = em == "exponential" ? 0 : 1;
  PARTHENON_REQUIRE(
      p.em_kind == 0 || p.em_alpha >= 0,
      "quadratic Maxwell coupling requires alpha>=0 to preserve f>0");
  p.em_linearized = pin.GetOrAddBoolean(block, "em_linearized", false);
  p.coupling_alpha = pin.GetOrAddReal(block, "gb_alpha", 1);
  const auto coupling = pin.GetOrAddString(block, "gb_coupling", "linear");
  PARTHENON_REQUIRE(coupling == "linear" || coupling == "quadratic" ||
                        coupling == "exponential",
                    "unknown GB coupling");
  p.coupling_kind =
      coupling == "linear" ? 0 : (coupling == "quadratic" ? 1 : 2);
  p.amplitude = pin.GetOrAddReal(block, "amplitude", 1.e-3);
  p.center = pin.GetOrAddReal(block, "center", 20);
  p.width = pin.GetOrAddReal(block, "width", 3);
  p.wave_number = pin.GetOrAddReal(block, "wave_number", 1);
  p.cfl = pin.GetOrAddReal(block, "cfl", 0.4);
  const auto initial = pin.GetOrAddString(block, "initial_data", "zero");
  PARTHENON_REQUIRE(initial == "zero" || initial == "gaussian" ||
                        initial == "linear_hair" || initial == "plane_wave" ||
                        initial == "homogeneous",
                    "unknown scalar initial_data");
  p.initial =
      initial == "zero"
          ? 0
          : (initial == "gaussian" ? 1
                                   : (initial == "linear_hair"
                                          ? 2
                                          : (initial == "plane_wave" ? 3 : 4)));
  PARTHENON_REQUIRE(p.fields == 1 || p.fields == 2,
                    "fields=1 (real) or 2 (complex pair) required");
  PARTHENON_REQUIRE(p.mass > 0 && fabs(p.charge) < p.mass,
                    "subextremal background requires M>0, |Q|<M");
  PARTHENON_REQUIRE(p.scalar_mass >= 0 && p.width > 0 && p.decay_constant > 0 &&
                        p.cfl > 0 && p.cfl <= 0.5 && p.ell >= 0,
                    "invalid scalar parameters");
  PARTHENON_REQUIRE(p.fields == 1 ||
                        (!p.axion && p.eta == 0 && p.em_alpha == 0),
                    "complex pair currently supports U(1) polynomial "
                    "potentials without GB/EM coupling");
  PARTHENON_REQUIRE(p.ell == 0 || (p.quartic == 0 && p.sextic == 0 &&
                                   !p.axion && p.eta == 0 && p.em_alpha == 0),
                    "nonlinear spherical-harmonic mixing requires ell=0");
  PARTHENON_REQUIRE(p.radial ||
                        (p.eta == 0 && p.em_alpha == 0 && p.charge == 0),
                    "flat backend has no curvature or electric invariant");
  PARTHENON_REQUIRE(
      p.initial != 2 ||
          (p.radial && p.charge == 0 && p.scalar_mass == 0 && p.quartic == 0 &&
           p.sextic == 0 && p.coupling_kind == 0 && !p.axion),
      "linear_hair requires the massless linear Schwarzschild GB model");
  PARTHENON_REQUIRE(p.initial != 3 || !p.radial,
                    "plane_wave uses cartesian_flat");
  PARTHENON_REQUIRE(p.radial || p.ell == 0,
                    "ell is an angular harmonic of the radial backend");
  PARTHENON_REQUIRE(p.initial != 4 || !p.radial,
                    "homogeneous uses cartesian_flat");
  PARTHENON_REQUIRE(
      !p.axion || (p.quartic == 0 && p.sextic == 0),
      "axion potential does not accept polynomial quartic/sextic coefficients");
  PARTHENON_REQUIRE(
      p.initial != 3 ||
          (!p.axion && (p.fields == 2 || (p.quartic == 0 && p.sextic == 0))),
      "analytic plane-wave initial momentum requires a linear real field or a "
      "U(1) polynomial complex pair");
  if (p.initial == 3 && p.fields == 2) {
    const Real a2 = p.amplitude * p.amplitude;
    PARTHENON_REQUIRE(p.wave_number * p.wave_number +
                              p.scalar_mass * p.scalar_mass + 2 * p.quartic * a2 +
                              p.sextic * a2 * a2 >= 0,
                      "complex plane-wave frequency squared must be nonnegative");
  }
  for (Real value : {p.mass, p.charge, p.scalar_mass, p.quartic, p.sextic,
                     p.decay_constant, p.eta, p.em_alpha, p.coupling_alpha,
                     p.amplitude, p.center, p.width, p.wave_number, p.cfl})
    PARTHENON_REQUIRE(std::isfinite(value), "scalar parameters must be finite");
  auto package = std::make_shared<StateDescriptor>(package_name);
  package->AddParam<Parameters>("parameters", p);
  package->AddParam<driver::StageContribution>(
      std::string(driver::stage_contribution_key),
      {"scalar_field", 200,
       [](const driver::StageBuildContext &) { return true; }, Build,
       Configure});
  const Metadata state({Metadata::Cell, Metadata::Independent,
                        Metadata::Restart, Metadata::FillGhost},
                       std::vector<int>{2 * p.fields});
  const Metadata work({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
                      std::vector<int>{2 * p.fields});
  package->AddField("scalar.state", state);
  package->AddField("scalar.rhs", work);
  package->AddField("scalar.initial", work);
  package->AddField("scalar.rk_sum", work);
  package->AddField(
      "scalar.background",
      Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
               std::vector<int>{5}));
  package->AddField("scalar.phi", Metadata({Metadata::Cell, Metadata::Derived,
                                            Metadata::OneCopy},
                                           std::vector<int>{p.fields}));
  package->PostProblemGeneratorMesh = InitializeState;
  // Reconstruct immutable background coefficients after restart as well.
  package->FillDerivedMesh = FillPhi;
  package->EstimateTimestepBlock = Timestep;
  package->EstimateTimestepMesh = MeshTimestep;
  return package;
}
} // namespace pangu::plugins::scalar_field
