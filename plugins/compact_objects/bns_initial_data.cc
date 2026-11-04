#include "bns_initial_data.h"

#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <string>
#include <vector>

#include "geometry_assembly.h"
#include "mhd/mhd_types.h"
#include "pangu/plugin_api/initial_data.h"
#include "z4c/core/component_indices.h"
#include "z4c/coupling/sync_grmhd.h"
#include "z4c/evolution/package.h"
#include "z4c/evolution/z4c_rhs.h"

namespace pangu::plugins::compact_objects {
namespace {
using namespace parthenon::package::prelude;
constexpr const char *label = "plugin.compact_objects";

struct BNSData {
  Kokkos::View<Real **> coefficients;
  Kokkos::View<Real *> moments, source;
  Real dr, source_dr, kappa, separation, surface;
  int nr, nl, ns;

  KOKKOS_INLINE_FUNCTION Real Source(const Real distance) const {
    if (distance >= surface) return 0;
    const Real coordinate = distance / source_dr;
    const int left = static_cast<int>(fmin(fmax(floor(coordinate)-1, 0), ns-4));
    const Real t = coordinate-left;
    const Real result = -(t-1)*(t-2)*(t-3)/6*source(left)
        +t*(t-2)*(t-3)/2*source(left+1)
        -t*(t-1)*(t-3)/2*source(left+2)
        +t*(t-1)*(t-2)/6*source(left+3);
    return fmax(result, 0);
  }

  KOKKOS_INLINE_FUNCTION Real Psi(const Real x, const Real y, const Real z) const {
    const Real radius = sqrt(x*x+y*y+z*z);
    const Real mu = radius > 0 ? z/radius : 0;
    const Real coordinate = radius/dr;
    const int left = static_cast<int>(fmin(fmax(floor(coordinate)-1, 0), nr-4));
    const Real t = coordinate-left;
    const Real w[4] = {-(t-1)*(t-2)*(t-3)/6, t*(t-2)*(t-3)/2,
                       -t*(t-1)*(t-3)/2, t*(t-1)*(t-2)/6};
    Real sum = 1, previous = 1, current = mu;
    Real inverse_power = radius > 0 ? 1/radius : 0;
    for (int ell = 0; ell <= 2*(nl-1); ++ell) {
      const Real polynomial = ell == 0 ? previous : current;
      if (ell % 2 == 0) {
        const int j = ell/2;
        Real radial = 0;
        if (coordinate <= nr-1) {
          for (int a = 0; a < 4; ++a) radial += w[a]*coefficients(left+a, j);
        } else radial = moments(j)*inverse_power;
        sum += radial*polynomial;
      }
      if (ell >= 1) {
        const Real next = ((2*ell+1)*mu*current-ell*previous)/(ell+1);
        previous = current;
        current = next;
      }
      if (radius > 0) inverse_power /= radius;
    }
    return sum;
  }
};

BNSData ReadData(const std::string &path) {
  std::ifstream file(path);
  PARTHENON_REQUIRE(file.good(), "cannot open BNS initial-data table: " + path);
  std::string schema;
  std::getline(file, schema);
  PARTHENON_REQUIRE(schema == "PANGU_BNS_TIME_SYMMETRIC_V1", "invalid BNS table schema");
  BNSData d{};
  Real residual;
  file >> d.nr >> d.nl >> d.ns >> d.dr >> d.source_dr >> d.kappa
       >> d.separation >> d.surface >> residual;
  PARTHENON_REQUIRE(file.good() && d.nr >= 4 && d.nr <= 200000 &&
                        d.nl >= 1 && d.nl <= 65 && d.ns >= 4 && d.ns <= 200000,
                    "invalid BNS table dimensions");
  for (Real value : {d.dr, d.source_dr, d.kappa, d.separation, d.surface})
    PARTHENON_REQUIRE(std::isfinite(value) && value > 0, "invalid BNS table parameter");
  PARTHENON_REQUIRE(std::isfinite(residual) && residual >= 0 && residual < 1e-9 &&
                        d.separation > 2*d.surface &&
                        (d.ns-1)*d.source_dr >= d.surface &&
                        (d.nr-1)*d.dr > d.separation/2+d.surface,
                    "unconverged, overlapping or truncated BNS initial data");
  d.moments = Kokkos::View<Real *>("BNS exterior moments", d.nl);
  d.source = Kokkos::View<Real *>("BNS TOV conformal energy", d.ns);
  d.coefficients = Kokkos::View<Real **>("BNS psi multipoles", d.nr, d.nl);
  auto hm = Kokkos::create_mirror_view(d.moments);
  auto hs = Kokkos::create_mirror_view(d.source);
  auto hc = Kokkos::create_mirror_view(d.coefficients);
  auto read = [&]() {
    Real value;
    file >> value;
    PARTHENON_REQUIRE(file && std::isfinite(value), "nonfinite or missing BNS table value");
    return value;
  };
  for (int j = 0; j < d.nl; ++j) hm(j) = read();
  PARTHENON_REQUIRE(hm(0) > 0, "BNS ADM mass must be positive");
  for (int i = 0; i < d.ns; ++i) {
    hs(i) = read();
    PARTHENON_REQUIRE(hs(i) >= 0, "negative BNS conformal energy");
  }
  for (int i = 0; i < d.nr; ++i)
    for (int j = 0; j < d.nl; ++j) hc(i, j) = read();
  std::string extra;
  PARTHENON_REQUIRE(!(file >> extra), "unexpected trailing BNS table data");
  Kokkos::deep_copy(d.moments, hm);
  Kokkos::deep_copy(d.source, hs);
  Kokkos::deep_copy(d.coefficients, hc);
  return d;
}

void InitializeBlock(MeshBlock *block, ParameterInput *) {
  PARTHENON_REQUIRE(block->pmy_mesh->ndim == 3, "BNS requires a three-dimensional mesh");
  const auto data = block->meshblock_data.Get();
  const auto d = block->packages.Get(label)->Param<BNSData>("bns_data");
  const auto eos = pangu::eos::ReadRelativistic(*block->packages.Get("mhd"));
  const auto options = block->packages.Get("numerical_relativity")
                           ->Param<nr::Z4cOptions>("z4c_options");
  const auto u = data->PackVariables(std::vector<std::string>{"nr.z4c"});
  const auto mask = data->PackVariables(std::vector<std::string>{"nr.constraint_mask"});
  const auto prim = data->PackVariables(std::vector<std::string>{"mhd.prim"});
  const auto bc = data->PackVariables(std::vector<std::string>{"mhd.b_cell"});
  const auto div = data->PackVariables(std::vector<std::string>{"mhd.divb"});
  const auto fofc = data->PackVariables(std::vector<std::string>{"mhd.fofc"});
  const auto recovery = data->PackVariables(std::vector<std::string>{"mhd.recovery"});
  const auto bf = data->Get("mhd.b_face").data;
  const auto ib = block->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto jb = block->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = block->cellbounds.GetBoundsK(IndexDomain::entire);
  const auto coords = block->coords;
  block->par_for("constraint-solved BNS initial data", kb.s, kb.e, jb.s, jb.e,
                 ib.s, ib.e, KOKKOS_LAMBDA(int k, int j, int i) {
    const Real x = coords.Xc<X1DIR>(k,j,i), y = coords.Xc<X2DIR>(k,j,i),
               z = coords.Xc<X3DIR>(k,j,i);
    const Real psi = d.Psi(x,y,z);
    if (!(psi >= 1) || !isfinite(psi)) Kokkos::abort("invalid BNS conformal factor");
    const Real ebar = d.Source(sqrt(x*x+y*y+pow(z-d.separation/2,2)))
                     +d.Source(sqrt(x*x+y*y+pow(z+d.separation/2,2)));
    const Real energy = ebar/pow(psi,8);
    const Real rho = 2*energy/(1+sqrt(1+4*d.kappa*energy));
    for (int c = 0; c < nr::kZ4cComponents; ++c) u(c,k,j,i) = 0;
    u(nr::Index(nr::Z4cComponent::chi),k,j,i) = pow(psi,options.chi_psi_power);
    u(1,k,j,i) = u(4,k,j,i) = u(6,k,j,i) = 1;
    u(nr::Index(nr::Z4cComponent::alpha),k,j,i) = 1/(psi*psi);
    mask(0,k,j,i) = 1;
    prim(mhd::IDN,k,j,i) = fmax(rho,eos.density_floor);
    prim(mhd::IPR,k,j,i) = eos.InternalEnergyDensity(fmax(d.kappa*rho*rho,eos.pressure_floor));
    for (int a = 0; a < 3; ++a) {
      prim(mhd::IV1+a,k,j,i) = 0;
      bc(a,k,j,i) = 0;
      recovery(a,k,j,i) = 0;
    }
    div(0,k,j,i) = 0;
    fofc(0,k,j,i) = 0;
  });
  Kokkos::deep_copy(bf,0.);
  nr::Z4cToADMFieldsBlockTask(data.get());
  nr::SyncGRMHDPrimitiveToConservedBlock(data.get());
  nr::BuildStressEnergyBlockTask(data.get(),0.);
  nr::Z4cToADMBlockTask(data.get());
}
} // namespace

std::shared_ptr<parthenon::StateDescriptor> InitializeBNS(plugin::InitContext &context) {
  auto &pin = context.input;
  const std::string section = "plugin/compact_objects";
  const std::set<std::string> keys{"model","initial_data_file"};
  for (const auto &key : pin.GetParameterNames(section))
    PARTHENON_REQUIRE(keys.count(key),"unknown BNS parameter: "+key);
  pin.GetString(section,"model");
  PARTHENON_REQUIRE(pin.GetString("parthenon/job","problem_id") == "sync_plugin_grmhd" &&
                        geometry::ConfiguredGeometryIsSynchronized(),
                    "BNS requires sync_plugin_grmhd, METRIC=z4c MODE=sync");
  const auto nrp = context.dependencies.Get("numerical_relativity");
  const auto mhd = context.dependencies.Get("mhd");
  PARTHENON_REQUIRE(nrp->Param<std::string>("matter_source_name") == "grmhd",
                    "BNS requires active matter_source=grmhd");
  PARTHENON_REQUIRE(mhd->Param<int>("physics_mode") == static_cast<int>(relativity::HydroMode::gr)
                        && std::abs(mhd->Param<Real>("gamma")-2.) < 1e-14
                        && mhd->Param<int>("nscalars") == 0,
                    "BNS cold Gamma=2 data requires GR ideal Gamma=2 evolution and nscalars=0");
  // Keep the atmosphere's unavoidable constraint source and mass negligible.
  const auto eos = pangu::eos::ReadRelativistic(*mhd);
  PARTHENON_REQUIRE(eos.density_floor <= 1e-12 && eos.pressure_floor <= 1e-20,
                    "BNS atmosphere floors exceed admission bounds");
  auto package = std::make_shared<StateDescriptor>(label);
  package->AddParam<BNSData>("bns_data",ReadData(pin.GetString(section,"initial_data_file")));
  package->AddParam<plugin_api::BlockInitialData>(plugin_api::block_initial_data_key,InitializeBlock);
  return package;
}
} // namespace pangu::plugins::compact_objects
