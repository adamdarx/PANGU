// CLEAN-1.7A acceptance: the reconstruction registry is the only place a model
// is named, its metadata comes from the model's own traits, and a model added
// outside the production tree reaches the device through the same path as the
// built-in ones.

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>

#include <Kokkos_Core.hpp>

#include "reconstruct/hydro_reconstruction.h"
#include "reconstruct/registry.h"

namespace {

using parthenon::Real;
namespace reconstruct = pangu::reconstruct;

void Require(const bool condition, const std::string& label) {
  if (!condition)
    throw std::runtime_error(label);
}

bool Near(const Real value, const Real reference, const Real tolerance = 0.0) {
  if (tolerance == 0.0)
    return value == reference;
  const Real scale = std::fmax(std::fabs(reference), 1.0);
  return std::fabs(value - reference) <= tolerance * scale;
}

// A reconstruction written outside PANGU: minmod-limited linear slopes.  It
// enters the build with the single registry line below and declares the physics
// it is valid for, exactly as the built-in models do.
struct MinmodLinear {
  static constexpr int stencil_radius = 2;
  static constexpr bool cell_edge_states = false;

  KOKKOS_INLINE_FUNCTION static Real Minmod(const Real left, const Real right) {
    if (left * right <= 0.0)
      return 0.0;
    return fabs(left) < fabs(right) ? left : right;
  }

  KOKKOS_INLINE_FUNCTION static void Reconstruct(const Real, const Real qmm, const Real qm,
                                                 const Real q0, const Real qp, const Real,
                                                 Real& left, Real& right) {
    left = qm + 0.5 * Minmod(qm - qmm, q0 - qm);
    right = q0 - 0.5 * Minmod(q0 - qm, qp - q0);
  }
};

inline constexpr auto test_registry = std::tuple_cat(
    reconstruct::registry,
    std::tuple{reconstruct::Entry<MinmodLinear>{"minmod_linear", 2, 2, true, false, false, false}});

// Reconstruct one face on the device, the way every fluid package does: the
// caller only supplies a stencil reader, and the model decides how much of it to
// touch.
template <typename Model>
void FaceStates(const Real stencil[6], Real& left, Real& right) {
  Kokkos::View<Real[6]> device_stencil("reconstruction stencil");
  auto host_stencil = Kokkos::create_mirror_view(device_stencil);
  for (int slot = 0; slot < 6; ++slot)
    host_stencil(slot) = stencil[slot];
  Kokkos::deep_copy(device_stencil, host_stencil);

  Kokkos::View<Real[2]> device_states("reconstruction face states");
  Kokkos::parallel_for(
      "reconstruct registry face", 1, KOKKOS_LAMBDA(const int) {
        Real face_left = 0.0;
        Real face_right = 0.0;
        reconstruct::ReconstructStencil<Model>(
            [&](const int offset) { return device_stencil(offset + 3); }, face_left, face_right);
        device_states(0) = face_left;
        device_states(1) = face_right;
      });
  auto host_states = Kokkos::create_mirror_view(device_states);
  Kokkos::deep_copy(host_states, device_states);
  left = host_states(0);
  right = host_states(1);
}

// Dispatch by registry position, as the packages do once the input is parsed.
template <const auto& Registry>
void FaceStatesOf(const reconstruct::Method method, const Real stencil[6], Real& left,
                  Real& right) {
  reconstruct::Visit<Registry>(method, [&]<reconstruct::Method Selected>() {
    FaceStates<reconstruct::ImplementationOf<Registry, Selected>>(stencil, left, right);
  });
}

// An asymmetric smooth stencil; every model sees a different sub-range of it.
constexpr Real kSmooth[6]{0.31, 0.57, 0.92, 1.44, 1.73, 1.81};
// A jump between the two cells sharing the face.
constexpr Real kJump[6]{1.0, 1.0, 1.0, 0.125, 0.125, 0.125};
// Exactly linear data: every registered model must reproduce it without limiting.
constexpr Real kLinear[6]{-2.0, -1.0, 0.0, 1.0, 2.0, 3.0};
constexpr Real kSteep[6]{0.10, 0.20, 1.00, 0.90, 0.15, 0.05};

struct Baseline {
  std::string_view name;
  Real smooth_left;
  Real smooth_right;
  Real jump_left;
  Real jump_right;
  Real steep_left;
  Real steep_right;
};

// Recorded from the implementations validated against the AthenaK comparisons in
// tst/reference/comparisons; a coefficient or limiter change moves these values.
constexpr Baseline kBaselines[]{
    {"dc", 0.92000000000000004, 1.4399999999999999, 1.0, 0.125, 1.0,
     0.90000000000000002},
    {"plm", 1.1291954022988506, 1.2538271604938271, 1.0, 0.125, 1.0,
     0.9882352941176471},
    {"ppm", 1.1850000000000001, 1.1850000000000001, 1.0, 0.125, 1.0, 1.0},
    {"ppmc", 1.1850000000000001, 1.1850000000000001, 1.0, 0.125, 1.0,
     1.0791666666666666},
    {"wenoz", 1.1583975251718901, 1.2038421551133138, 1.0, 0.125,
     1.0956143916580119, 1.1041318590772766},
};

const Baseline& BaselineFor(const std::string_view name) {
  for (const auto& baseline : kBaselines) {
    if (baseline.name == name)
      return baseline;
  }
  throw std::runtime_error("no recorded baseline for " + std::string(name));
}

void CheckRegistryMetadata() {
  Require(reconstruct::Parse("ppm4") == reconstruct::Parse("ppm"),
          "the ppm4 alias must resolve to the ppm entry");
  Require(!reconstruct::Parse("not_a_reconstruction").has_value(),
          "an unregistered name must be rejected");
  for (const auto& descriptor : reconstruct::descriptorsOf<reconstruct::registry>) {
    Require(reconstruct::Parse(descriptor.name) == descriptor.method,
            "every registered name must parse back to its own entry");
    Require(descriptor.stencil_radius >= 1 &&
                descriptor.stencil_radius <= reconstruct::kStencilCenter,
            "stencil radius must fit the shared stencil");
    Require(descriptor.ghost_zones >= descriptor.stencil_radius,
            "ghost zones must cover the stencil the model reads");
    Require(descriptor.relativistic_ghost_zones >= descriptor.stencil_radius,
            "relativistic ghost zones must cover the stencil the model reads");
  }
}

void CheckBuiltinFaceStates() {
  for (const auto& descriptor : reconstruct::descriptorsOf<reconstruct::registry>) {
    const auto& baseline = BaselineFor(descriptor.name);
    Real left = 0.0, right = 0.0;
    FaceStatesOf<reconstruct::registry>(descriptor.method, kSmooth, left, right);
    Require(Near(left, baseline.smooth_left, 1.0e-14),
            std::string(descriptor.name) + " left state left its recorded baseline");
    Require(Near(right, baseline.smooth_right, 1.0e-14),
            std::string(descriptor.name) + " right state left its recorded baseline");

    FaceStatesOf<reconstruct::registry>(descriptor.method, kJump, left, right);
    Require(Near(left, baseline.jump_left, 1.0e-14),
            std::string(descriptor.name) + " left state at a jump left its recorded baseline");
    Require(Near(right, baseline.jump_right, 1.0e-14),
            std::string(descriptor.name) + " right state at a jump left its recorded baseline");
    Require(left <= 1.0 && left >= 0.125 && right <= 1.0 && right >= 0.125,
            std::string(descriptor.name) + " introduced a new extremum at a jump");

    FaceStatesOf<reconstruct::registry>(descriptor.method, kSteep, left, right);
    Require(Near(left, baseline.steep_left, 1.0e-14),
            std::string(descriptor.name) + " left state on a steep profile left its baseline");
    Require(Near(right, baseline.steep_right, 1.0e-14),
            std::string(descriptor.name) + " right state on a steep profile left its baseline");

    FaceStatesOf<reconstruct::registry>(descriptor.method, kLinear, left, right);
    if (descriptor.stencil_radius == 1) {
      // A donor-cell face keeps the two cell averages it reads.
      Require(Near(left, kLinear[2]) && Near(right, kLinear[3]),
              "donor cell must return the adjacent cell averages");
    } else {
      Require(Near(left, 0.5, 1.0e-14) && Near(right, 0.5, 1.0e-14),
              std::string(descriptor.name) + " is not exact on linear data");
    }
  }
}

void CheckCapabilityDispatch() {
  const auto dc = *reconstruct::Parse("dc");
  const auto plm = *reconstruct::Parse("plm");
  const auto wenoz = *reconstruct::Parse("wenoz");

  bool relativistic_mhd_rejected = false;
  try {
    reconstruct::VisitRelativisticMHD(dc, []<reconstruct::Method>() {});
  } catch (const std::invalid_argument&) {
    relativistic_mhd_rejected = true;
  }
  Require(relativistic_mhd_rejected,
          "a model without the relativistic MHD capability must be refused, not replaced");

  bool passive_rejected = false;
  try {
    reconstruct::VisitPassiveTransport(wenoz, []<reconstruct::Method>() {});
  } catch (const std::invalid_argument&) {
    passive_rejected = true;
  }
  Require(passive_rejected, "passive transport must refuse a model that does not declare it");

  bool radiation_rejected = false;
  try {
    reconstruct::VisitRadiationTransport(wenoz, []<reconstruct::Method>() {});
  } catch (const std::invalid_argument&) {
    radiation_rejected = true;
  }
  Require(radiation_rejected, "radiation transport must refuse a model that does not declare it");

  reconstruct::Method dispatched{};
  reconstruct::VisitPassiveTransport(plm, [&]<reconstruct::Method Selected>() {
    dispatched = Selected;
  });
  Require(dispatched == plm, "a declared capability must reach the selected model");
}

// The single registry line is the whole integration: the name parses, the
// metadata follows the model's traits, the capability filter honours the line,
// and the device kernel runs the model itself.
void CheckRegisteredTestModel() {
  const auto added = reconstruct::Parse<test_registry>("minmod_linear");
  Require(added.has_value(), "a registered model must parse by its own name");
  const auto& descriptor = reconstruct::Describe<test_registry>(*added);
  Require(descriptor.stencil_radius == MinmodLinear::stencil_radius,
          "the stencil radius must come from the model, not from the registry line");
  Require(descriptor.ghost_zones == 2, "the registry line must supply the ghost-zone width");

  Real left = 0.0, right = 0.0;
  FaceStatesOf<test_registry>(*added, kSmooth, left, right);
  Real expected_left = 0.0, expected_right = 0.0;
  MinmodLinear::Reconstruct(kSmooth[0], kSmooth[1], kSmooth[2], kSmooth[3], kSmooth[4], kSmooth[5],
                            expected_left, expected_right);
  Require(Near(left, expected_left) && Near(right, expected_right),
          "the registered model must run unchanged on the device");

  FaceStatesOf<test_registry>(*added, kLinear, left, right);
  Require(Near(left, 0.5, 1.0e-14) && Near(right, 0.5, 1.0e-14),
          "the registered model must be exact on linear data");

  bool rejected = false;
  try {
    reconstruct::VisitRelativisticMHD<test_registry>(*added, []<reconstruct::Method>() {});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Require(rejected, "the registry line decides which physics may select the model");

  // The added line must not disturb the entries that were already there.
  for (const auto& production : reconstruct::descriptorsOf<reconstruct::registry>) {
    const auto same = reconstruct::Parse<test_registry>(production.name);
    Require(same.has_value() && *same == production.method,
            "adding a line must not renumber the existing entries");
  }
}

} // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard kokkos(argc, argv);
  try {
    CheckRegistryMetadata();
    CheckBuiltinFaceStates();
    CheckCapabilityDispatch();
    CheckRegisteredTestModel();
  } catch (const std::exception& error) {
    std::cerr << "reconstruct registry contract failed: " << error.what() << "\n";
    return 1;
  }
  std::cout << "reconstruct registry contract passed\n";
  return 0;
}
