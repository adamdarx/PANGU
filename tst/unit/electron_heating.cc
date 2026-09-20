#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include "electron/model/registry.h"

namespace {

using Real = double;

void CheckClose(const Real actual, const Real expected, const char* label) {
  const Real scale = fmax(1.0, fmax(fabs(actual), fabs(expected)));
  if (fabs(actual - expected) > 5.0e-15 * scale)
    throw std::runtime_error(std::string(label) + " mismatch");
}

void CheckKHARMAConstantUpdate() {
  constexpr Real gamma = 4.0 / 3.0;
  constexpr Real gamma_e = 4.0 / 3.0;
  constexpr Real gamma_p = 5.0 / 3.0;
  constexpr Real rho_old = 1.2;
  constexpr Real ktot_old = 0.07;
  constexpr Real rho_new = 1.1;
  constexpr Real internal_new = 0.31;
  constexpr Real ktot_advected = 0.065;
  constexpr Real kel_advected = 0.012;
  constexpr Real fraction = 0.1;
  const Real k_energy = (gamma - 1.0) * internal_new * pow(rho_new, -gamma);
  const Real dissipation =
      (gamma_e - 1.0) / (gamma - 1.0) * pow(rho_old, gamma - gamma_e) * (k_energy - ktot_advected);
  const auto result = pangu::electron::ApplyConstantHeating(
      rho_old, ktot_old, rho_new, internal_new, ktot_advected, kel_advected, gamma, gamma_e,
      gamma_p, fraction, false, false, 0.0, 1.0, false, 1.0e-3, 1.0e3);
  CheckClose(result.total_entropy, k_energy, "energy-conserving entropy");
  CheckClose(result.raw_dissipation, dissipation, "raw dissipative entropy");
  CheckClose(result.applied_dissipation, dissipation, "applied dissipative entropy");
  CheckClose(result.electron_entropy, kel_advected + fraction * dissipation,
             "constant electron heating");
  if (result.flags != pangu::electron::heating_none)
    throw std::runtime_error("unlimited positive update set a flag");
}

void CheckNegativeAndMagnetizationSuppression() {
  constexpr Real gamma = 4.0 / 3.0;
  constexpr Real gamma_e = 4.0 / 3.0;
  constexpr Real gamma_p = 5.0 / 3.0;
  const auto clipped =
      pangu::electron::ApplyConstantHeating(1.0, 0.2, 1.0, 0.1, 0.2, 0.02, gamma, gamma_e, gamma_p,
                                             0.25, true, false, 0.0, 1.0, false, 1.0e-3, 1.0e3);
  CheckClose(clipped.applied_dissipation, 0.0, "negative dissipation clipping");
  CheckClose(clipped.electron_entropy, 0.02, "negative dissipation electron entropy");
  if ((clipped.flags & pangu::electron::heating_negative_dissipation_clipped) == 0)
    throw std::runtime_error("negative dissipation flag was not set");

  const auto suppressed = pangu::electron::ApplyConstantHeating(
      1.0, 0.02, 1.0, 0.4, 0.02, 0.01, gamma, gamma_e, gamma_p, 0.25, false, true, 2.0, 1.0, false,
      1.0e-3, 1.0e3);
  CheckClose(suppressed.applied_dissipation, 0.0, "magnetization suppression");
  CheckClose(suppressed.heating_fraction, 0.0, "effective suppressed fraction");
  CheckClose(suppressed.electron_entropy, 0.01, "suppressed electron entropy");
  if ((suppressed.flags & pangu::electron::heating_high_magnetization_suppressed) == 0)
    throw std::runtime_error("magnetization suppression flag was not set");
}

void CheckEntropyLimits() {
  constexpr Real gamma = 4.0 / 3.0;
  constexpr Real gamma_e = 4.0 / 3.0;
  constexpr Real gamma_p = 5.0 / 3.0;
  constexpr Real rho = 1.0;
  constexpr Real ktot = 0.1;
  constexpr Real ratio_min = 0.5;
  constexpr Real ratio_max = 20.0;
  const auto bounds = pangu::electron::ElectronEntropyBounds(rho, ktot, gamma, gamma_e, gamma_p,
                                                              ratio_min, ratio_max);
  const auto lower = pangu::electron::ApplyConstantHeating(rho, ktot, rho, 0.3, ktot, -1.0, gamma,
                                                            gamma_e, gamma_p, 0.0, false, false,
                                                            0.0, 1.0, true, ratio_min, ratio_max);
  const auto upper = pangu::electron::ApplyConstantHeating(rho, ktot, rho, 0.3, ktot, 1.0, gamma,
                                                            gamma_e, gamma_p, 0.0, false, false,
                                                            0.0, 1.0, true, ratio_min, ratio_max);
  CheckClose(lower.electron_entropy, bounds.minimum, "lower electron entropy limit");
  CheckClose(upper.electron_entropy, bounds.maximum, "upper electron entropy limit");
  if ((lower.flags & pangu::electron::heating_kel_min_applied) == 0 ||
      (upper.flags & pangu::electron::heating_kel_max_applied) == 0)
    throw std::runtime_error("electron entropy limit flag was not set");
}

} // namespace

int main() {
  try {
    CheckKHARMAConstantUpdate();
    CheckNegativeAndMagnetizationSuppression();
    CheckEntropyLimits();
    std::cout << "Electron dissipative Constant heating PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Electron dissipative Constant heating FAIL: " << error.what() << '\n';
    return 1;
  }
}
