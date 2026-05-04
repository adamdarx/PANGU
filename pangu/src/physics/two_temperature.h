// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#ifndef PANGU_SRC_PHYSICS_TWOTEMPERATURE_H
#define PANGU_SRC_PHYSICS_TWOTEMPERATURE_H

#include <algorithm>
#include <cctype>
#include <parthenon/package.hpp>
#include <stdexcept>
#include <string>

namespace two_temperature {

enum MODEL {
  CONSTANT = 0,
  HOWES = 1,
  KAWAZURA = 2,
  WERNER = 3,
  ROWAN = 4,
  SHARMA = 5
};

inline std::string ToLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

inline MODEL StringToMODEL(const std::string &model_name) {
  const std::string name = ToLower(model_name);
  if (name == "constant") return MODEL::CONSTANT;
  if (name == "howes") return MODEL::HOWES;
  if (name == "kawazura") return MODEL::KAWAZURA;
  if (name == "werner") return MODEL::WERNER;
  if (name == "rowan") return MODEL::ROWAN;
  if (name == "sharma") return MODEL::SHARMA;
  throw std::invalid_argument(
      "Invalid electrons/model. Allowed values: constant, howes, kawazura, "
      "werner, rowan, sharma.");
}

inline std::string MODELToString(const MODEL model) {
  switch (model) {
    case MODEL::CONSTANT:
      return "constant";
    case MODEL::HOWES:
      return "howes";
    case MODEL::KAWAZURA:
      return "kawazura";
    case MODEL::WERNER:
      return "werner";
    case MODEL::ROWAN:
      return "rowan";
    case MODEL::SHARMA:
      return "sharma";
  }
  throw std::invalid_argument("Invalid MODEL enum value.");
}

KOKKOS_INLINE_FUNCTION
parthenon::Real ComputeTotalEntropy(const parthenon::Real gamma,
                                    const parthenon::Real rho,
                                    const parthenon::Real energy) {
  constexpr parthenon::Real kSmall = 1.0e-20;
  return (gamma - 1.0) * energy * Kokkos::pow(Kokkos::max(rho, kSmall), -gamma);
}

KOKKOS_INLINE_FUNCTION
parthenon::Real RecoverAdvectedScalar(const parthenon::Real cons_rho,
                                      const parthenon::Real cons_scalar) {
  constexpr parthenon::Real kSmall = 1.0e-20;
  return cons_scalar / Kokkos::max(cons_rho, kSmall);
}

KOKKOS_INLINE_FUNCTION
parthenon::Real Clip(const parthenon::Real value, const parthenon::Real min_value,
                     const parthenon::Real max_value) {
  return Kokkos::max(min_value, Kokkos::min(value, max_value));
}

KOKKOS_INLINE_FUNCTION
parthenon::Real ComputeHeatingFraction(
    const MODEL heating_model, const parthenon::Real gamma,
    const parthenon::Real gamma_p, const parthenon::Real gamma_e,
    const parthenon::Real rho, const parthenon::Real energy,
    const parthenon::Real bsq, const parthenon::Real electron_entropy,
    const parthenon::Real fel_constant) {
  constexpr parthenon::Real kSmall = 1.0e-20;
  constexpr parthenon::Real kElectronMass = 9.1093826e-28;
  constexpr parthenon::Real kProtonMass = 1.67262171e-24;
  const parthenon::Real rho_safe = Kokkos::max(rho, kSmall);
  const parthenon::Real bsq_safe = Kokkos::max(bsq, kSmall);
  const parthenon::Real energy_safe = Kokkos::max(energy, kSmall);
  const parthenon::Real tpr =
      Kokkos::max((gamma_p - 1.0) * energy_safe / rho_safe, kSmall);
  const parthenon::Real tel = Kokkos::max(
      electron_entropy * Kokkos::pow(rho_safe, gamma_e - 1.0), kSmall);

  parthenon::Real fel = fel_constant;
  if (heating_model == MODEL::HOWES) {
    const parthenon::Real trat = tpr / tel;
    const parthenon::Real pres = rho_safe * tpr;
    const parthenon::Real beta = Kokkos::min(2.0 * pres / bsq_safe, 1.0e20);
    const parthenon::Real log_trat = Kokkos::log(trat) / Kokkos::log(10.0);
    const parthenon::Real mbeta = 2.0 - 0.2 * log_trat;
    const parthenon::Real c2 = (trat <= 1.0) ? 1.6 / trat : 1.2 / trat;
    const parthenon::Real c3 = (trat <= 1.0) ? 18.0 + 5.0 * log_trat : 18.0;
    const parthenon::Real beta_pow = Kokkos::pow(beta, mbeta);
    const parthenon::Real qrat =
        0.92 * (c2 * c2 + beta_pow) / (c3 * c3 + beta_pow) *
        Kokkos::exp(-1.0 / beta) *
        Kokkos::sqrt(kProtonMass / kElectronMass * trat);
    fel = 1.0 / (1.0 + qrat);
  } else if (heating_model == MODEL::KAWAZURA) {
    const parthenon::Real trat = tpr / tel;
    const parthenon::Real pres = rho_safe * tpr;
    const parthenon::Real beta = Kokkos::min(2.0 * pres / bsq_safe, 1.0e20);
    const parthenon::Real qi_qe =
        35.0 / (1.0 + Kokkos::pow(beta / 15.0, -1.4) *
                          Kokkos::exp(-0.1 / trat));
    fel = 1.0 / (1.0 + qi_qe);
  } else if (heating_model == MODEL::WERNER) {
    const parthenon::Real sigma = bsq_safe / rho_safe;
    fel = 0.25 * (1.0 + Kokkos::sqrt((sigma / 5.0) / (2.0 + sigma / 5.0)));
  } else if (heating_model == MODEL::ROWAN) {
    const parthenon::Real pres = (gamma_p - 1.0) * energy_safe;
    const parthenon::Real pg = (gamma - 1.0) * energy_safe;
    const parthenon::Real beta = 2.0 * pres / bsq_safe;
    const parthenon::Real sigma = bsq_safe / (rho_safe + energy_safe + pg);
    const parthenon::Real betamax = 0.25 / Kokkos::max(sigma, kSmall);
    fel = 0.5 * Kokkos::exp(
                    -Kokkos::pow(1.0 - beta / betamax, 3.3) /
                    (1.0 + 1.2 * Kokkos::pow(sigma, 0.7)));
  } else if (heating_model == MODEL::SHARMA) {
    const parthenon::Real trat_inv = tel / tpr;
    const parthenon::Real qe_qi = 0.33 * Kokkos::sqrt(trat_inv);
    fel = 1.0 / (1.0 + 1.0 / Kokkos::max(qe_qi, kSmall));
  }
  return fel;
}

KOKKOS_INLINE_FUNCTION
parthenon::Real ComputeModelElectronEntropy(
    const MODEL heating_model,
    const parthenon::Real gamma,
    const parthenon::Real gamma_p,
    const parthenon::Real gamma_e,
    const parthenon::Real rho,
    const parthenon::Real energy,
    const parthenon::Real bsq,
    const parthenon::Real advected_total_entropy,
    const parthenon::Real recovered_total_entropy,
    const parthenon::Real advected_electron_entropy,
    const parthenon::Real fel_constant,
    const bool limit_kel,
    const parthenon::Real ratio_min,
    const parthenon::Real ratio_max,
    const bool suppress_highb_heat,
    const bool enforce_positive_dissipation) {
  constexpr parthenon::Real kSmall = 1.0e-20;
  const parthenon::Real rho_safe = Kokkos::max(rho, kSmall);
  const parthenon::Real fel = ComputeHeatingFraction(
      heating_model, gamma, gamma_p, gamma_e, rho_safe, energy, bsq,
      advected_electron_entropy, fel_constant);

  parthenon::Real dissipation =
      (gamma_e - 1.0) / (gamma - 1.0) * Kokkos::pow(rho_safe, gamma - gamma_e) *
      (recovered_total_entropy - advected_total_entropy);
  if (suppress_highb_heat && (bsq / rho_safe > 1.0)) {
    dissipation = 0.0;
  }
  if (enforce_positive_dissipation) {
    dissipation = Kokkos::max(dissipation, 0.0);
  }

  parthenon::Real kel = advected_electron_entropy + fel * dissipation;
  if (!limit_kel && heating_model == MODEL::CONSTANT) {
    return kel;
  }

  const parthenon::Real kel_max =
      advected_total_entropy * Kokkos::pow(rho_safe, gamma - gamma_e) /
      (ratio_min * (gamma - 1.0) / (gamma_p - 1.0) +
       (gamma - 1.0) / (gamma_e - 1.0));
  const parthenon::Real kel_min =
      advected_total_entropy * Kokkos::pow(rho_safe, gamma - gamma_e) /
      (ratio_max * (gamma - 1.0) / (gamma_p - 1.0) +
       (gamma - 1.0) / (gamma_e - 1.0));
  return Clip(kel, kel_min, kel_max);
}

KOKKOS_INLINE_FUNCTION
parthenon::Real ClampElectronEntropyByRatio(const parthenon::Real total_entropy,
                                            const parthenon::Real electron_entropy,
                                            const parthenon::Real ratio_min,
                                            const parthenon::Real ratio_max) {
  constexpr parthenon::Real kSmall = 1.0e-20;
  const parthenon::Real rmin = Kokkos::max(ratio_min, kSmall);
  const parthenon::Real rmax = Kokkos::max(ratio_max, rmin);

  const parthenon::Real kel_low = total_entropy / (1.0 + rmax);
  const parthenon::Real kel_high = total_entropy / (1.0 + rmin);
  const parthenon::Real kel_bounded = Kokkos::max(0.0, Kokkos::min(electron_entropy, total_entropy));
  return Kokkos::max(kel_low, Kokkos::min(kel_bounded, kel_high));
}

}  // namespace two_temperature

#endif  // PANGU_SRC_PHYSICS_TWOTEMPERATURE_H
