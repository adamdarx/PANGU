// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#ifndef PANGU_SRC_PHYSICS_HEATINGMODEL_H
#define PANGU_SRC_PHYSICS_HEATINGMODEL_H

#include <parthenon/package.hpp>
#include <stdexcept>
#include <string>

using parthenon::Real;

enum Model {
  CONSTANT = 0,
  HOWES = 1,
  KAWAZURA = 2,
  WERNER = 3,
  ROWAN = 4,
  SHARMA = 5
};

struct RatioLimits {
  Real min;
  Real max;
};

struct Parameters {
  Model model;
  Real gamma;
  Real gamma_p;
  Real gamma_e;
  Real fel_constant;
  RatioLimits ratio;
  bool limit_kel;
  bool suppress_highb_heat;
  bool enforce_positive_dissipation;
};

struct CellState {
  Real rho;
  Real energy;
  Real bsq;
  Real advected_total_entropy;
  Real recovered_total_entropy;
  Real advected_electron_entropy;
};

KOKKOS_INLINE_FUNCTION
Real clip(const Real value, const Real min_value, const Real max_value) {
  return Kokkos::max(min_value, Kokkos::min(value, max_value));
}

KOKKOS_INLINE_FUNCTION
Real computeHeatingFraction(const Parameters params, const CellState cell) {
  constexpr Real kElectronMass = 9.1093826e-28;
  constexpr Real kProtonMass = 1.67262171e-24;
  const Real tpr = (params.gamma_p - 1.0) * cell.energy / cell.rho;
  const Real tel =
      cell.advected_electron_entropy * Kokkos::pow(cell.rho, params.gamma_e - 1.0);

  switch (params.model) {
    case Model::CONSTANT:
      return params.fel_constant;
    case Model::HOWES: {
      const Real trat = tpr / tel;
      const Real pres = cell.rho * tpr;
      const Real beta = Kokkos::min(2.0 * pres / cell.bsq, 1.0e20);
      const Real log_trat = Kokkos::log(trat) / Kokkos::log(10.0);
      const Real mbeta = 2.0 - 0.2 * log_trat;
      const Real c2 = (trat <= 1.0) ? 1.6 / trat : 1.2 / trat;
      const Real c3 = (trat <= 1.0) ? 18.0 + 5.0 * log_trat : 18.0;
      const Real beta_pow = Kokkos::pow(beta, mbeta);
      const Real qrat =
          0.92 * (c2 * c2 + beta_pow) / (c3 * c3 + beta_pow) *
          Kokkos::exp(-1.0 / beta) *
          Kokkos::sqrt(kProtonMass / kElectronMass * trat);
      return 1.0 / (1.0 + qrat);
    }
    case Model::KAWAZURA: {
      const Real trat = tpr / tel;
      const Real pres = cell.rho * tpr;
      const Real beta = Kokkos::min(2.0 * pres / cell.bsq, 1.0e20);
      const Real qi_qe =
          35.0 / (1.0 + Kokkos::pow(beta / 15.0, -1.4) *
                            Kokkos::exp(-0.1 / trat));
      return 1.0 / (1.0 + qi_qe);
    }
    case Model::WERNER: {
      const Real sigma = cell.bsq / cell.rho;
      return 0.25 * (1.0 + Kokkos::sqrt((sigma / 5.0) / (2.0 + sigma / 5.0)));
    }
    case Model::ROWAN: {
      const Real pres = (params.gamma_p - 1.0) * cell.energy;
      const Real pg = (params.gamma - 1.0) * cell.energy;
      const Real beta = 2.0 * pres / cell.bsq;
      const Real sigma = cell.bsq / (cell.rho + cell.energy + pg);
      const Real betamax = 0.25 / sigma;
      return 0.5 * Kokkos::exp(
                       -Kokkos::pow(1.0 - beta / betamax, 3.3) /
                       (1.0 + 1.2 * Kokkos::pow(sigma, 0.7)));
    }
    case Model::SHARMA: {
      const Real trat_inv = tel / tpr;
      const Real qe_qi = 0.33 * Kokkos::sqrt(trat_inv);
      return 1.0 / (1.0 + 1.0 / qe_qi);
    }
  }
  return params.fel_constant;
}

inline Model parseModel(const std::string &model_name) {
  if (model_name == "constant") return Model::CONSTANT;
  if (model_name == "howes") return Model::HOWES;
  if (model_name == "kawazura") return Model::KAWAZURA;
  if (model_name == "werner") return Model::WERNER;
  if (model_name == "rowan") return Model::ROWAN;
  if (model_name == "sharma") return Model::SHARMA;
  throw std::invalid_argument(
      "Invalid electrons/model. Allowed values: constant, howes, kawazura, "
      "werner, rowan, sharma.");
}

inline std::string modelName(const Model model) {
  switch (model) {
    case Model::CONSTANT:
      return "constant";
    case Model::HOWES:
      return "howes";
    case Model::KAWAZURA:
      return "kawazura";
    case Model::WERNER:
      return "werner";
    case Model::ROWAN:
      return "rowan";
    case Model::SHARMA:
      return "sharma";
  }
  throw std::invalid_argument("Invalid heating model enum value.");
}

KOKKOS_INLINE_FUNCTION
Real apply(const Parameters params, const CellState cell) {
  const Real fel = computeHeatingFraction(params, cell);

  Real dissipation =
      (params.gamma_e - 1.0) / (params.gamma - 1.0) *
      Kokkos::pow(cell.rho, params.gamma - params.gamma_e) *
      (cell.recovered_total_entropy - cell.advected_total_entropy);
  if (params.suppress_highb_heat && (cell.bsq / cell.rho > 1.0)) {
    dissipation = 0.0;
  }
  if (params.enforce_positive_dissipation) {
    dissipation = Kokkos::max(dissipation, 0.0);
  }

  Real kel = cell.advected_electron_entropy + fel * dissipation;
  if (!params.limit_kel && params.model == Model::CONSTANT) {
    return kel;
  }

  const Real kel_max =
      cell.advected_total_entropy * Kokkos::pow(cell.rho, params.gamma - params.gamma_e) /
      (params.ratio.min * (params.gamma - 1.0) / (params.gamma_p - 1.0) +
       (params.gamma - 1.0) / (params.gamma_e - 1.0));
  const Real kel_min =
      cell.advected_total_entropy * Kokkos::pow(cell.rho, params.gamma - params.gamma_e) /
      (params.ratio.max * (params.gamma - 1.0) / (params.gamma_p - 1.0) +
       (params.gamma - 1.0) / (params.gamma_e - 1.0));
  return clip(kel, kel_min, kel_max);
}

KOKKOS_INLINE_FUNCTION
Real clampByRatio(const RatioLimits ratio, const Real total_entropy,
                  const Real electron_entropy) {
  const Real kel_low = total_entropy / (1.0 + ratio.max);
  const Real kel_high = total_entropy / (1.0 + ratio.min);
  const Real kel_bounded = Kokkos::max(0.0, Kokkos::min(electron_entropy, total_entropy));
  return Kokkos::max(kel_low, Kokkos::min(kel_bounded, kel_high));
}

#endif  // PANGU_SRC_PHYSICS_HEATINGMODEL_H
