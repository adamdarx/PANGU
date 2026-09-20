#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include "mhd/passive_transport.h"

namespace {

using parthenon::Real;

void CheckClose(const Real actual, const Real expected, const char* label) {
  const Real scale = fmax(1.0, fmax(fabs(actual), fabs(expected)));
  if (fabs(actual - expected) > 4.0e-15 * scale)
    throw std::runtime_error(std::string(label) + " mismatch");
}

void CheckLLF() {
  constexpr Real density_left = 2.0;
  constexpr Real density_right = 0.75;
  constexpr Real mass_flux_left = 0.6;
  constexpr Real mass_flux_right = -0.15;
  constexpr Real speed = 0.8;
  constexpr Real entropy_left = 1.25;
  constexpr Real entropy_right = 0.4;
  const auto coefficients = pangu::mhd::LLFPassiveTransport(density_left, density_right,
                                                            mass_flux_left, mass_flux_right, speed);
  const Real scalar_flux = coefficients.left * entropy_left + coefficients.right * entropy_right;
  const Real expected =
      0.5 * (mass_flux_left * entropy_left + mass_flux_right * entropy_right -
             speed * (density_right * entropy_right - density_left * entropy_left));
  CheckClose(scalar_flux, expected, "LLF passive flux");
  CheckClose(coefficients.left + coefficients.right,
             0.5 * (mass_flux_left + mass_flux_right - speed * (density_right - density_left)),
             "LLF constant-state mass flux");
}

void CheckHLLEInterior() {
  constexpr Real density_left = 1.4;
  constexpr Real density_right = 0.9;
  constexpr Real mass_flux_left = 0.35;
  constexpr Real mass_flux_right = -0.2;
  constexpr Real lambda_left = -0.7;
  constexpr Real lambda_right = 0.9;
  constexpr Real entropy_left = 0.8;
  constexpr Real entropy_right = 1.1;
  const auto coefficients = pangu::mhd::HLLEPassiveTransport(
      density_left, density_right, mass_flux_left, mass_flux_right, lambda_left, lambda_right);
  const Real wave_product = lambda_right * lambda_left;
  const Real inverse = 1.0 / (lambda_right - lambda_left);
  const Real expected =
      (lambda_right * mass_flux_left * entropy_left -
       lambda_left * mass_flux_right * entropy_right +
       wave_product * (density_right * entropy_right - density_left * entropy_left)) *
      inverse;
  CheckClose(coefficients.left * entropy_left + coefficients.right * entropy_right, expected,
             "HLLE passive flux");
  const Real mass_hlle = (lambda_right * mass_flux_left - lambda_left * mass_flux_right +
                          wave_product * (density_right - density_left)) *
                         inverse;
  CheckClose(coefficients.left + coefficients.right, mass_hlle, "HLLE constant-state mass flux");
}

void CheckHLLEUpwindBranches() {
  const auto left = pangu::mhd::HLLEPassiveTransport(1.0, 2.0, 0.4, 0.8, 0.1, 0.9);
  CheckClose(left.left, 0.4, "HLLE positive left coefficient");
  CheckClose(left.right, 0.0, "HLLE positive right coefficient");
  const auto right = pangu::mhd::HLLEPassiveTransport(1.0, 2.0, -0.4, -0.8, -0.9, -0.1);
  CheckClose(right.left, 0.0, "HLLE negative left coefficient");
  CheckClose(right.right, -0.8, "HLLE negative right coefficient");
}

} // namespace

int main() {
  try {
    CheckLLF();
    CheckHLLEInterior();
    CheckHLLEUpwindBranches();
    std::cout << "Electron passive transport coefficients PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Electron passive transport coefficients FAIL: " << error.what() << '\n';
    return 1;
  }
}
