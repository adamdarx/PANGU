#ifndef PANGU_Z4C_STRESS_ENERGY_H_
#define PANGU_Z4C_STRESS_ENERGY_H_

#include <parthenon/parthenon.hpp>

#include "z4c/core/component_indices.h"

namespace pangu::nr {

// Undensitized ADM projection of the stress-energy tensor.  The component
// order is intentionally identical to AthenaK Tmunu: six symmetric S_ij
// components, followed by E and the three covariant momenta S_i.
struct StressEnergy {
  parthenon::Real stress[6]{};
  parthenon::Real energy = 0.0;
  parthenon::Real momentum[3]{};
};

enum class StressEnergySource : int { vacuum = 0, zero, constant, analytic, grhd, grmhd };

struct StressEnergySourceOptions {
  StressEnergySource source = StressEnergySource::vacuum;
  parthenon::Real constant[kStressEnergyComponents]{};
  parthenon::Real amplitude = 0.0;
  parthenon::Real wave_number[3]{1.0, 0.0, 0.0};
  parthenon::Real angular_frequency = 1.0;
};

constexpr bool HasStressEnergyField(const StressEnergySource source) {
  return source != StressEnergySource::vacuum;
}

template <class Accessor>
KOKKOS_INLINE_FUNCTION StressEnergy LoadStressEnergy(const Accessor& value) {
  StressEnergy result{};
  for (int component = 0; component < 6; ++component)
    result.stress[component] = value(component);
  result.energy = value(Index(StressEnergyComponent::energy));
  for (int axis = 0; axis < 3; ++axis)
    result.momentum[axis] = value(Index(StressEnergyComponent::momentum_x) + axis);
  return result;
}

template <class Pack>
KOKKOS_INLINE_FUNCTION StressEnergy LoadStressEnergy(const Pack& values, const int block,
                                                     const int k, const int j, const int i) {
  return LoadStressEnergy(
      [&](const int component) { return values(block, component, k, j, i); });
}

KOKKOS_INLINE_FUNCTION StressEnergy
EvaluateStressEnergySource(const StressEnergySourceOptions& options, const parthenon::Real x,
                           const parthenon::Real y, const parthenon::Real z,
                           const parthenon::Real time) {
  if (options.source == StressEnergySource::vacuum || options.source == StressEnergySource::zero)
    return {};
  if (options.source == StressEnergySource::constant)
    return LoadStressEnergy([&](const int component) { return options.constant[component]; });
  if (options.source != StressEnergySource::analytic)
    return {};

  const parthenon::Real phase = options.wave_number[0] * x + options.wave_number[1] * y +
                                options.wave_number[2] * z - options.angular_frequency * time;
  return LoadStressEnergy([&](const int component) {
    return options.amplitude * static_cast<parthenon::Real>(component + 1) *
           sin(phase + 0.125 * static_cast<parthenon::Real>(component));
  });
}

static_assert(Index(StressEnergyComponent::sxx) == 0);
static_assert(Index(StressEnergyComponent::sxy) == 1);
static_assert(Index(StressEnergyComponent::sxz) == 2);
static_assert(Index(StressEnergyComponent::syy) == 3);
static_assert(Index(StressEnergyComponent::syz) == 4);
static_assert(Index(StressEnergyComponent::szz) == 5);
static_assert(Index(StressEnergyComponent::energy) == 6);
static_assert(Index(StressEnergyComponent::momentum_x) == 7);
static_assert(Index(StressEnergyComponent::momentum_y) == 8);
static_assert(Index(StressEnergyComponent::momentum_z) == 9);

} // namespace pangu::nr

#endif
