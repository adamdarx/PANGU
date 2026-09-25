#ifndef PANGU_Z4C_COMPONENT_INDICES_H_
#define PANGU_Z4C_COMPONENT_INDICES_H_

#include <cstddef>

namespace pangu::nr {

enum class Z4cComponent : int {
  chi = 0,
  gxx,
  gxy,
  gxz,
  gyy,
  gyz,
  gzz,
  khat,
  axx,
  axy,
  axz,
  ayy,
  ayz,
  azz,
  gamx,
  gamy,
  gamz,
  theta,
  alpha,
  betax,
  betay,
  betaz,
  count
};

enum class ADMComponent : int {
  gxx = 0,
  gxy,
  gxz,
  gyy,
  gyz,
  gzz,
  kxx,
  kxy,
  kxz,
  kyy,
  kyz,
  kzz,
  psi4,
  count
};

enum class ConstraintComponent : int {
  combined = 0,
  hamiltonian,
  momentum_norm,
  z_norm,
  momentum_x,
  momentum_y,
  momentum_z,
  theta,
  count
};

enum class StressEnergyComponent : int {
  sxx = 0,
  sxy,
  sxz,
  syy,
  syz,
  szz,
  energy,
  momentum_x,
  momentum_y,
  momentum_z,
  count
};

enum class WeylComponent : int { psi4_real = 0, psi4_imag, count };

template <class Component> constexpr int Index(const Component component) {
  return static_cast<int>(component);
}

inline constexpr int kZ4cComponents = Index(Z4cComponent::count);
inline constexpr int kADMComponents = Index(ADMComponent::count);
inline constexpr int kConstraintComponents = Index(ConstraintComponent::count);
inline constexpr int kStressEnergyComponents = Index(StressEnergyComponent::count);
inline constexpr int kWeylComponents = Index(WeylComponent::count);
inline constexpr int kADMMetricDerivativeComponents = 3 * 6;

constexpr int SpatialSymmetricComponent(int first, int second) {
  if (second < first) {
    const int temporary = first;
    first = second;
    second = temporary;
  }
  return first * 3 - first * (first - 1) / 2 + second - first;
}

constexpr int ADMMetricDerivativeComponent(const int derivative, const int first,
                                           const int second) {
  return 6 * derivative + SpatialSymmetricComponent(first, second);
}

static_assert(kZ4cComponents == 22);
static_assert(kADMComponents == 13);
static_assert(kConstraintComponents == 8);
static_assert(kStressEnergyComponents == 10);
static_assert(kWeylComponents == 2);
static_assert(kADMMetricDerivativeComponents == 18);
static_assert(ADMMetricDerivativeComponent(0, 0, 0) == 0);
static_assert(ADMMetricDerivativeComponent(2, 2, 2) == 17);
static_assert(SpatialSymmetricComponent(0, 0) == 0);
static_assert(SpatialSymmetricComponent(0, 2) == 2);
static_assert(SpatialSymmetricComponent(1, 1) == 3);
static_assert(SpatialSymmetricComponent(2, 2) == 5);

} // namespace pangu::nr

#endif
