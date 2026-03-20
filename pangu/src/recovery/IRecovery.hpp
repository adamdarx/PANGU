
#pragma once

#include <cstdint>

#include <parthenon/package.hpp>

using parthenon::Real;

enum Mnemonic { RHO, UU, U1, U2, U3, B1, B2, B3 };
constexpr int64_t kMaxNewtIter = 30;
constexpr Real kNewtTol = 1e-10;
constexpr Real kMinNewtTol = 1e-10;
constexpr int64_t kExtraNewtIter = 2;
constexpr Real kNewtTol2 = 1.0e-15;
constexpr Real kMinNewtTol2 = 1.0e-10;
constexpr Real kWTooBig = 1.e20;
constexpr Real kUtsqTooBig = 1.e20;
constexpr Real kFailVal = 1.e30;

class IRecovery {
    public:
    virtual ~IRecovery() = default;
    virtual int invert(Real U[8], Real prim[8], Real gamma) const = 0;

    KOKKOS_INLINE_FUNCTION
    static Real pressure_rho0_u(Real rho0, Real u, Real gamma) {
        (void)rho0;
        return (gamma - 1.) * u;
    }

    KOKKOS_INLINE_FUNCTION
    static Real pressure_rho0_w(Real rho0, Real w, Real gamma) {
        return (gamma - 1.) * (w - rho0) / gamma;
    }

    KOKKOS_INLINE_FUNCTION
    static void ncov_calc(const Real gcon[4][4], Real ncov[4]) {
        const Real lapse = Kokkos::sqrt(-1.0 / gcon[0][0]);
        for (int i = 0; i < 4; ++i) {
            ncov[i] = -lapse * (i == 0);
        }
    }

    KOKKOS_INLINE_FUNCTION
    static void raise_g(const Real vcov[4], const Real gcon[4][4], Real vcon[4]) {
        for (int i = 0; i < 4; ++i) {
            vcon[i] = 0.0;
            for (int j = 0; j < 4; ++j) {
                vcon[i] += gcon[i][j] * vcov[j];
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    static void lower_g(const Real vcon[4], const Real gcov[4][4], Real vcov[4]) {
        for (int i = 0; i < 4; ++i) {
            vcov[i] = 0.0;
            for (int j = 0; j < 4; ++j) {
                vcov[i] += gcov[i][j] * vcon[j];
            }
        }
    }
};