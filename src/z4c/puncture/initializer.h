#ifndef PANGU_Z4C_PUNCTURE_INITIALIZER_H_
#define PANGU_Z4C_PUNCTURE_INITIALIZER_H_

#include <vector>

#include "z4c/core/adm_conversion.h"
#include "z4c/puncture/interpolation.h"

namespace pangu::nr::puncture {

// Construct maximal, conformally flat physical ADM data from a converged
// puncture solution. This is the sole boundary between the native elliptic
// solver and the Z4c evolution variables.
bool BuildPunctureADM(double x, double y, double z,
                      const std::vector<Puncture> &punctures,
                      const SpectralInterpolator &interpolator,
                      ADMState &output);

// ADM mass measured at each additional asymptotically flat puncture end.
// For psi = m_i/(2 r_i) + C_i + O(r_i), inversion about that end gives
// M_i = m_i C_i, where C_i includes the regular correction and the other
// punctures' singular fields evaluated at puncture i.
std::vector<double>
PunctureEndADMMasses(const std::vector<Puncture> &punctures,
                     const SpectralInterpolator &interpolator);

// Adjust bare masses until the internal-end ADM masses match target_masses.
// On success, punctures contains the tuned bare masses and the returned
// Hamiltonian solution is the constraint-solved data for those masses.
HamiltonianSolution
SolveTargetADMMasses(std::vector<Puncture> &punctures,
                     const std::vector<double> &target_masses,
                     const HamiltonianSolveOptions &solve_options,
                     double mass_tolerance, int maximum_mass_iterations);

} // namespace pangu::nr::puncture

#endif
