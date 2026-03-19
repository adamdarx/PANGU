#pragma once
#include <basic_types.hpp>

KOKKOS_INLINE_FUNCTION
parthenon::Real InterpolateMC(const parthenon::Real LeftValue, const parthenon::Real CenterValue, const parthenon::Real RightValue) {
    const parthenon::Real ForwardDifference = 2. * (CenterValue - LeftValue);      
    const parthenon::Real BackwardDifference = 2. * (RightValue - CenterValue);      
    const parthenon::Real CenterDifference = 0.5 * (RightValue - LeftValue);
    const parthenon::Real Sign = ForwardDifference * BackwardDifference;
    
    if (Sign <= 0.)
        return 0.;
    else {
        if (Kokkos::abs(ForwardDifference) < Kokkos::abs(BackwardDifference) && Kokkos::abs(ForwardDifference) < Kokkos::abs(CenterDifference))
            return (ForwardDifference);  
        else if (Kokkos::abs(BackwardDifference) < Kokkos::abs(CenterDifference))
            return (BackwardDifference);  
        else
            return (CenterDifference);  
    }
}
