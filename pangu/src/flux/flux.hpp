#pragma once
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <basic_types.hpp>
#include <parthenon/package.hpp>

#include "../initialize/mnemonic.hpp"
#include "../physics/alfven_velocity.hpp"
#include "../physics/stress_tensor.hpp"
#include "../reconstruct/InterpolaterMC.hpp"

KOKKOS_INLINE_FUNCTION
parthenon::Real LAXFFluxSRMHD(const parthenon::Real leftFlux, const parthenon::Real rightFlux,
                              const parthenon::Real leftConservative,
                              const parthenon::Real rightConservative,
                              const parthenon::Real alpha) {
    return 0.5 * (leftFlux + rightFlux - alpha * (rightConservative - leftConservative));
}

KOKKOS_INLINE_FUNCTION
parthenon::Real HLLFluxSRMHD(const parthenon::Real leftFlux, const parthenon::Real rightFlux,
                             const parthenon::Real leftConservative,
                             const parthenon::Real rightConservative,
                             const parthenon::Real alpha) {
    return LAXFFluxSRMHD(leftFlux, rightFlux, leftConservative, rightConservative, alpha);
}

KOKKOS_INLINE_FUNCTION
parthenon::Real HLLFluxGRMHD(const parthenon::Real leftFlux, const parthenon::Real rightFlux,
                             const parthenon::Real leftConservative,
                             const parthenon::Real rightConservative,
                             const parthenon::Real alpha) {
    return LAXFFluxSRMHD(leftFlux, rightFlux, leftConservative, rightConservative, alpha);
}

KOKKOS_INLINE_FUNCTION
parthenon::Real ComputeAlfvenVelocityCenter(
    const parthenon::Real adiabatic_index,
    const parthenon::Real primitive_left_c_array[PrimitiveVariableNumber],
    const parthenon::Real primitive_right_c_array[PrimitiveVariableNumber],
    const int direction, const int is_gr) {
    parthenon::Real maximum_alfven_velocity_left, maximum_alfven_velocity_right;
    parthenon::Real minimum_alfven_velocity_left, minimum_alfven_velocity_right;
    if (is_gr) {
        CalculateAlfvenVelocityGRMHD(adiabatic_index, primitive_left_c_array, direction,
                                     maximum_alfven_velocity_left,
                                     minimum_alfven_velocity_left);
        CalculateAlfvenVelocityGRMHD(adiabatic_index, primitive_right_c_array, direction,
                                     maximum_alfven_velocity_right,
                                     minimum_alfven_velocity_right);
    } else {
        CalculateAlfvenVelocitySRMHD(adiabatic_index, primitive_left_c_array, direction,
                                     maximum_alfven_velocity_left,
                                     minimum_alfven_velocity_left);
        CalculateAlfvenVelocitySRMHD(adiabatic_index, primitive_right_c_array, direction,
                                     maximum_alfven_velocity_right,
                                     minimum_alfven_velocity_right);
    }

    const auto maximum_alfven_velocity_center =
        Kokkos::fabs(Kokkos::max(Kokkos::max(0., maximum_alfven_velocity_left),
                                 maximum_alfven_velocity_right));
    const auto minimum_alfven_velocity_center =
        Kokkos::fabs(Kokkos::max(Kokkos::max(0., -minimum_alfven_velocity_left),
                                 -minimum_alfven_velocity_right));
    return Kokkos::max(maximum_alfven_velocity_center,
                       minimum_alfven_velocity_center);
}

parthenon::TaskStatus CalculateFluxes(std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource) {
    using namespace parthenon;
    PARTHENON_INSTRUMENT

    const auto meshblock_pointer = resource->GetBlockPointer();
    const auto package = meshblock_pointer->packages.Get("PANGU");
    const auto &adiabatic_index = package->Param<Real>("AdiabaticIndex");
    const auto mode = package->Param<std::string>("Mode");
    const int is_gr = (mode == "GR");

    const auto bound_x1 = meshblock_pointer->cellbounds.GetBoundsI(IndexDomain::interior);
    const auto bound_x2 = meshblock_pointer->cellbounds.GetBoundsJ(IndexDomain::interior);
    const auto bound_x3 = meshblock_pointer->cellbounds.GetBoundsK(IndexDomain::interior);

    PackIndexMap primitive_index_map;
    const std::vector<std::string> primitive_tags = {"Density", "Energy", "WeightedVelocity", "MagneticField"};
    const auto primitive = resource->PackVariables(primitive_tags, primitive_index_map);
    PackIndexMap conservative_index_map;
    const std::vector<std::string> conservative_tags = {"Conservative"};
    auto conservative = resource->PackVariablesAndFluxes(conservative_tags, conservative_index_map);
    
    const int scratch_level = 1;
    const auto meshgrid_size_x1 = meshblock_pointer->cellbounds.ncellsi(IndexDomain::entire);
    const auto meshgrid_size_x2 = meshblock_pointer->cellbounds.ncellsj(IndexDomain::entire);
    const auto meshgrid_size_x3 = meshblock_pointer->cellbounds.ncellsk(IndexDomain::entire);
    
    const size_t scratch_size_in_bytes_x1 = ScratchPad2D<Real>::shmem_size(PrimitiveVariableNumber, meshgrid_size_x1);
    const size_t scratch_size_in_bytes_x2 = ScratchPad2D<Real>::shmem_size(PrimitiveVariableNumber, meshgrid_size_x2);
    const size_t scratch_size_in_bytes_x3 = ScratchPad2D<Real>::shmem_size(PrimitiveVariableNumber, meshgrid_size_x3);

    const int offset_x1 = (meshgrid_size_x1 > 1) ? 1 : 0;
    const int offset_x2 = (meshgrid_size_x2 > 1) ? 1 : 0;
    const int offset_x3 = (meshgrid_size_x3 > 1) ? 1 : 0;

    meshblock_pointer->par_for_outer(
        PARTHENON_AUTO_LABEL, 2 * scratch_size_in_bytes_x1, scratch_level, bound_x3.s - offset_x3, bound_x3.e + offset_x3, bound_x2.s - offset_x2, bound_x2.e + offset_x2,
        KOKKOS_LAMBDA(team_mbr_t member, const int k, const int j) {
            ScratchPad2D<Real> primitive_left(member.team_scratch(scratch_level), PrimitiveVariableNumber, meshgrid_size_x1);
            ScratchPad2D<Real> primitive_right(member.team_scratch(scratch_level), PrimitiveVariableNumber, meshgrid_size_x1);

            par_for_inner(member, 0, PrimitiveVariableNumber - 1, bound_x1.s, bound_x1.e + 1, [&](const int n, const int i) {
                primitive_left(n, i) = primitive(n, k, j, i - 1) + 0.5 * InterpolateMC(primitive(n, k, j, i - 2), primitive(n, k, j, i - 1), primitive(n, k, j, i));
                primitive_right(n, i) = primitive(n, k, j, i) - 0.5 * InterpolateMC(primitive(n, k, j, i - 1), primitive(n, k, j, i), primitive(n, k, j, i + 1));
            });
            
            member.team_barrier();

            par_for_inner(member, bound_x1.s, bound_x1.e + 1, [&](const int i) {
                Real directed_energy_momentum_tensor_left[4];
                Real directed_energy_momentum_tensor_right[4];
                Real conservative_left[PrimitiveVariableNumber];
                Real conservative_right[PrimitiveVariableNumber];
                Real flux_left[PrimitiveVariableNumber];
                Real flux_right[PrimitiveVariableNumber];
                const Real primitive_left_c_array[PrimitiveVariableNumber] = {
                    primitive_left(DensityIndex, i),
                    primitive_left(EnergyIndex, i),
                    primitive_left(WeightedVelocityX1, i),
                    primitive_left(WeightedVelocityX2, i),
                    primitive_left(WeightedVelocityX3, i),
                    primitive_left(MagneticFieldX1, i),
                    primitive_left(MagneticFieldX2, i),
                    primitive_left(MagneticFieldX3, i),
                };
                const Real primitive_right_c_array[PrimitiveVariableNumber] = {
                    primitive_right(DensityIndex, i),
                    primitive_right(EnergyIndex, i),
                    primitive_right(WeightedVelocityX1, i),
                    primitive_right(WeightedVelocityX2, i),
                    primitive_right(WeightedVelocityX3, i),
                    primitive_right(MagneticFieldX1, i),
                    primitive_right(MagneticFieldX2, i),
                    primitive_right(MagneticFieldX3, i),
                };

                const auto squared_weighted_velocity_left = Kokkos::pow(primitive_left_c_array[WeightedVelocityX1], 2) + Kokkos::pow(primitive_left_c_array[WeightedVelocityX2], 2) + Kokkos::pow(primitive_left_c_array[WeightedVelocityX3], 2);
                const auto squared_lorentz_factor_left = 1 + squared_weighted_velocity_left;
                const auto lorentz_factor_left = Kokkos::sqrt(squared_lorentz_factor_left);
                const auto magnetic_field_three_vector_dot_weighted_velocity_left = primitive_left_c_array[WeightedVelocityX1] * primitive_left_c_array[MagneticFieldX1] + primitive_left_c_array[WeightedVelocityX2] * primitive_left_c_array[MagneticFieldX2] + primitive_left_c_array[WeightedVelocityX3] * primitive_left_c_array[MagneticFieldX3];

                const auto squared_weighted_velocity_right = Kokkos::pow(primitive_right_c_array[WeightedVelocityX1], 2) + Kokkos::pow(primitive_right_c_array[WeightedVelocityX2], 2) + Kokkos::pow(primitive_right_c_array[WeightedVelocityX3], 2);
                const auto squared_lorentz_factor_right = 1 + squared_weighted_velocity_right;
                const auto lorentz_factor_right = Kokkos::sqrt(squared_lorentz_factor_right);
                const auto magnetic_field_three_vector_dot_weighted_velocity_right = primitive_right_c_array[WeightedVelocityX1] * primitive_right_c_array[MagneticFieldX1] + primitive_right_c_array[WeightedVelocityX2] * primitive_right_c_array[MagneticFieldX2] + primitive_right_c_array[WeightedVelocityX3] * primitive_right_c_array[MagneticFieldX3];

                const auto alfven_velocity_center =
                    ComputeAlfvenVelocityCenter(adiabatic_index,
                                                primitive_left_c_array,
                                                primitive_right_c_array, X1DIR,
                                                is_gr);

                CalculateEnergyMomentumTensorInDir(adiabatic_index, primitive_left_c_array, X0DIR, directed_energy_momentum_tensor_left);
                conservative_left[DensityIndex] = primitive_left_c_array[DensityIndex] * lorentz_factor_left;
                conservative_left[EnergyIndex] = directed_energy_momentum_tensor_left[0] + conservative_left[DensityIndex];
                conservative_left[WeightedVelocityX1] = directed_energy_momentum_tensor_left[1];
                conservative_left[WeightedVelocityX2] = directed_energy_momentum_tensor_left[2];
                conservative_left[WeightedVelocityX3] = directed_energy_momentum_tensor_left[3];
                conservative_left[MagneticFieldX1] = primitive_left_c_array[MagneticFieldX1];
                conservative_left[MagneticFieldX2] = primitive_left_c_array[MagneticFieldX2];
                conservative_left[MagneticFieldX3] = primitive_left_c_array[MagneticFieldX3];

                CalculateEnergyMomentumTensorInDir(adiabatic_index, primitive_left_c_array, X1DIR, directed_energy_momentum_tensor_left);
                flux_left[DensityIndex] = primitive_left_c_array[DensityIndex] * primitive_left_c_array[WeightedVelocityX1];
                flux_left[EnergyIndex] = directed_energy_momentum_tensor_left[0] + flux_left[DensityIndex];
                flux_left[WeightedVelocityX1] = directed_energy_momentum_tensor_left[1];
                flux_left[WeightedVelocityX2] = directed_energy_momentum_tensor_left[2];
                flux_left[WeightedVelocityX3] = directed_energy_momentum_tensor_left[3];
                flux_left[MagneticFieldX1] = 0;
                flux_left[MagneticFieldX2] = ((primitive_left_c_array[MagneticFieldX2] + magnetic_field_three_vector_dot_weighted_velocity_left * primitive_left_c_array[WeightedVelocityX2]) * primitive_left_c_array[WeightedVelocityX1] - (primitive_left_c_array[MagneticFieldX1] + magnetic_field_three_vector_dot_weighted_velocity_left * primitive_left_c_array[WeightedVelocityX1]) * primitive_left_c_array[WeightedVelocityX2]) / lorentz_factor_left;
                flux_left[MagneticFieldX3] = ((primitive_left_c_array[MagneticFieldX3] + magnetic_field_three_vector_dot_weighted_velocity_left * primitive_left_c_array[WeightedVelocityX3]) * primitive_left_c_array[WeightedVelocityX1] - (primitive_left_c_array[MagneticFieldX1] + magnetic_field_three_vector_dot_weighted_velocity_left * primitive_left_c_array[WeightedVelocityX1]) * primitive_left_c_array[WeightedVelocityX3]) / lorentz_factor_left;

                CalculateEnergyMomentumTensorInDir(adiabatic_index, primitive_right_c_array, X0DIR, directed_energy_momentum_tensor_right);
                conservative_right[DensityIndex] = primitive_right_c_array[DensityIndex] * lorentz_factor_right;
                conservative_right[EnergyIndex] = directed_energy_momentum_tensor_right[0] + conservative_right[DensityIndex];
                conservative_right[WeightedVelocityX1] = directed_energy_momentum_tensor_right[1];
                conservative_right[WeightedVelocityX2] = directed_energy_momentum_tensor_right[2];
                conservative_right[WeightedVelocityX3] = directed_energy_momentum_tensor_right[3];
                conservative_right[MagneticFieldX1] = primitive_right_c_array[MagneticFieldX1];
                conservative_right[MagneticFieldX2] = primitive_right_c_array[MagneticFieldX2];
                conservative_right[MagneticFieldX3] = primitive_right_c_array[MagneticFieldX3];

                CalculateEnergyMomentumTensorInDir(adiabatic_index, primitive_right_c_array, X1DIR, directed_energy_momentum_tensor_right);
                flux_right[DensityIndex] = primitive_right_c_array[DensityIndex] * primitive_right_c_array[WeightedVelocityX1];
                flux_right[EnergyIndex] = directed_energy_momentum_tensor_right[0] + flux_right[DensityIndex];
                flux_right[WeightedVelocityX1] = directed_energy_momentum_tensor_right[1];
                flux_right[WeightedVelocityX2] = directed_energy_momentum_tensor_right[2];
                flux_right[WeightedVelocityX3] = directed_energy_momentum_tensor_right[3];
                flux_right[MagneticFieldX1] = 0;
                flux_right[MagneticFieldX2] = ((primitive_right_c_array[MagneticFieldX2] + magnetic_field_three_vector_dot_weighted_velocity_right * primitive_right_c_array[WeightedVelocityX2]) * primitive_right_c_array[WeightedVelocityX1] - (primitive_right_c_array[MagneticFieldX1] + magnetic_field_three_vector_dot_weighted_velocity_right * primitive_right_c_array[WeightedVelocityX1]) * primitive_right_c_array[WeightedVelocityX2]) / lorentz_factor_right;
                flux_right[MagneticFieldX3] = ((primitive_right_c_array[MagneticFieldX3] + magnetic_field_three_vector_dot_weighted_velocity_right * primitive_right_c_array[WeightedVelocityX3]) * primitive_right_c_array[WeightedVelocityX1] - (primitive_right_c_array[MagneticFieldX1] + magnetic_field_three_vector_dot_weighted_velocity_right * primitive_right_c_array[WeightedVelocityX1]) * primitive_right_c_array[WeightedVelocityX3]) / lorentz_factor_right;
                
                for (int index = 0; index < PrimitiveVariableNumber; ++index) {
                    conservative.flux(X1DIR, index, k, j, i) = is_gr
                        ? HLLFluxGRMHD(flux_left[index], flux_right[index], conservative_left[index], conservative_right[index], alfven_velocity_center)
                        : HLLFluxSRMHD(flux_left[index], flux_right[index], conservative_left[index], conservative_right[index], alfven_velocity_center);
                }
            });
        });

    if (meshblock_pointer->pmy_mesh->ndim >= 2) {
        meshblock_pointer->par_for_outer(
            PARTHENON_AUTO_LABEL, 2 * scratch_size_in_bytes_x2, scratch_level,
            bound_x3.s - offset_x3, bound_x3.e + offset_x3,
            bound_x1.s - offset_x1, bound_x1.e + offset_x1,
            KOKKOS_LAMBDA(team_mbr_t member, const int k, const int i) {
            ScratchPad2D<Real> primitive_left(member.team_scratch(scratch_level), PrimitiveVariableNumber, meshgrid_size_x2);
            ScratchPad2D<Real> primitive_right(member.team_scratch(scratch_level), PrimitiveVariableNumber, meshgrid_size_x2);

            par_for_inner(member, 0, PrimitiveVariableNumber - 1, bound_x2.s, bound_x2.e + 1, [&](const int n, const int j) {
                primitive_left(n, j) = primitive(n, k, j - 1, i) + 0.5 * InterpolateMC(primitive(n, k, j - 2, i), primitive(n, k, j - 1, i), primitive(n, k, j, i));
                primitive_right(n, j) = primitive(n, k, j, i) - 0.5 * InterpolateMC(primitive(n, k, j - 1, i), primitive(n, k, j, i), primitive(n, k, j + 1, i));
            });
            
            member.team_barrier();

            par_for_inner(member, bound_x2.s, bound_x2.e + 1, [&](const int j) {
                Real directed_energy_momentum_tensor_left[4];
                Real directed_energy_momentum_tensor_right[4];
                Real conservative_left[PrimitiveVariableNumber];
                Real conservative_right[PrimitiveVariableNumber];
                Real flux_left[PrimitiveVariableNumber];
                Real flux_right[PrimitiveVariableNumber];
                const Real primitive_left_c_array[PrimitiveVariableNumber] = {
                    primitive_left(DensityIndex, j),
                    primitive_left(EnergyIndex, j),
                    primitive_left(WeightedVelocityX1, j),
                    primitive_left(WeightedVelocityX2, j),
                    primitive_left(WeightedVelocityX3, j),
                    primitive_left(MagneticFieldX1, j),
                    primitive_left(MagneticFieldX2, j),
                    primitive_left(MagneticFieldX3, j),
                };
                const Real primitive_right_c_array[PrimitiveVariableNumber] = {
                    primitive_right(DensityIndex, j),
                    primitive_right(EnergyIndex, j),
                    primitive_right(WeightedVelocityX1, j),
                    primitive_right(WeightedVelocityX2, j),
                    primitive_right(WeightedVelocityX3, j),
                    primitive_right(MagneticFieldX1, j),
                    primitive_right(MagneticFieldX2, j),
                    primitive_right(MagneticFieldX3, j),
                };

                const auto squared_weighted_velocity_left = Kokkos::pow(primitive_left_c_array[WeightedVelocityX1], 2) + Kokkos::pow(primitive_left_c_array[WeightedVelocityX2], 2) + Kokkos::pow(primitive_left_c_array[WeightedVelocityX3], 2);
                const auto squared_lorentz_factor_left = 1 + squared_weighted_velocity_left;
                const auto lorentz_factor_left = Kokkos::sqrt(squared_lorentz_factor_left);
                const auto magnetic_field_three_vector_dot_weighted_velocity_left = primitive_left_c_array[WeightedVelocityX1] * primitive_left_c_array[MagneticFieldX1] + primitive_left_c_array[WeightedVelocityX2] * primitive_left_c_array[MagneticFieldX2] + primitive_left_c_array[WeightedVelocityX3] * primitive_left_c_array[MagneticFieldX3];

                const auto squared_weighted_velocity_right = Kokkos::pow(primitive_right_c_array[WeightedVelocityX1], 2) + Kokkos::pow(primitive_right_c_array[WeightedVelocityX2], 2) + Kokkos::pow(primitive_right_c_array[WeightedVelocityX3], 2);
                const auto squared_lorentz_factor_right = 1 + squared_weighted_velocity_right;
                const auto lorentz_factor_right = Kokkos::sqrt(squared_lorentz_factor_right);
                const auto magnetic_field_three_vector_dot_weighted_velocity_right = primitive_right_c_array[WeightedVelocityX1] * primitive_right_c_array[MagneticFieldX1] + primitive_right_c_array[WeightedVelocityX2] * primitive_right_c_array[MagneticFieldX2] + primitive_right_c_array[WeightedVelocityX3] * primitive_right_c_array[MagneticFieldX3];

                const auto alfven_velocity_center =
                    ComputeAlfvenVelocityCenter(adiabatic_index,
                                                primitive_left_c_array,
                                                primitive_right_c_array, X2DIR,
                                                is_gr);

                CalculateEnergyMomentumTensorInDir(adiabatic_index, primitive_left_c_array, X0DIR, directed_energy_momentum_tensor_left);
                conservative_left[DensityIndex] = primitive_left_c_array[DensityIndex] * lorentz_factor_left;
                conservative_left[EnergyIndex] = directed_energy_momentum_tensor_left[0] + conservative_left[DensityIndex];
                conservative_left[WeightedVelocityX1] = directed_energy_momentum_tensor_left[1];
                conservative_left[WeightedVelocityX2] = directed_energy_momentum_tensor_left[2];
                conservative_left[WeightedVelocityX3] = directed_energy_momentum_tensor_left[3];
                conservative_left[MagneticFieldX1] = primitive_left_c_array[MagneticFieldX1];
                conservative_left[MagneticFieldX2] = primitive_left_c_array[MagneticFieldX2];
                conservative_left[MagneticFieldX3] = primitive_left_c_array[MagneticFieldX3];

                CalculateEnergyMomentumTensorInDir(adiabatic_index, primitive_left_c_array, X2DIR, directed_energy_momentum_tensor_left);
                flux_left[DensityIndex] = primitive_left_c_array[DensityIndex] * primitive_left_c_array[WeightedVelocityX2];
                flux_left[EnergyIndex] = directed_energy_momentum_tensor_left[0] + flux_left[DensityIndex];
                flux_left[WeightedVelocityX1] = directed_energy_momentum_tensor_left[1];
                flux_left[WeightedVelocityX2] = directed_energy_momentum_tensor_left[2];
                flux_left[WeightedVelocityX3] = directed_energy_momentum_tensor_left[3];
                flux_left[MagneticFieldX1] = ((primitive_left_c_array[MagneticFieldX1] + magnetic_field_three_vector_dot_weighted_velocity_left * primitive_left_c_array[WeightedVelocityX1]) * primitive_left_c_array[WeightedVelocityX2] - (primitive_left_c_array[MagneticFieldX2] + magnetic_field_three_vector_dot_weighted_velocity_left * primitive_left_c_array[WeightedVelocityX2]) * primitive_left_c_array[WeightedVelocityX1]) / lorentz_factor_left;
                flux_left[MagneticFieldX2] = 0;
                flux_left[MagneticFieldX3] = ((primitive_left_c_array[MagneticFieldX3] + magnetic_field_three_vector_dot_weighted_velocity_left * primitive_left_c_array[WeightedVelocityX3]) * primitive_left_c_array[WeightedVelocityX2] - (primitive_left_c_array[MagneticFieldX2] + magnetic_field_three_vector_dot_weighted_velocity_left * primitive_left_c_array[WeightedVelocityX2]) * primitive_left_c_array[WeightedVelocityX3]) / lorentz_factor_left;

                CalculateEnergyMomentumTensorInDir(adiabatic_index, primitive_right_c_array, X0DIR, directed_energy_momentum_tensor_right);
                conservative_right[DensityIndex] = primitive_right_c_array[DensityIndex] * lorentz_factor_right;
                conservative_right[EnergyIndex] = directed_energy_momentum_tensor_right[0] + conservative_right[DensityIndex];
                conservative_right[WeightedVelocityX1] = directed_energy_momentum_tensor_right[1];
                conservative_right[WeightedVelocityX2] = directed_energy_momentum_tensor_right[2];
                conservative_right[WeightedVelocityX3] = directed_energy_momentum_tensor_right[3];
                conservative_right[MagneticFieldX1] = primitive_right_c_array[MagneticFieldX1];
                conservative_right[MagneticFieldX2] = primitive_right_c_array[MagneticFieldX2];
                conservative_right[MagneticFieldX3] = primitive_right_c_array[MagneticFieldX3];

                CalculateEnergyMomentumTensorInDir(adiabatic_index, primitive_right_c_array, X2DIR, directed_energy_momentum_tensor_right);
                flux_right[DensityIndex] = primitive_right_c_array[DensityIndex] * primitive_right_c_array[WeightedVelocityX2];
                flux_right[EnergyIndex] = directed_energy_momentum_tensor_right[0] + flux_right[DensityIndex];
                flux_right[WeightedVelocityX1] = directed_energy_momentum_tensor_right[1];
                flux_right[WeightedVelocityX2] = directed_energy_momentum_tensor_right[2];
                flux_right[WeightedVelocityX3] = directed_energy_momentum_tensor_right[3];
                flux_right[MagneticFieldX1] = ((primitive_right_c_array[MagneticFieldX1] + magnetic_field_three_vector_dot_weighted_velocity_right * primitive_right_c_array[WeightedVelocityX1]) * primitive_right_c_array[WeightedVelocityX2] - (primitive_right_c_array[MagneticFieldX2] + magnetic_field_three_vector_dot_weighted_velocity_right * primitive_right_c_array[WeightedVelocityX2]) * primitive_right_c_array[WeightedVelocityX1]) / lorentz_factor_right;
                flux_right[MagneticFieldX2] = 0;
                flux_right[MagneticFieldX3] = ((primitive_right_c_array[MagneticFieldX3] + magnetic_field_three_vector_dot_weighted_velocity_right * primitive_right_c_array[WeightedVelocityX3]) * primitive_right_c_array[WeightedVelocityX2] - (primitive_right_c_array[MagneticFieldX2] + magnetic_field_three_vector_dot_weighted_velocity_right * primitive_right_c_array[WeightedVelocityX2]) * primitive_right_c_array[WeightedVelocityX3]) / lorentz_factor_right;

                for (int index = 0; index < PrimitiveVariableNumber; ++index) {
                    conservative.flux(X2DIR, index, k, j, i) = is_gr
                        ? HLLFluxGRMHD(flux_left[index], flux_right[index], conservative_left[index], conservative_right[index], alfven_velocity_center)
                        : HLLFluxSRMHD(flux_left[index], flux_right[index], conservative_left[index], conservative_right[index], alfven_velocity_center);
                }
            });
        });
    }
    
    if (meshblock_pointer->pmy_mesh->ndim == 3) {
        meshblock_pointer->par_for_outer(
            PARTHENON_AUTO_LABEL, 2 * scratch_size_in_bytes_x3, scratch_level,
            bound_x2.s, bound_x2.e, bound_x1.s, bound_x1.e,
            KOKKOS_LAMBDA(team_mbr_t member, const int j, const int i) {
            ScratchPad2D<Real> primitive_left(member.team_scratch(scratch_level), PrimitiveVariableNumber, meshgrid_size_x3);
            ScratchPad2D<Real> primitive_right(member.team_scratch(scratch_level), PrimitiveVariableNumber, meshgrid_size_x3);

            par_for_inner(member, 0, PrimitiveVariableNumber - 1, bound_x3.s, bound_x3.e + 1, [&](const int n, const int k) {
                primitive_left(n, k) = primitive(n, k - 1, j, i) + 0.5 * InterpolateMC(primitive(n, k - 2, j, i), primitive(n, k - 1, j, i), primitive(n, k, j, i));
                primitive_right(n, k) = primitive(n, k, j, i) - 0.5 * InterpolateMC(primitive(n, k - 1, j, i), primitive(n, k, j, i), primitive(n, k + 1, j, i));
            });
            
            member.team_barrier();

            par_for_inner(member, bound_x3.s, bound_x3.e + 1, [&](const int k) {
                Real directed_energy_momentum_tensor_left[4];
                Real directed_energy_momentum_tensor_right[4];
                Real conservative_left[PrimitiveVariableNumber];
                Real conservative_right[PrimitiveVariableNumber];
                Real flux_left[PrimitiveVariableNumber];
                Real flux_right[PrimitiveVariableNumber];
                const Real primitive_left_c_array[PrimitiveVariableNumber] = {
                    primitive_left(DensityIndex, k),
                    primitive_left(EnergyIndex, k),
                    primitive_left(WeightedVelocityX1, k),
                    primitive_left(WeightedVelocityX2, k),
                    primitive_left(WeightedVelocityX3, k),
                    primitive_left(MagneticFieldX1, k),
                    primitive_left(MagneticFieldX2, k),
                    primitive_left(MagneticFieldX3, k),
                };
                const Real primitive_right_c_array[PrimitiveVariableNumber] = {
                    primitive_right(DensityIndex, k),
                    primitive_right(EnergyIndex, k),
                    primitive_right(WeightedVelocityX1, k),
                    primitive_right(WeightedVelocityX2, k),
                    primitive_right(WeightedVelocityX3, k),
                    primitive_right(MagneticFieldX1, k),
                    primitive_right(MagneticFieldX2, k),
                    primitive_right(MagneticFieldX3, k),
                };

                const auto squared_weighted_velocity_left = Kokkos::pow(primitive_left_c_array[WeightedVelocityX1], 2) + Kokkos::pow(primitive_left_c_array[WeightedVelocityX2], 2) + Kokkos::pow(primitive_left_c_array[WeightedVelocityX3], 2);
                const auto squared_lorentz_factor_left = 1 + squared_weighted_velocity_left;
                const auto lorentz_factor_left = Kokkos::sqrt(squared_lorentz_factor_left);
                const auto magnetic_field_three_vector_dot_weighted_velocity_left = primitive_left_c_array[WeightedVelocityX1] * primitive_left_c_array[MagneticFieldX1] + primitive_left_c_array[WeightedVelocityX2] * primitive_left_c_array[MagneticFieldX2] + primitive_left_c_array[WeightedVelocityX3] * primitive_left_c_array[MagneticFieldX3];

                const auto squared_weighted_velocity_right = Kokkos::pow(primitive_right_c_array[WeightedVelocityX1], 2) + Kokkos::pow(primitive_right_c_array[WeightedVelocityX2], 2) + Kokkos::pow(primitive_right_c_array[WeightedVelocityX3], 2);
                const auto squared_lorentz_factor_right = 1 + squared_weighted_velocity_right;
                const auto lorentz_factor_right = Kokkos::sqrt(squared_lorentz_factor_right);
                const auto magnetic_field_three_vector_dot_weighted_velocity_right = primitive_right_c_array[WeightedVelocityX1] * primitive_right_c_array[MagneticFieldX1] + primitive_right_c_array[WeightedVelocityX2] * primitive_right_c_array[MagneticFieldX2] + primitive_right_c_array[WeightedVelocityX3] * primitive_right_c_array[MagneticFieldX3];

                const auto alfven_velocity_center =
                    ComputeAlfvenVelocityCenter(adiabatic_index,
                                                primitive_left_c_array,
                                                primitive_right_c_array, X3DIR,
                                                is_gr);

                CalculateEnergyMomentumTensorInDir(adiabatic_index, primitive_left_c_array, X0DIR, directed_energy_momentum_tensor_left);
                conservative_left[DensityIndex] = primitive_left_c_array[DensityIndex] * lorentz_factor_left;
                conservative_left[EnergyIndex] = directed_energy_momentum_tensor_left[0] + conservative_left[DensityIndex];
                conservative_left[WeightedVelocityX1] = directed_energy_momentum_tensor_left[1];
                conservative_left[WeightedVelocityX2] = directed_energy_momentum_tensor_left[2];
                conservative_left[WeightedVelocityX3] = directed_energy_momentum_tensor_left[3];
                conservative_left[MagneticFieldX1] = primitive_left_c_array[MagneticFieldX1];
                conservative_left[MagneticFieldX2] = primitive_left_c_array[MagneticFieldX2];
                conservative_left[MagneticFieldX3] = primitive_left_c_array[MagneticFieldX3];

                CalculateEnergyMomentumTensorInDir(adiabatic_index, primitive_left_c_array, X3DIR, directed_energy_momentum_tensor_left);
                flux_left[DensityIndex] = primitive_left_c_array[DensityIndex] * primitive_left_c_array[WeightedVelocityX3];
                flux_left[EnergyIndex] = directed_energy_momentum_tensor_left[0] + flux_left[DensityIndex];
                flux_left[WeightedVelocityX1] = directed_energy_momentum_tensor_left[1];
                flux_left[WeightedVelocityX2] = directed_energy_momentum_tensor_left[2];
                flux_left[WeightedVelocityX3] = directed_energy_momentum_tensor_left[3];
                flux_left[MagneticFieldX1] = ((primitive_left_c_array[MagneticFieldX1] + magnetic_field_three_vector_dot_weighted_velocity_left * primitive_left_c_array[WeightedVelocityX1]) * primitive_left_c_array[WeightedVelocityX3] - (primitive_left_c_array[MagneticFieldX3] + magnetic_field_three_vector_dot_weighted_velocity_left * primitive_left_c_array[WeightedVelocityX3]) * primitive_left_c_array[WeightedVelocityX1]) / lorentz_factor_left;
                flux_left[MagneticFieldX2] = ((primitive_left_c_array[MagneticFieldX2] + magnetic_field_three_vector_dot_weighted_velocity_left * primitive_left_c_array[WeightedVelocityX2]) * primitive_left_c_array[WeightedVelocityX3] - (primitive_left_c_array[MagneticFieldX3] + magnetic_field_three_vector_dot_weighted_velocity_left * primitive_left_c_array[WeightedVelocityX3]) * primitive_left_c_array[WeightedVelocityX2]) / lorentz_factor_left;
                flux_left[MagneticFieldX3] = 0;

                CalculateEnergyMomentumTensorInDir(adiabatic_index, primitive_right_c_array, X0DIR, directed_energy_momentum_tensor_right);
                conservative_right[DensityIndex] = primitive_right_c_array[DensityIndex] * lorentz_factor_right;
                conservative_right[EnergyIndex] = directed_energy_momentum_tensor_right[0] + conservative_right[DensityIndex];
                conservative_right[WeightedVelocityX1] = directed_energy_momentum_tensor_right[1];
                conservative_right[WeightedVelocityX2] = directed_energy_momentum_tensor_right[2];
                conservative_right[WeightedVelocityX3] = directed_energy_momentum_tensor_right[3];
                conservative_right[MagneticFieldX1] = primitive_right_c_array[MagneticFieldX1];
                conservative_right[MagneticFieldX2] = primitive_right_c_array[MagneticFieldX2];
                conservative_right[MagneticFieldX3] = primitive_right_c_array[MagneticFieldX3];

                CalculateEnergyMomentumTensorInDir(adiabatic_index, primitive_right_c_array, X3DIR, directed_energy_momentum_tensor_right);
                flux_right[DensityIndex] = primitive_right_c_array[DensityIndex] * primitive_right_c_array[WeightedVelocityX3];
                flux_right[EnergyIndex] = directed_energy_momentum_tensor_right[0] + flux_right[DensityIndex];
                flux_right[WeightedVelocityX1] = directed_energy_momentum_tensor_right[1];
                flux_right[WeightedVelocityX2] = directed_energy_momentum_tensor_right[2];
                flux_right[WeightedVelocityX3] = directed_energy_momentum_tensor_right[3];
                flux_right[MagneticFieldX1] = ((primitive_right_c_array[MagneticFieldX1] + magnetic_field_three_vector_dot_weighted_velocity_right * primitive_right_c_array[WeightedVelocityX1]) * primitive_right_c_array[WeightedVelocityX3] - (primitive_right_c_array[MagneticFieldX3] + magnetic_field_three_vector_dot_weighted_velocity_right * primitive_right_c_array[WeightedVelocityX3]) * primitive_right_c_array[WeightedVelocityX1]) / lorentz_factor_right;
                flux_right[MagneticFieldX2] = ((primitive_right_c_array[MagneticFieldX2] + magnetic_field_three_vector_dot_weighted_velocity_right * primitive_right_c_array[WeightedVelocityX2]) * primitive_right_c_array[WeightedVelocityX3] - (primitive_right_c_array[MagneticFieldX3] + magnetic_field_three_vector_dot_weighted_velocity_right * primitive_right_c_array[WeightedVelocityX3]) * primitive_right_c_array[WeightedVelocityX2]) / lorentz_factor_right;
                flux_right[MagneticFieldX3] = 0;

                for (int index = 0; index < PrimitiveVariableNumber; ++index) {
                    conservative.flux(X3DIR, index, k, j, i) = is_gr
                        ? HLLFluxGRMHD(flux_left[index], flux_right[index], conservative_left[index], conservative_right[index], alfven_velocity_center)
                        : HLLFluxSRMHD(flux_left[index], flux_right[index], conservative_left[index], conservative_right[index], alfven_velocity_center);
                }
            });
        });
    }

    return TaskStatus::complete;
}
