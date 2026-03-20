#pragma once
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <parthenon/package.hpp>

#include "../initialize/mnemonic.hpp"
#include "../physics/alfven_velocity.hpp"
#include "../reconstruct/InterpolaterMC.hpp"

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
        CalculateAlfvenVelocityGRMHD(adiabatic_index, primitive_right_c_array,
                                     direction, maximum_alfven_velocity_right,
                                     minimum_alfven_velocity_right);
    } else {
        CalculateAlfvenVelocitySRMHD(adiabatic_index, primitive_left_c_array, direction,
                                     maximum_alfven_velocity_left,
                                     minimum_alfven_velocity_left);
        CalculateAlfvenVelocitySRMHD(adiabatic_index, primitive_right_c_array,
                                     direction, maximum_alfven_velocity_right,
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

parthenon::Real EstimateTimestepBlock(
    parthenon::MeshBlockData<parthenon::Real> *resource) {
    using namespace parthenon;

    const auto meshblock_pointer = resource->GetBlockPointer();
    const auto package = meshblock_pointer->packages.Get("PANGU");
    const auto cfl_number = package->Param<Real>("CFLNumber");
    const auto adiabatic_index = package->Param<Real>("AdiabaticIndex");
    const auto mode = package->Param<std::string>("Mode");
    const int is_gr = (mode == "GR");

    const auto bound_x1 =
        meshblock_pointer->cellbounds.GetBoundsI(IndexDomain::interior);
    const auto bound_x2 =
        meshblock_pointer->cellbounds.GetBoundsJ(IndexDomain::interior);
    const auto bound_x3 =
        meshblock_pointer->cellbounds.GetBoundsK(IndexDomain::interior);

    const auto &coords = meshblock_pointer->coords;

    Real minimum_of_timestep_x1 = std::numeric_limits<Real>::max();
    Real minimum_of_timestep_x2 = std::numeric_limits<Real>::max();
    Real minimum_of_timestep_x3 = std::numeric_limits<Real>::max();

    PackIndexMap primitive_index_map;
    const std::vector<std::string> primitive_tags = {
        "Density", "Energy", "WeightedVelocity", "MagneticField"};
    const auto primitive = resource->PackVariables(primitive_tags, primitive_index_map);

    PackIndexMap alfven_velocity_index_map;
    const std::vector<std::string> alfven_tags = {"Alfven"};
    auto alfven_velocity =
        resource->PackVariables(alfven_tags, alfven_velocity_index_map);

    const int scratch_level = 1;
    const int meshgrid_size_x1 =
        meshblock_pointer->cellbounds.ncellsi(IndexDomain::entire);
    const int meshgrid_size_x2 =
        meshblock_pointer->cellbounds.ncellsj(IndexDomain::entire);
    const int meshgrid_size_x3 =
        meshblock_pointer->cellbounds.ncellsk(IndexDomain::entire);

    const size_t scratch_size_in_bytes_x1 =
        ScratchPad2D<Real>::shmem_size(PrimitiveVariableNumber, meshgrid_size_x1);
    const size_t scratch_size_in_bytes_x2 =
        ScratchPad2D<Real>::shmem_size(PrimitiveVariableNumber, meshgrid_size_x2);
    const size_t scratch_size_in_bytes_x3 =
        ScratchPad2D<Real>::shmem_size(PrimitiveVariableNumber, meshgrid_size_x3);

    const int offset_x1 = (meshgrid_size_x1 > 1) ? 1 : 0;
    const int offset_x2 = (meshgrid_size_x2 > 1) ? 1 : 0;
    const int offset_x3 = (meshgrid_size_x3 > 1) ? 1 : 0;

    if (offset_x1) {
        meshblock_pointer->par_for_outer(
            PARTHENON_AUTO_LABEL, 2 * scratch_size_in_bytes_x1, scratch_level,
            bound_x3.s - offset_x3, bound_x3.e + offset_x3,
            bound_x2.s - offset_x2, bound_x2.e + offset_x2,
            KOKKOS_LAMBDA(team_mbr_t member, const int k, const int j) {
                ScratchPad2D<Real> primitive_left(
                    member.team_scratch(scratch_level), PrimitiveVariableNumber,
                    meshgrid_size_x1);
                ScratchPad2D<Real> primitive_right(
                    member.team_scratch(scratch_level), PrimitiveVariableNumber,
                    meshgrid_size_x1);

                par_for_inner(
                    member, 0, PrimitiveVariableNumber - 1, bound_x1.s,
                    bound_x1.e + 1, [&](const int n, const int i) {
                        primitive_left(n, i) =
                            primitive(n, k, j, i - 1) +
                            0.5 * InterpolateMC(primitive(n, k, j, i - 2),
                                                primitive(n, k, j, i - 1),
                                                primitive(n, k, j, i));
                        primitive_right(n, i) =
                            primitive(n, k, j, i) -
                            0.5 * InterpolateMC(primitive(n, k, j, i - 1),
                                                primitive(n, k, j, i),
                                                primitive(n, k, j, i + 1));
                    });

                member.team_barrier();

                par_for_inner(member, bound_x1.s, bound_x1.e + 1,
                              [&](const int i) {
                                  const Real primitive_left_c_array
                                      [PrimitiveVariableNumber] = {
                                          primitive_left(DensityIndex, i),
                                          primitive_left(EnergyIndex, i),
                                          primitive_left(WeightedVelocityX1, i),
                                          primitive_left(WeightedVelocityX2, i),
                                          primitive_left(WeightedVelocityX3, i),
                                          primitive_left(MagneticFieldX1, i),
                                          primitive_left(MagneticFieldX2, i),
                                          primitive_left(MagneticFieldX3, i),
                                      };
                                  const Real primitive_right_c_array
                                      [PrimitiveVariableNumber] = {
                                          primitive_right(DensityIndex, i),
                                          primitive_right(EnergyIndex, i),
                                          primitive_right(WeightedVelocityX1, i),
                                          primitive_right(WeightedVelocityX2, i),
                                          primitive_right(WeightedVelocityX3, i),
                                          primitive_right(MagneticFieldX1, i),
                                          primitive_right(MagneticFieldX2, i),
                                          primitive_right(MagneticFieldX3, i),
                                      };

                                  alfven_velocity(Vector3D::X1, k, j, i) =
                                      ComputeAlfvenVelocityCenter(
                                          adiabatic_index, primitive_left_c_array,
                                          primitive_right_c_array, X1DIR, is_gr);
                              });
            });
    }

    if (offset_x2) {
        meshblock_pointer->par_for_outer(
            PARTHENON_AUTO_LABEL, 2 * scratch_size_in_bytes_x2, scratch_level,
            bound_x3.s - offset_x3, bound_x3.e + offset_x3,
            bound_x1.s - offset_x1, bound_x1.e + offset_x1,
            KOKKOS_LAMBDA(team_mbr_t member, const int k, const int i) {
                ScratchPad2D<Real> primitive_left(
                    member.team_scratch(scratch_level), PrimitiveVariableNumber,
                    meshgrid_size_x2);
                ScratchPad2D<Real> primitive_right(
                    member.team_scratch(scratch_level), PrimitiveVariableNumber,
                    meshgrid_size_x2);

                par_for_inner(
                    member, 0, PrimitiveVariableNumber - 1, bound_x2.s,
                    bound_x2.e + 1, [&](const int n, const int j) {
                        primitive_left(n, j) =
                            primitive(n, k, j - 1, i) +
                            0.5 * InterpolateMC(primitive(n, k, j - 2, i),
                                                primitive(n, k, j - 1, i),
                                                primitive(n, k, j, i));
                        primitive_right(n, j) =
                            primitive(n, k, j, i) -
                            0.5 * InterpolateMC(primitive(n, k, j - 1, i),
                                                primitive(n, k, j, i),
                                                primitive(n, k, j + 1, i));
                    });

                member.team_barrier();

                par_for_inner(member, bound_x2.s, bound_x2.e + 1,
                              [&](const int j) {
                                  const Real primitive_left_c_array
                                      [PrimitiveVariableNumber] = {
                                          primitive_left(DensityIndex, j),
                                          primitive_left(EnergyIndex, j),
                                          primitive_left(WeightedVelocityX1, j),
                                          primitive_left(WeightedVelocityX2, j),
                                          primitive_left(WeightedVelocityX3, j),
                                          primitive_left(MagneticFieldX1, j),
                                          primitive_left(MagneticFieldX2, j),
                                          primitive_left(MagneticFieldX3, j),
                                      };
                                  const Real primitive_right_c_array
                                      [PrimitiveVariableNumber] = {
                                          primitive_right(DensityIndex, j),
                                          primitive_right(EnergyIndex, j),
                                          primitive_right(WeightedVelocityX1, j),
                                          primitive_right(WeightedVelocityX2, j),
                                          primitive_right(WeightedVelocityX3, j),
                                          primitive_right(MagneticFieldX1, j),
                                          primitive_right(MagneticFieldX2, j),
                                          primitive_right(MagneticFieldX3, j),
                                      };

                                  alfven_velocity(Vector3D::X2, k, j, i) =
                                      ComputeAlfvenVelocityCenter(
                                          adiabatic_index, primitive_left_c_array,
                                          primitive_right_c_array, X2DIR, is_gr);
                              });
            });
    }

    if (offset_x3) {
        meshblock_pointer->par_for_outer(
            PARTHENON_AUTO_LABEL, 2 * scratch_size_in_bytes_x3, scratch_level,
            bound_x2.s - offset_x2, bound_x2.e + offset_x2,
            bound_x1.s - offset_x1, bound_x1.e + offset_x1,
            KOKKOS_LAMBDA(team_mbr_t member, const int j, const int i) {
                ScratchPad2D<Real> primitive_left(
                    member.team_scratch(scratch_level), PrimitiveVariableNumber,
                    meshgrid_size_x3);
                ScratchPad2D<Real> primitive_right(
                    member.team_scratch(scratch_level), PrimitiveVariableNumber,
                    meshgrid_size_x3);

                par_for_inner(
                    member, 0, PrimitiveVariableNumber - 1, bound_x3.s,
                    bound_x3.e + 1, [&](const int n, const int k) {
                        primitive_left(n, k) =
                            primitive(n, k - 1, j, i) +
                            0.5 * InterpolateMC(primitive(n, k - 2, j, i),
                                                primitive(n, k - 1, j, i),
                                                primitive(n, k, j, i));
                        primitive_right(n, k) =
                            primitive(n, k, j, i) -
                            0.5 * InterpolateMC(primitive(n, k - 1, j, i),
                                                primitive(n, k, j, i),
                                                primitive(n, k + 1, j, i));
                    });

                member.team_barrier();

                par_for_inner(member, bound_x3.s, bound_x3.e + 1,
                              [&](const int k) {
                                  const Real primitive_left_c_array
                                      [PrimitiveVariableNumber] = {
                                          primitive_left(DensityIndex, k),
                                          primitive_left(EnergyIndex, k),
                                          primitive_left(WeightedVelocityX1, k),
                                          primitive_left(WeightedVelocityX2, k),
                                          primitive_left(WeightedVelocityX3, k),
                                          primitive_left(MagneticFieldX1, k),
                                          primitive_left(MagneticFieldX2, k),
                                          primitive_left(MagneticFieldX3, k),
                                      };
                                  const Real primitive_right_c_array
                                      [PrimitiveVariableNumber] = {
                                          primitive_right(DensityIndex, k),
                                          primitive_right(EnergyIndex, k),
                                          primitive_right(WeightedVelocityX1, k),
                                          primitive_right(WeightedVelocityX2, k),
                                          primitive_right(WeightedVelocityX3, k),
                                          primitive_right(MagneticFieldX1, k),
                                          primitive_right(MagneticFieldX2, k),
                                          primitive_right(MagneticFieldX3, k),
                                      };

                                  alfven_velocity(Vector3D::X3, k, j, i) =
                                      ComputeAlfvenVelocityCenter(
                                          adiabatic_index, primitive_left_c_array,
                                          primitive_right_c_array, X3DIR, is_gr);
                              });
            });
    }

    auto reduce_directional_dt =
        [&](const int component, const Real dx, Real &minimum_of_timestep) {
            meshblock_pointer->par_reduce(
                PARTHENON_AUTO_LABEL, bound_x3.s, bound_x3.e, bound_x2.s,
                bound_x2.e, bound_x1.s, bound_x1.e,
                KOKKOS_LAMBDA(const int k, const int j, const int i, Real &dt_local) {
                    const Real local_dt = dx / alfven_velocity(component, k, j, i);
                    if (dt_local > local_dt) {
                        dt_local = local_dt;
                    }
                },
                Kokkos::Min<Real>(minimum_of_timestep));
        };

    if (offset_x1) {
        reduce_directional_dt(Vector3D::X1, coords.Dx<X1DIR>(), minimum_of_timestep_x1);
    }
    if (offset_x2) {
        reduce_directional_dt(Vector3D::X2, coords.Dx<X2DIR>(), minimum_of_timestep_x2);
    }
    if (offset_x3) {
        reduce_directional_dt(Vector3D::X3, coords.Dx<X3DIR>(), minimum_of_timestep_x3);
    }

    Real inverse_timestep_sum = 0.0;
    if (offset_x1) {
        inverse_timestep_sum += 1.0 / minimum_of_timestep_x1;
    }
    if (offset_x2) {
        inverse_timestep_sum += 1.0 / minimum_of_timestep_x2;
    }
    if (offset_x3) {
        inverse_timestep_sum += 1.0 / minimum_of_timestep_x3;
    }

    const Real minimum_of_timestep = 1.0 / inverse_timestep_sum;
    return cfl_number * minimum_of_timestep;
}
