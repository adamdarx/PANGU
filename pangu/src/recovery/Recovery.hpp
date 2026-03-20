#pragma once

#include <memory>
#include <string>
#include <vector>

#include <parthenon/package.hpp>

#include "../initialize/mnemonic.hpp"
#include "../metric/kerr_metric.hpp"
#include "../metric/schwarzschild_metric.hpp"

#include "Scheme1D.hpp"
#include "Scheme1Dvsq.hpp"
#include "Scheme2D.hpp"
#include "transform.hpp"

parthenon::TaskStatus TransformConservativeToPrimitive(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource) {
    PARTHENON_INSTRUMENT

    const auto meshblock_pointer = resource->GetBlockPointer();
    const auto package = meshblock_pointer->packages.Get("PANGU");
    const auto adiabatic_index = package->Param<parthenon::Real>("AdiabaticIndex");
    const auto mode = package->Param<std::string>("Mode");
    const auto metric_name = package->Param<std::string>("MetricName");
    const auto metric_spin = package->Param<parthenon::Real>("MetricSpin");
    const int is_gr = (mode == "GR");
    const int metric_type =
        (metric_name == "Minkowski") ? 0 : ((metric_name == "Schwarzschild") ? 1 : 2);

    const auto bound_x1 =
        meshblock_pointer->cellbounds.GetBoundsI(parthenon::IndexDomain::interior);
    const auto bound_x2 =
        meshblock_pointer->cellbounds.GetBoundsJ(parthenon::IndexDomain::interior);
    const auto bound_x3 =
        meshblock_pointer->cellbounds.GetBoundsK(parthenon::IndexDomain::interior);
    const auto &coords = meshblock_pointer->coords;

    PackIndexMap primitive_index_map;
    const std::vector<std::string> primitive_tags = {
        "Density", "Energy", "WeightedVelocity", "MagneticField"};
    auto primitive = resource->PackVariables(primitive_tags, primitive_index_map);

    PackIndexMap conservative_index_map;
    const std::vector<std::string> conservative_tags = {"Conservative"};
    const auto conservative =
        resource->PackVariables(conservative_tags, conservative_index_map);
    auto &flag = resource->Get("Flag").data;

    meshblock_pointer->par_for(
        PARTHENON_AUTO_LABEL, bound_x3.s, bound_x3.e, bound_x2.s, bound_x2.e,
        bound_x1.s, bound_x1.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
            parthenon::Real conservative_c_array[PrimitiveVariableNumber];
            parthenon::Real primitive_c_array[PrimitiveVariableNumber];
            for (int index = 0; index < PrimitiveVariableNumber; ++index) {
                conservative_c_array[index] = conservative(index, k, j, i);
                primitive_c_array[index] = primitive(index, k, j, i);
            }

            if (is_gr) {
                parthenon::Real gcov[4][4], gcon[4][4];
                if (metric_type == 0) {
                    for (int mu = 0; mu < 4; ++mu) {
                        for (int nu = 0; nu < 4; ++nu) {
                            gcov[mu][nu] = 0.0;
                            gcon[mu][nu] = 0.0;
                        }
                    }
                    gcov[0][0] = -1.0;
                    gcov[1][1] = 1.0;
                    gcov[2][2] = 1.0;
                    gcov[3][3] = 1.0;
                    gcon[0][0] = -1.0;
                    gcon[1][1] = 1.0;
                    gcon[2][2] = 1.0;
                    gcon[3][3] = 1.0;
                } else if (metric_type == 1) {
                    ComputeSchwarzschildMetric(coords.Xc<X1DIR>(i), coords.Xc<X2DIR>(j),
                                                coords.Xc<X3DIR>(k), gcov, gcon);
                } else {
                    ComputeKerrMetric(coords.Xc<X1DIR>(i), coords.Xc<X2DIR>(j),
                                        coords.Xc<X3DIR>(k), metric_spin, gcov, gcon);
                }

                parthenon::Real conservative_sr[PrimitiveVariableNumber];
                TransformConservativeToSRMHD(conservative_c_array, gcov, gcon,
                                                conservative_sr);
                for (int index = 0; index < PrimitiveVariableNumber; ++index) {
                    conservative_c_array[index] = conservative_sr[index];
                }
            }

            conservative_c_array[EnergyIndex] -= conservative_c_array[DensityIndex];

            flag(k, j, i) = 0;
            if (Scheme2D::invert(conservative_c_array, primitive_c_array,
                                    adiabatic_index) == 0 ||
                Scheme1D::invert(conservative_c_array, primitive_c_array,
                                    adiabatic_index) == 0 ||
                Scheme1Dvsq::invert(conservative_c_array, primitive_c_array,
                                        adiabatic_index) == 0) {
                for (int index = 0; index < PrimitiveVariableNumber; ++index) {
                    primitive(index, k, j, i) = primitive_c_array[index];
                }
                flag(k, j, i) = 1;
            }
        });
    return parthenon::TaskStatus::complete;
}
