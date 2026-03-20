#pragma once

#include <memory>
#include <string>
#include <vector>

#include <basic_types.hpp>
#include <parthenon/package.hpp>

#include "../initialize/mnemonic.hpp"
#include "../metric/kerr_metric.hpp"
#include "../metric/schwarzschild_metric.hpp"
#include "../physics/stress_tensor.hpp"

KOKKOS_INLINE_FUNCTION
void ComputeMinkowskiMetricForSource(parthenon::Real gcov[4][4],
                                     parthenon::Real gcon[4][4]) {
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
}

KOKKOS_INLINE_FUNCTION
void ComputeMetricDerivativesNumerical(const parthenon::Real x1,
                                       const parthenon::Real x2,
                                       const parthenon::Real x3,
                                       const parthenon::Real spin,
                                       const int metric_type,
                                       parthenon::Real dg_dx1[4][4],
                                       parthenon::Real dg_dx2[4][4],
                                       parthenon::Real dg_dx3[4][4]) {
    const parthenon::Real h = 1e-4;
    parthenon::Real gp[4][4], gm[4][4], dummy[4][4];

    if (metric_type == 0) {
        for (int mu = 0; mu < 4; ++mu) {
            for (int nu = 0; nu < 4; ++nu) {
                dg_dx1[mu][nu] = 0.0;
                dg_dx2[mu][nu] = 0.0;
                dg_dx3[mu][nu] = 0.0;
            }
        }
        return;
    }

    if (metric_type == 1) {
        ComputeSchwarzschildMetric(x1 + h, x2, x3, gp, dummy);
        ComputeSchwarzschildMetric(x1 - h, x2, x3, gm, dummy);
    } else {
        ComputeKerrMetric(x1 + h, x2, x3, spin, gp, dummy);
        ComputeKerrMetric(x1 - h, x2, x3, spin, gm, dummy);
    }
    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            dg_dx1[mu][nu] = (gp[mu][nu] - gm[mu][nu]) / (2.0 * h);
        }
    }

    if (metric_type == 1) {
        ComputeSchwarzschildMetric(x1, x2 + h, x3, gp, dummy);
        ComputeSchwarzschildMetric(x1, x2 - h, x3, gm, dummy);
    } else {
        ComputeKerrMetric(x1, x2 + h, x3, spin, gp, dummy);
        ComputeKerrMetric(x1, x2 - h, x3, spin, gm, dummy);
    }
    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            dg_dx2[mu][nu] = (gp[mu][nu] - gm[mu][nu]) / (2.0 * h);
        }
    }

    if (metric_type == 1) {
        ComputeSchwarzschildMetric(x1, x2, x3 + h, gp, dummy);
        ComputeSchwarzschildMetric(x1, x2, x3 - h, gm, dummy);
    } else {
        ComputeKerrMetric(x1, x2, x3 + h, spin, gp, dummy);
        ComputeKerrMetric(x1, x2, x3 - h, spin, gm, dummy);
    }
    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            dg_dx3[mu][nu] = (gp[mu][nu] - gm[mu][nu]) / (2.0 * h);
        }
    }
}

parthenon::TaskStatus ApplySourceTerms(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource) {
    using namespace parthenon;
    PARTHENON_INSTRUMENT

    const auto meshblock_pointer = resource->GetBlockPointer();
    const auto package = meshblock_pointer->packages.Get("PANGU");
    const auto mode = package->Param<std::string>("Mode");
    if (mode != "GR") {
        return TaskStatus::complete;
    }

    const auto adiabatic_index = package->Param<Real>("AdiabaticIndex");
    const auto metric_name = package->Param<std::string>("MetricName");
    const auto metric_spin = package->Param<Real>("MetricSpin");
    const int metric_type =
        (metric_name == "Minkowski") ? 0 : ((metric_name == "Schwarzschild") ? 1 : 2);
    const auto dt = meshblock_pointer->pmy_mesh->dt;

    const auto bound_x1 =
        meshblock_pointer->cellbounds.GetBoundsI(IndexDomain::interior);
    const auto bound_x2 =
        meshblock_pointer->cellbounds.GetBoundsJ(IndexDomain::interior);
    const auto bound_x3 =
        meshblock_pointer->cellbounds.GetBoundsK(IndexDomain::interior);
    const auto &coords = meshblock_pointer->coords;

    PackIndexMap primitive_index_map;
    auto primitive = resource->PackVariables(
        {"Density", "Energy", "WeightedVelocity", "MagneticField"},
        primitive_index_map);
    PackIndexMap conservative_index_map;
    auto conservative =
        resource->PackVariables({"Conservative"}, conservative_index_map);

    meshblock_pointer->par_for(
        PARTHENON_AUTO_LABEL, bound_x3.s, bound_x3.e, bound_x2.s, bound_x2.e,
        bound_x1.s, bound_x1.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
            const Real x1 = coords.Xc<X1DIR>(i);
            const Real x2 = coords.Xc<X2DIR>(j);
            const Real x3 = coords.Xc<X3DIR>(k);

            Real gcov[4][4], gcon[4][4];
            if (metric_type == 0) {
                ComputeMinkowskiMetricForSource(gcov, gcon);
            } else if (metric_type == 1) {
                ComputeSchwarzschildMetric(x1, x2, x3, gcov, gcon);
            } else {
                ComputeKerrMetric(x1, x2, x3, metric_spin, gcov, gcon);
            }

            Real primitive_c_array[PrimitiveVariableNumber] = {
                primitive(DensityIndex, k, j, i), primitive(EnergyIndex, k, j, i),
                primitive(WeightedVelocityX1, k, j, i),
                primitive(WeightedVelocityX2, k, j, i),
                primitive(WeightedVelocityX3, k, j, i),
                primitive(MagneticFieldX1, k, j, i),
                primitive(MagneticFieldX2, k, j, i),
                primitive(MagneticFieldX3, k, j, i)};

            Real tt[4][4];
            CalculateEnergyMomentumTensor(adiabatic_index, primitive_c_array, tt);

            Real dg_dx1[4][4], dg_dx2[4][4], dg_dx3[4][4];
            ComputeMetricDerivativesNumerical(x1, x2, x3, metric_spin, metric_type,
                                              dg_dx1, dg_dx2, dg_dx3);

            Real s1 = 0.0, s2 = 0.0, s3 = 0.0;
            for (int mu = 0; mu < 4; ++mu) {
                for (int nu = mu; nu < 4; ++nu) {
                    const Real factor = (mu == nu) ? 0.5 : 1.0;
                    s1 += factor * dg_dx1[mu][nu] * tt[mu][nu];
                    s2 += factor * dg_dx2[mu][nu] * tt[mu][nu];
                    s3 += factor * dg_dx3[mu][nu] * tt[mu][nu];
                }
            }

            conservative(WeightedVelocityX1, k, j, i) += dt * s1;
            conservative(WeightedVelocityX2, k, j, i) += dt * s2;
            conservative(WeightedVelocityX3, k, j, i) += dt * s3;
        });

    return TaskStatus::complete;
}
