#include "Simulator.hpp"

#include "../constrainted_transport/constrainted_transport.hpp"
#include "../fixer/fix_primitive.hpp"
#include "../fixer/fix_recovery.hpp"
#include "../flux/conservative.hpp"
#include "../flux/flux.hpp"
#include "../flux/source.hpp"
#include "../mesh/q_factor.hpp"
#include "../metric/compute_metric.hpp"
#include "../recovery/Recovery.hpp"

#include "amr_criteria/refinement_package.hpp"
#include "bvals/comms/bvals_in_one.hpp"
#include "interface/metadata.hpp"
#include "interface/update.hpp"
#include "mesh/meshblock_pack.hpp"
#include "parthenon/driver.hpp"
#include "prolong_restrict/prolong_restrict.hpp"

TaskCollection Simulator::MakeTaskCollectionGRMHD(BlockList_t &blocks, const int stage) {
    using namespace parthenon::Update;
    TaskCollection tc;
    TaskID none(0);

    const Real beta = integrator->beta[stage % 2];
    const Real dt = integrator->dt;
    const auto &stage_name = integrator->stage_name;

    auto partitions = pmesh->GetDefaultBlockPartitions();
    const int num_partitions = partitions.size();
    TaskRegion &recv_region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; ++i) {
        auto &tl = recv_region[i];
        auto &mbase = pmesh->mesh_data.Add("base", partitions[i]);
        auto &mc0 = pmesh->mesh_data.Add(stage_name[stage - 1], mbase);
        auto &mc1 = pmesh->mesh_data.Add(stage_name[stage], mbase);

        const auto any = parthenon::BoundaryType::any;
        tl.AddTask(none, parthenon::StartReceiveBoundBufs<any>, mc1);
        tl.AddTask(none, parthenon::StartReceiveFluxCorrections, mc0);
    }

    const auto num_async = blocks.size();
    TaskRegion &flux_region = tc.AddRegion(num_async);
    assert(blocks.size() == flux_region.size());
    for (int i = 0; i < num_async; ++i) {
        auto &pmb = blocks[i];
        auto &tl = flux_region[i];

        auto &sc0 = pmb->meshblock_data.Get(stage_name[stage - 1]);
        auto compute_metric = tl.AddTask(none, ComputeMetric, sc0);
        auto calc_cons = tl.AddTask(compute_metric, CalculateConservative, sc0);
        auto calc_flux = tl.AddTask(calc_cons, CalculateFluxes, sc0);
        tl.AddTask(calc_flux, ConstraintedTransport, sc0);
    }

    TaskRegion &update_region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; ++i) {
        auto &tl = update_region[i];
        auto &mbase = pmesh->mesh_data.Add("base", partitions[i]);
        auto &mc_init = pmesh->mesh_data.Add(stage_name[0], mbase);
        auto &mc0 = pmesh->mesh_data.Add(stage_name[stage - 1], mbase);
        auto &mc1 = pmesh->mesh_data.Add(stage_name[stage], mbase);
        auto &mdudt = pmesh->mesh_data.Add("dUdt", mbase);

        auto set_flx =
                parthenon::AddFluxCorrectionTasks(none, tl, mc0, pmesh->multilevel);
        auto flux_div =
            tl.AddTask(set_flx, FluxDivergence<MeshData<Real>>, mc0.get(),
                   mdudt.get());
        auto update = tl.AddTask(
            flux_div, UpdateIndependentData<MeshData<Real>>, mc_init.get(),
            mdudt.get(), beta * dt, mc1.get());

        parthenon::AddBoundaryExchangeTasks(update, tl, mc1, pmesh->multilevel);
    }

    // Following AthenaK ordering: source terms are applied after RK update.
    TaskRegion &source_region = tc.AddRegion(num_async);
    assert(blocks.size() == source_region.size());
    for (int i = 0; i < num_async; ++i) {
        auto &pmb = blocks[i];
        auto &tl = source_region[i];

        auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);
        auto compute_metric = tl.AddTask(none, ComputeMetric, sc1);
        auto source_task = tl.AddTask(compute_metric, ApplySourceTerms, sc1);
        tl.AddTask(source_task, TransformConservativeToPrimitive, sc1);
    }

    TaskRegion &fix_region = tc.AddRegion(num_async);
    assert(blocks.size() == fix_region.size());
    for (int i = 0; i < num_async; ++i) {
        auto &pmb = blocks[i];
        auto &tl = fix_region[i];

        auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);
        auto fix_rec = tl.AddTask(none, FixRecovery, sc1);
        tl.AddTask(fix_rec, FixPrimitive, sc1);
    }

    TaskRegion &q_region = tc.AddRegion(num_async);
    assert(blocks.size() == q_region.size());
    for (int i = 0; i < num_async; ++i) {
        auto &pmb = blocks[i];
        auto &tl = q_region[i];

        auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);
        tl.AddTask(none, CalculateQFactor, sc1);
    }

    TaskRegion &final_region = tc.AddRegion(num_async);
    assert(blocks.size() == final_region.size());
    for (int i = 0; i < num_async; ++i) {
        auto &pmb = blocks[i];
        auto &tl = final_region[i];
        auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);

        auto set_bc = tl.AddTask(none, parthenon::ApplyBoundaryConditions, sc1);
        auto fill_derived = tl.AddTask(
                set_bc, parthenon::Update::FillDerived<MeshBlockData<Real>>, sc1.get());
        if (stage == integrator->nstages) {
            tl.AddTask(fill_derived, EstimateTimestep<MeshBlockData<Real>>, sc1.get());
            if (pmesh->adaptive) {
                tl.AddTask(fill_derived, parthenon::Refinement::Tag<MeshBlockData<Real>>,
                           sc1.get());
            }
        }
    }

    return tc;
}
