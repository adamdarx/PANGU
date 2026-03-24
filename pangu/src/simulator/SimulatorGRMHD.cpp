#include "Simulator.hpp"
#include "../initialize/initialize.hpp"
#include "../flux/conservative.hpp"
#include "../flux/flux_lax.hpp"
#include "../flux/source.hpp"
#include "../constrainted_transport/constrainted_transport.hpp"
#include "../mesh/q_factor.hpp"
#include "../recovery/Recovery.hpp"
#include "../recovery/Transform.hpp"
#include "../fixer/fix_primitive.hpp"
#include "../fixer/fix_recovery.hpp"

#include <memory>
#include <string>
#include <vector>

#include "amr_criteria/refinement_package.hpp"
#include "bvals/comms/bvals_in_one.hpp"
#include "interface/metadata.hpp"
#include "interface/update.hpp"
#include "mesh/meshblock_pack.hpp"
#include "parthenon/driver.hpp"
#include "prolong_restrict/prolong_restrict.hpp"

using namespace parthenon::driver::prelude;

Simulator::Simulator(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
    : MultiStageDriver(pin, app_in, pm) {

    pin->CheckRequired("parthenon/mesh", "ix1_bc");
    pin->CheckRequired("parthenon/mesh", "ox1_bc");
    pin->CheckRequired("parthenon/mesh", "ix2_bc");
    pin->CheckRequired("parthenon/mesh", "ox2_bc");

    pin->CheckDesired("CORE", "CFLNumber");
    pin->CheckDesired("CORE", "AdiabaticIndex");
    pin->CheckDesired("CORE", "QFactorFloor");
    pin->CheckDesired("CORE", "QFactorCeiling");
}

TaskCollection Simulator::MakeTaskCollection(BlockList_t &blocks, const int stage) {
    using namespace parthenon::Update;
    TaskCollection tc;
    TaskID none(0);

    const Real beta = integrator->beta[stage % 2];
    const Real dt = integrator->dt;
    const auto &stage_name = integrator->stage_name;

    auto partitions = pmesh->GetDefaultBlockPartitions();
    int num_partitions = partitions.size();
    TaskRegion &single_tasklist_per_pack_region2 = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
        auto &tl = single_tasklist_per_pack_region2[i];
        auto &mbase = pmesh->mesh_data.Add("base", partitions[i]);
        auto &mc0 = pmesh->mesh_data.Add(stage_name[stage - 1], mbase);
        auto &mc1 = pmesh->mesh_data.Add(stage_name[stage], mbase);
        auto &mdudt = pmesh->mesh_data.Add("dUdt", mbase);

        const auto any = parthenon::BoundaryType::any;

        tl.AddTask(none, parthenon::StartReceiveBoundBufs<any>, mc1);
        tl.AddTask(none, parthenon::StartReceiveFluxCorrections, mc0);
    }

    auto num_task_lists_executed_independently = blocks.size();
    TaskRegion &async_region1 = tc.AddRegion(num_task_lists_executed_independently);

    assert(blocks.size() == async_region1.size());
    for (int i = 0; i < blocks.size(); i++) {
        auto &pmb = blocks[i];
        auto &tl = async_region1[i];

        auto &sc0 = pmb->meshblock_data.Get(stage_name[stage - 1]);
        auto &dudt = pmb->meshblock_data.Get("dUdt");
    
        auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);
        auto calc_cons = tl.AddTask(none, CalculateConservativeGRMHD, sc0);
        auto calc_flux = tl.AddTask(calc_cons, CalculateFluxesGRMHD, sc0);
        auto ct_task = tl.AddTask(calc_flux, ConstraintedTransport, sc0);
    }

    TaskRegion &single_tasklist_per_pack_region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
        auto &tl = single_tasklist_per_pack_region[i];
        auto &mbase = pmesh->mesh_data.Add("base", partitions[i]);
        auto &mc_init = pmesh->mesh_data.Add(stage_name[0], mbase);
        auto &mc0 = pmesh->mesh_data.Add(stage_name[stage - 1], mbase);
        auto &mc1 = pmesh->mesh_data.Add(stage_name[stage], mbase);
        auto &mdudt = pmesh->mesh_data.Add("dUdt", mbase);

        auto set_flx = parthenon::AddFluxCorrectionTasks(none, tl, mc0, pmesh->multilevel);
        auto flux_div = tl.AddTask(set_flx, FluxDivergence<MeshData<Real>>, mc0.get(), mdudt.get());
        auto update = tl.AddTask(flux_div, UpdateIndependentData<MeshData<Real>>, mc_init.get(),
                    mdudt.get(), beta * dt, mc1.get());

        parthenon::AddBoundaryExchangeTasks(update, tl, mc1, pmesh->multilevel);
    }

    TaskRegion &async_region2 = tc.AddRegion(num_task_lists_executed_independently);
    assert(blocks.size() == async_region2.size());
    for (int i = 0; i < blocks.size(); i++) {
        auto &pmb = blocks[i];
        auto &tl = async_region2[i];

        auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);
        
        auto source_task = tl.AddTask(none, AddSourceGRMHD, sc1);
        auto pre_transform = tl.AddTask(source_task, Transform::TransformConservativeGRMHDToSRMHD, sc1);
        auto recover_task = tl.AddTask(pre_transform, TransformConservativeToPrimitive, sc1);
        auto post_transform = tl.AddTask(recover_task, Transform::TransformPrimitiveSRMHDToGRMHD, sc1);
    }

    TaskRegion &async_region3 = tc.AddRegion(num_task_lists_executed_independently);
    assert(blocks.size() == async_region3.size());
    for (int i = 0; i < blocks.size(); i++) {
        auto &pmb = blocks[i];
        auto &tl = async_region3[i];

        auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);
        
        auto fix_rec = tl.AddTask(none, FixRecovery, sc1);
        auto fix_prim = tl.AddTask(fix_rec, FixPrimitive, sc1);
    }

    TaskRegion &async_region4 = tc.AddRegion(num_task_lists_executed_independently);
    assert(blocks.size() == async_region4.size());
    for (int i = 0; i < blocks.size(); i++) {
        auto &pmb = blocks[i];
        auto &tl = async_region4[i];

        auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);
        
        tl.AddTask(none, CalculateQFactor, sc1);
    }

    TaskRegion &async_region5 = tc.AddRegion(num_task_lists_executed_independently);
    assert(blocks.size() == async_region5.size());
    for (int i = 0; i < blocks.size(); i++) {
        auto &pmb = blocks[i];
        auto &tl = async_region5[i];
        auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);

        auto set_bc = tl.AddTask(none, parthenon::ApplyBoundaryConditions, sc1);
        auto fill_derived = tl.AddTask(
            set_bc, parthenon::Update::FillDerived<MeshBlockData<Real>>, sc1.get());
        if (stage == integrator->nstages) {
        auto new_dt =
            tl.AddTask(fill_derived, EstimateTimestep<MeshBlockData<Real>>, sc1.get());
        if (pmesh->adaptive) {
            auto tag_refine = tl.AddTask(
                fill_derived, parthenon::Refinement::Tag<MeshBlockData<Real>>, sc1.get());
            }
        }
    }
    return tc;
}

parthenon::Packages_t ProcessPackages(std::unique_ptr<parthenon::ParameterInput> &pin) {
    Packages_t packages;
    auto PackageCORE = CORE::Initialize(pin.get());
    auto PackageMETRIC = METRIC::Initialize(pin.get());
    packages.Add(PackageCORE);
    packages.Add(PackageMETRIC);

    return packages;
}
