#pragma once
#include <memory>
#include <string>
#include <vector>

#include <parthenon/package.hpp>
#include "timestep.hpp"
#include "../mesh/amr_criteria.hpp"

std::shared_ptr<parthenon::StateDescriptor> Initialize(parthenon::ParameterInput *pin) {
    auto Package = std::make_shared<parthenon::StateDescriptor>("PANGU");

    auto CFLNumber = pin->GetOrAddReal("PANGU", "CFLNumber", 0.8);
    auto AdiabaticIndex = pin->GetOrAddReal("PANGU", "AdiabaticIndex", 5. / 3.);
    auto QFactorFloor = pin->GetOrAddReal("PANGU", "QFactorFloor", 0.3);
    auto QFactorCeiling = pin->GetOrAddReal("PANGU", "QFactorCeiling", 0.03);

    Package->AddParam<>("CFLNumber", CFLNumber);
    Package->AddParam<>("AdiabaticIndex", AdiabaticIndex);
    Package->AddParam<>("QFactorFloor", QFactorFloor);
    Package->AddParam<>("QFactorCeiling", QFactorCeiling);

    parthenon::Metadata m;
    m = parthenon::Metadata({parthenon::Metadata::Cell, parthenon::Metadata::FillGhost});
    Package->AddField(std::string("Density"), m);
    m = parthenon::Metadata({parthenon::Metadata::Cell, parthenon::Metadata::FillGhost});
    Package->AddField(std::string("Energy"), m);
    m = parthenon::Metadata({parthenon::Metadata::Cell, parthenon::Metadata::FillGhost});
    Package->AddField(std::string("QFactor"), m);
    m = parthenon::Metadata({parthenon::Metadata::Cell, parthenon::Metadata::FillGhost, parthenon::Metadata::Vector}, std::vector<int>({3}));
    Package->AddField(std::string("WeightedVelocity"), m);
    m = parthenon::Metadata({parthenon::Metadata::Cell, parthenon::Metadata::FillGhost}, std::vector<int>({3}));
    Package->AddField(std::string("Alfven"), m);
    m = parthenon::Metadata({parthenon::Metadata::Cell, parthenon::Metadata::FillGhost, parthenon::Metadata::Vector}, std::vector<int>({3}));
    Package->AddField(std::string("MagneticField"), m);
    m = parthenon::Metadata({parthenon::Metadata::Cell, parthenon::Metadata::FillGhost}, std::vector<int>({3}));
    Package->AddField(std::string("ElectricField"), m);
    m = parthenon::Metadata({parthenon::Metadata::Cell, parthenon::Metadata::WithFluxes, parthenon::Metadata::Independent, parthenon::Metadata::FillGhost}, std::vector<int>({8}));
    Package->AddField(std::string("Conservative"), m);
    m = parthenon::Metadata({parthenon::Metadata::Cell, parthenon::Metadata::FillGhost});
    Package->AddField(std::string("Flag"), m);

    Package->CheckRefinementBlock = CheckRefinement;
    Package->EstimateTimestepBlock = EstimateTimestepBlock;
    return Package;
}
