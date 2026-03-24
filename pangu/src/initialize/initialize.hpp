#pragma once
#include <memory>
#include <string>
#include <vector>

#include <parthenon/package.hpp>
#include "timestep.hpp"
#include "../mesh/amr_criteria.hpp"

namespace CORE {
std::shared_ptr<parthenon::StateDescriptor> Initialize(parthenon::ParameterInput *pin) {
    const auto PackageCORE = std::make_shared<parthenon::StateDescriptor>("CORE");

    const auto CFLNumber = pin->GetOrAddReal("CORE", "CFLNumber", 0.8);
    const auto AdiabaticIndex = pin->GetOrAddReal("CORE", "AdiabaticIndex", 5. / 3.);
    const auto QFactorFloor = pin->GetOrAddReal("CORE", "QFactorFloor", 0.3);
    const auto QFactorCeiling = pin->GetOrAddReal("CORE", "QFactorCeiling", 0.03);

    PackageCORE->AddParam<>("CFLNumber", CFLNumber);
    PackageCORE->AddParam<>("AdiabaticIndex", AdiabaticIndex);
    PackageCORE->AddParam<>("QFactorFloor", QFactorFloor);
    PackageCORE->AddParam<>("QFactorCeiling", QFactorCeiling);

    parthenon::Metadata m;
    m = parthenon::Metadata({parthenon::Metadata::Cell, parthenon::Metadata::FillGhost});
    PackageCORE->AddField(std::string("Density"), m);
    m = parthenon::Metadata({parthenon::Metadata::Cell, parthenon::Metadata::FillGhost});
    PackageCORE->AddField(std::string("Energy"), m);
    m = parthenon::Metadata({parthenon::Metadata::Cell, parthenon::Metadata::FillGhost});
    PackageCORE->AddField(std::string("QFactor"), m);
    m = parthenon::Metadata({parthenon::Metadata::Cell, parthenon::Metadata::FillGhost, parthenon::Metadata::Vector}, std::vector<int>({3}));
    PackageCORE->AddField(std::string("WeightedVelocity"), m);
    m = parthenon::Metadata({parthenon::Metadata::Cell, parthenon::Metadata::FillGhost}, std::vector<int>({3}));
    PackageCORE->AddField(std::string("Alfven"), m);
    m = parthenon::Metadata({parthenon::Metadata::Cell, parthenon::Metadata::FillGhost, parthenon::Metadata::Vector}, std::vector<int>({3}));
    PackageCORE->AddField(std::string("MagneticField"), m);
    m = parthenon::Metadata({parthenon::Metadata::Cell, parthenon::Metadata::FillGhost}, std::vector<int>({3}));
    PackageCORE->AddField(std::string("ElectricField"), m);
    m = parthenon::Metadata({parthenon::Metadata::Cell, parthenon::Metadata::WithFluxes, parthenon::Metadata::Independent, parthenon::Metadata::FillGhost}, std::vector<int>({8}));
    PackageCORE->AddField(std::string("Conservative"), m);
    m = parthenon::Metadata({parthenon::Metadata::Cell, parthenon::Metadata::FillGhost});
    PackageCORE->AddField(std::string("Flag"), m);

    PackageCORE->CheckRefinementBlock = CheckRefinement;
    PackageCORE->EstimateTimestepBlock = EstimateTimestepBlock;
    return PackageCORE;
}
} // namespace CORE

namespace METRIC {
std::shared_ptr<parthenon::StateDescriptor> Initialize(parthenon::ParameterInput *pin) {
    const auto PackageMETRIC = std::make_shared<parthenon::StateDescriptor>("METRIC");

	const auto MetricName = pin->GetOrAddString("METRIC", "name", "Minkowski");
    PackageMETRIC->AddParam<>("MetricName", MetricName);

	parthenon::Metadata m;
	m = parthenon::Metadata({parthenon::Metadata::Cell, parthenon::Metadata::FillGhost},
							std::vector<int>({4, 4}));
	PackageMETRIC->AddField(std::string("CovariantMetric"), m);

	m = parthenon::Metadata({parthenon::Metadata::Cell, parthenon::Metadata::FillGhost},
							std::vector<int>({4, 4}));
	PackageMETRIC->AddField(std::string("ContravariantMetric"), m);

	m = parthenon::Metadata({parthenon::Metadata::Cell, parthenon::Metadata::FillGhost});
	PackageMETRIC->AddField(std::string("MetricDeterminant"), m);

	m = parthenon::Metadata({parthenon::Metadata::Cell, parthenon::Metadata::FillGhost},
							std::vector<int>({4, 4, 4}));
	PackageMETRIC->AddField(std::string("CovariantMetricDerivative"), m);

	return PackageMETRIC;
}
} // namespace METRIC
