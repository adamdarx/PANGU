#include <exception>
#include <iostream>
#include "app/application_input.h"
#include "app/command_line.h"
#include "app/package_registry.h"
#include "app/parameter_compat.h"
#include "app/problem_registry.h"
#include "driver/driver.h"
#include "estimator/estimator.h"
#include "geometry_assembly.h"
#include "pangu.h"
#include "pangu_config.h"
#include "plugin/registry.h"

int main(int argc, char** argv) {
  using parthenon::ParthenonManager;
  using parthenon::ParthenonStatus;

  auto command_line = pangu::app::ParseCommandLine(argc, argv);
  if (command_line.exit_code >= 0)
    return command_line.exit_code;
  if (command_line.version) {
    std::cout << pangu::kFrameworkTitle << ' ' << PANGU_VERSION << "\n"
              << "PANGU " << PANGU_GIT_COMMIT << " (" << PANGU_GIT_DIRTY << ")\n"
              << "Parthenon " << PANGU_PARTHENON_COMMIT << '\n'
              << "Physics " << PANGU_PHYSICS << '\n'
              << "Metric " << pangu::geometry::ConfiguredMetricName() << '\n'
              << "Geometry mode " << pangu::geometry::ConfiguredModeName() << '\n'
              << "Estimator " << pangu::estimator::estimator_name << '\n'
              << "Plugins: use --list-plugins\n";
    return 0;
  }
  if (command_line.list_plugins) {
    pangu::plugin::ListCompiled(std::cout);
    return 0;
  }
  argc = static_cast<int>(command_line.parthenon_arguments.size());
  for (int index = 0; index < argc; ++index)
    argv[index] = command_line.parthenon_arguments[index].data();
  ParthenonManager manager;

  try {
    pangu::app::ConfigureApplicationDefaults(manager.app_input.get());
    const auto init_status = manager.ParthenonInitEnv(argc, argv);
    if (init_status != ParthenonStatus::ok) {
      manager.ParthenonFinalize();
      return init_status == ParthenonStatus::complete ? 0 : 1;
    }

    pangu::app::NormalizeParameters(manager.pinput.get());
    pangu::geometry::ValidateConfiguredGeometryInput(manager.pinput.get());
    pangu::app::ValidateParameters(manager.pinput.get());
    const auto problem = manager.pinput->GetString("parthenon/job", "problem_id");
    pangu::app::ConfigureProblem(problem, manager.app_input.get());

    if (command_line.check_input || manager.pinput->GetBoolean("pangu", "check_input_only")) {
      // Some package parameters own Kokkos Views (for example the tabulated
      // TOV profile).  Destroy the validation-only package collection before
      // Parthenon finalizes Kokkos.
      {
        [[maybe_unused]] auto validated_packages = pangu::app::ProcessPackages(manager.pinput);
      }
      if (parthenon::Globals::my_rank == 0) {
        pangu::app::PrintConfiguration(manager.pinput.get(), std::cout);
        std::cout << "Input validation: PASS\n";
      }
      manager.ParthenonFinalize();
      return 0;
    }

    manager.ParthenonInitPackagesAndMesh();
    int return_code = 0;
    {
      pangu::driver::Driver driver(manager.pinput.get(), manager.app_input.get(),
                                   manager.pmesh.get());
      return_code = driver.Execute() == parthenon::DriverStatus::complete ? 0 : 1;
    }
    manager.ParthenonFinalize();
    return return_code;
  } catch (const std::exception& error) {
    std::cerr << "PANGU fatal error: " << error.what() << '\n';
    if (Kokkos::is_initialized()) {
      manager.ParthenonFinalize();
#ifdef MPI_PARALLEL
    } else {
      int mpi_initialized = 0;
      int mpi_finalized = 0;
      MPI_Initialized(&mpi_initialized);
      if (mpi_initialized != 0)
        MPI_Finalized(&mpi_finalized);
      if (mpi_initialized != 0 && mpi_finalized == 0)
        MPI_Finalize();
#endif
    }
    return 1;
  }
}
