#include "app/application_input.h"

#include "app/package_registry.h"
#include "geometry_assembly.h"
#include "mhd/mhd_package.h"

namespace pangu::app {
namespace {

void PreStep(parthenon::Mesh* mesh, parthenon::ParameterInput* pin, parthenon::SimTime& time) {
  mhd::UpdateAMRSchedule(mesh, time);
}

} // namespace

void ConfigureApplicationDefaults(parthenon::ApplicationInput* app_input) {
  app_input->ProcessPackages = ProcessPackages;
  app_input->InitMeshBlockUserData = geometry::InitializeConfiguredGeometry;
  app_input->PreStepMeshUserWorkInLoop = PreStep;
  app_input->RegisterDefaultReflectingBoundaryConditions();
}

} // namespace pangu::app
