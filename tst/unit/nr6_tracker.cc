#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include "z4c/tracking/interpolation.h"
#include "z4c/tracking/position_update.h"

int main() {
  namespace tracking = pangu::nr::tracking;
  int failures = 0;
  const auto check = [&](bool condition, const char *message) {
    if (!condition) {
      std::cerr << message << '\n';
      ++failures;
    }
  };
  check(tracking::ParseMode("ode") == tracking::Mode::ode,
        "ODE selection failed");
  check(tracking::ParseMode("walk") == tracking::Mode::walk,
        "walk selection failed");
  for (const auto *name : {"", "ODE", "frozen", "invalid"}) {
    bool rejected = false;
    try {
      tracking::ParseMode(name);
    } catch (const std::invalid_argument &error) {
      rejected =
          std::string(error.what()).find("tracker_mode") != std::string::npos;
    }
    check(rejected, "invalid tracker mode was accepted");
  }
  // Match the actual AthenaK scalar update, including a shortened last dt.
  std::array<double, 3> actual{0.0, 0.25, -0.75};
  auto reference = actual;
  const std::array<double, 3> shift{0.5, -0.125, 0.25};
  for (double dt : {0.025, 0.025, 0.01, 6.7501559897209518e-14}) {
    for (int axis = 0; axis < 3; ++axis) {
      const double velocity = -shift[axis];
      reference[axis] += dt * velocity;
      actual[axis] = tracking::AdvanceOdePosition(actual[axis], velocity, dt);
      check(actual[axis] == reference[axis], "ODE update differs from AthenaK");
    }
  }
  check(tracking::AdvanceOdePosition(0.0, 0.0, 0.025) == 0.0,
        "stationary puncture snapped away from the origin");
  check(tracking::AdvanceOdePosition(0.125, 0.5, 0.0) == 0.125,
        "zero timestep changed the position");
  check(tracking::AdvanceOdePosition(0.0, -0.5, 0.025) == -0.0125,
        "puncture velocity sign is wrong");
  const double tiny = tracking::AdvanceOdePosition(0.0, 1.0e-25, 0.025);
  check(tiny > 0.0 && tiny < 1.0e-26,
        "roundoff-scale motion was snapped or suppressed");
  // Literal independent transcriptions of the reference host expressions.
  for (double lower : {-32., -2., 0., 2., 1.e6}) {
    const double upper = lower + 2.;
    for (int index = -2; index < 18; ++index) {
      const double x = (static_cast<double>(index) + 0.5) / 16.;
      const double expected = (x * upper - x * lower) -
                              (0.5 * upper - 0.5 * lower) +
                              (0.5 * lower + 0.5 * upper);
      check(tracking::InterpolationCellCenter(index, 16, lower, upper) ==
                expected,
            "interpolation coordinate differs from AthenaK CellCenterX");
    }
  }
  const std::array<double, 12> weights{-.0625, .5625, .5625, -.0625,
                                       -.0625, .5625, .5625, -.0625,
                                       -.0625, .5625, .5625, -.0625};
  const auto sample = [](int index) {
    return (index % 2 ? -1. : 1.) / (index + 1.);
  };
  double expected = 0.;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      for (int k = 0; k < 4; ++k) {
        const double w = weights[i] * weights[4 + j] * weights[8 + k];
        const double scalar = sample((i * 4 + j) * 4 + k);
        expected += w * scalar;
      }
  check(tracking::InterpolateHost<double>(4, 4, weights, sample) == expected,
        "host tensor interpolation differs from AthenaK accumulation");
  if (failures)
    return 1;
  std::cout << "NR tracker mode/ODE update PASS (logic only, not a speed "
               "benchmark)\n";
  return 0;
}
