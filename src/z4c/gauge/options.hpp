#ifndef PANGU_Z4C_GAUGE_OPTIONS_HPP_
#define PANGU_Z4C_GAUGE_OPTIONS_HPP_

#include <parthenon/parthenon.hpp>

namespace pangu::nr::gauge {

enum class Lapse : int { legacy, frozen, harmonic, one_plus_log, shifted_gauge_wave };
enum class Shift : int { legacy, frozen, gamma_driver, harmonic, shifted_gauge_wave };

struct Options {
  Lapse lapse = Lapse::legacy;
  Shift shift = Shift::legacy;
  parthenon::Real lapse_oplog = 2.0;
  parthenon::Real lapse_harmonic_factor = 1.0;
  parthenon::Real lapse_harmonic = 0.0;
  parthenon::Real lapse_advect = 1.0;
  bool slow_start_lapse = false;
  parthenon::Real slow_start_amplitude = 0.6;
  parthenon::Real slow_start_time = 20.0;
  parthenon::Real slow_start_index = 1.0;
  parthenon::Real shift_gamma = 1.0;
  parthenon::Real shift_alpha2_gamma = 0.0;
  parthenon::Real shift_harmonic = 0.0;
  parthenon::Real shift_advect = 1.0;
  parthenon::Real shift_eta = 2.0;
};

} // namespace pangu::nr::gauge

#endif
