#ifndef PANGU_RADIATION_GEODESIC_GRID_H_
#define PANGU_RADIATION_GEODESIC_GRID_H_

#include <array>
#include <vector>

#include <parthenon/parthenon.hpp>

namespace pangu::radiation {

struct AngularQuadrature {
  std::vector<std::array<parthenon::Real, 3>> directions;
  std::vector<parthenon::Real> solid_angles;
};

// Construct the same dual icosahedral angular quadrature used by AthenaK.
// Level zero retains the eight-octant testing quadrature; positive levels
// contain 10 * level^2 + 2 directions.
AngularQuadrature BuildGeodesicQuadrature(int level, bool rotate);

} // namespace pangu::radiation

#endif
