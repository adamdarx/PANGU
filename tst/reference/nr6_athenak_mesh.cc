// Mesh topology correctness replay, NOT a CPU solver or a timing benchmark.
// Both actual tree implementations receive identical tracker coordinates.
#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "athena.hpp"
#include "mesh/forest/tree.hpp"
#include "meshblock_tree.hpp"
#include "z4c/evolution/amr.h"

using Location = parthenon::LogicalLocation;
using Key = std::tuple<int, int, int, int>;
Key key(const Location &loc) {
  return {loc.level(), loc.lx1(), loc.lx2(), loc.lx3()};
}
Location ploc(const LogicalLocation &loc) {
  return Location(0, loc.level, loc.lx1, loc.lx2, loc.lx3);
}
LogicalLocation aloc(const Location &loc) {
  return {static_cast<int>(loc.lx1()), static_cast<int>(loc.lx2()),
          static_cast<int>(loc.lx3()), loc.level()};
}
std::vector<Location> leaves(MeshBlockTree &tree) {
  int count = 0;
  tree.CountMeshBlocks(count);
  std::vector<LogicalLocation> list(count);
  count = 0;
  tree.CreateZOrderedLLList(list.data(), nullptr, count);
  std::vector<Location> result;
  for (const auto &loc : list)
    result.push_back(ploc(loc));
  return result;
}
std::vector<Key> keys(const std::vector<Location> &list) {
  std::vector<Key> result;
  for (const auto &loc : list)
    result.push_back(key(loc));
  std::sort(result.begin(), result.end());
  return result;
}
struct Requests {
  std::vector<Location> refine, coarsen;
};
Requests requests(const std::vector<Location> &list,
                  const std::array<double, 3> &center,
                  const std::shared_ptr<parthenon::forest::Tree> &old_tree,
                  bool legacy_neighbor_veto) {
  Requests result;
  std::map<Key, int> family;
  for (const auto &loc : list) {
    const double width = std::ldexp(64., -loc.level());
    const std::array<double, 6> bounds{
        -32. + loc.lx1() * width, -32. + (loc.lx1() + 1) * width,
        -32. + loc.lx2() * width, -32. + (loc.lx2() + 1) * width,
        -32. + loc.lx3() * width, -32. + (loc.lx3() + 1) * width};
    int flag = pangu::nr::amr::TrackerTag<double>(
        bounds, loc.level() - 1, 1, center, std::array<double, 1>{4.},
        std::array<int, 1>{4});
    if (loc.level() == 5 && flag > 0)
      flag = 0;
    if (loc.level() == 1 && flag < 0)
      flag = 0;
    if (flag > 0)
      result.refine.push_back(loc);
    if (flag < 0) {
      if (legacy_neighbor_veto) {
        bool finer = false;
        for (const auto &nb : old_tree->FindNeighbors(loc))
          finer = finer || nb.global_loc.level() > loc.level();
        if (finer)
          continue;
      }
      const auto parent = loc.GetParent();
      ++family[key(parent)];
    }
  }
  // Retain Morton traversal order, as in both production AMR drivers.
  for (const auto &loc : list) {
    if ((loc.lx1() & 1) || (loc.lx2() & 1) || (loc.lx3() & 1) ||
        loc.level() <= 1)
      continue;
    const auto parent = loc.GetParent();
    if (family[key(parent)] == 8)
      result.coarsen.push_back(parent);
  }
  // Both inspected drivers exclude the final element from this sort. Preserve
  // that reference behavior here; do not fix one framework's ordering only.
  if (result.coarsen.size() > 1)
    std::sort(
        result.coarsen.begin(), result.coarsen.end() - 1,
        [](const auto &a, const auto &b) { return a.level() > b.level(); });
  return result;
}
int main(int argc, char **argv) {
  try {
    if (argc < 4 || argc > 5)
      throw std::runtime_error(
          "usage: mesh_replay seed.txt centers.txt topology.txt [legacy]");
    Mesh mesh;
    MeshBlockTree athena(&mesh);
    athena.CreateRootGrid();
    auto pangu = parthenon::forest::Tree::create(0, 3, 1);
    std::ifstream seed(argv[1]), centers(argv[2]);
    std::ofstream output(argv[3]);
    if (!seed || !centers || !output)
      throw std::runtime_error("cannot open replay files");
    int level, x, y, z;
    while (seed >> level >> x >> y >> z) {
      int added = 0;
      athena.AddNode({x, y, z, level}, added);
      pangu->AddMeshBlock(Location(0, level, x, y, z));
    }
    int cycle = 0;
    long long block_cycles = 0;
    auto check = [&] {
      const auto a = keys(leaves(athena));
      const auto p = keys(pangu->GetSortedMeshBlockList());
      if (a != p)
        throw std::runtime_error("full leaf-set mismatch at cycle " +
                                 std::to_string(cycle));
      output << cycle << ' ' << a.size();
      for (const auto &[l, i, j, k] : a)
        output << ' ' << l << ',' << i << ',' << j << ',' << k;
      output << '\n';
    };
    check();
    std::array<double, 3> center;
    while (centers >> center[0] >> center[1] >> center[2]) {
      block_cycles += leaves(athena).size();
      const auto a = requests(leaves(athena), center, pangu, false);
      const auto p =
          requests(pangu->GetSortedMeshBlockList(), center, pangu, argc == 5);
      for (const auto &loc : a.refine) {
        int added = 0;
        athena.FindMeshBlock(aloc(loc))->Refine(added);
      }
      for (const auto &loc : a.coarsen) {
        int removed = 0;
        athena.FindMeshBlock(aloc(loc))->Derefine(removed);
      }
      for (const auto &loc : p.refine)
        pangu->Refine(loc);
      for (const auto &loc : p.coarsen)
        pangu->Derefine(loc);
      ++cycle;
      check();
    }
    std::cout << "Native AthenaK/Parthenon mesh replay PASS: " << cycle
              << " updates, every leaf location/level matched; block-cycles="
              << block_cycles << '\n';
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
