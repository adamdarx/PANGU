#ifndef PANGU_AWA_DIAGNOSTIC_HPP_
#define PANGU_AWA_DIAGNOSTIC_HPP_
#include "z4c/diagnostics/awa_core.hpp"
#include "z4c/evolution/package.h"

namespace pangu::nr {
inline void WriteAwADiagnostics(parthenon::Mesh *mesh, double time, int cycle) {
  static const awa_diagnostic::Config config;
  if (!config.enabled()) return;
  static double next_time=0.0;
  const auto interval_env=std::getenv("AWA_INTERVAL");
  const double interval=interval_env ? std::stod(interval_env) : 1.0;
  const double tolerance=1e-8*std::max(1.0,std::abs(time));
  if (time+tolerance<next_time && time+tolerance<config.final_time) return;
  next_time=(std::floor((time+tolerance)/interval)+1)*interval;
  if (mesh->block_list.size()!=1 || parthenon::Globals::nranks!=1)
    throw std::runtime_error("AwA diagnostic requires one periodic block and one rank");
  const auto block=mesh->block_list.front();
  const auto data=block->meshblock_data.Get();
  Z4cToADMBlockTask(data.get());
  Kokkos::fence("AwA fresh ADM and constraints");
  const auto u=data->Get("nr.z4c").data.GetHostMirrorAndCopy();
  const auto con=data->Get("nr.constraints").data.GetHostMirrorAndCopy();
  const auto ib=block->cellbounds.GetBoundsI(parthenon::IndexDomain::interior);
  const auto jb=block->cellbounds.GetBoundsJ(parthenon::IndexDomain::interior);
  const auto kb=block->cellbounds.GetBoundsK(parthenon::IndexDomain::interior);
  const std::array<int,3> n={ib.e-ib.s+1,jb.e-jb.s+1,kb.e-kb.s+1};
  const std::array<double,3> dx={block->coords.Dxc<parthenon::X1DIR>(kb.s,jb.s,ib.s),
      block->coords.Dxc<parthenon::X2DIR>(kb.s,jb.s,ib.s),
      block->coords.Dxc<parthenon::X3DIR>(kb.s,jb.s,ib.s)};
  const auto state=[&](int a,int k,int j,int i) {
    i=(i%n[0]+n[0])%n[0]; j=(j%n[1]+n[1])%n[1]; k=(k%n[2]+n[2])%n[2];
    return double(u(a,k+kb.s,j+jb.s,i+ib.s));
  };
  const auto constraint=[&](int a,int k,int j,int i) {
    return double(con(a,k+kb.s,j+jb.s,i+ib.s));
  };
  awa_diagnostic::write(config,time,cycle,n,dx,state,constraint);
}
} // namespace pangu::nr
#endif
