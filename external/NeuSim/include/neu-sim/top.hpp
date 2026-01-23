#pragma once

#include "neu-sim/ClockDomain.hpp"
#include "neu-sim/SpinBarrier.hpp"
#include "neu-sim/core/BaseCore.hpp"
#include "neu-sim/noc/flit.hpp"
#include "neu-sim/noc/interface.hpp"
#include "neu-sim/noc/router.hpp"
#include "neu-sim/noc/topo/mesh2d.hpp"
#include "neu-sim/params.hpp"
#include <atomic>
#include <chrono>
#include <concepts>
#include <fstream>
#include <thread>

template <
  std::derived_from<BaseCore> CoreT,
  std::derived_from<Router> RouterT = Router,
  typename TopoT = Mesh2DXY>
  requires Topology<TopoT, RouterT>
class TopBase {
protected:
  uint32_t tick_count { 0 };

  std::vector<ClockDomain> domains;

  TopoT topo;

  std::vector<RouterT> routers;
  std::vector<Interface> noc_ifs;
  std::vector<CoreT> cores;

  struct WorkItem {
    uint32_t domain_id;
    uint32_t task_id;
  };

  struct WorkerParams {
    uint32_t num_modules;
    uint32_t num_signals;
    std::vector<WorkItem> modules;
    std::vector<WorkItem> signals;
  };

  std::vector<WorkerParams> worker_params;

  SpinBarrier barrier { global_params.num_workers + 1 };

  auto worker_loop(std::stop_token stop_token, uint32_t worker_id) -> void {
    while (true) {
      barrier.arrive_and_wait();
      if (stop_token.stop_requested())
        break;
      for (const auto &item : worker_params[worker_id].signals) {
        domains[item.domain_id].signals[item.task_id]();
      }

      barrier.arrive_and_wait();
      for (const auto &item : worker_params[worker_id].modules) {
        if (domains[item.domain_id].cycle_arrival)
          domains[item.domain_id].modules[item.task_id]->cycle();
      }

      barrier.arrive_and_wait();
      // main thread check stop condition
    }
  }

public:
  TopBase() : topo(global_params.topo_size[0], global_params.topo_size[1]) {
    domains.reserve(2);
    domains.emplace_back(global_params.tick_noc);
    domains.emplace_back(global_params.tick_core);

    routers.reserve(topo.size());
    for (auto i = 0; i < topo.size(); i++) {
      routers.emplace_back(domains[0], i);
      domains[0].add_module(&routers[i]);
    }
    topo.link(routers);
    for (auto &r : routers) {
      r.init_power(); // initialize power model for each router
    }

    noc_ifs.reserve(topo.size());
    for (auto i = 0; i < topo.size(); i++) {
      noc_ifs.emplace_back(domains[0], i);
      domains[0].add_module(&noc_ifs[i]);
      routers[i].get_local_port().link_to(noc_ifs[i].get_port());
      noc_ifs[i].get_port().link_to(routers[i].get_local_port());
    }

    cores.reserve(topo.size());
    for (auto i = 0; i < topo.size(); i++) {
      cores.emplace_back(domains[1], i);
      domains[1].add_module(&cores[i]);
      cores[i].set_noc_if(&noc_ifs[i]);
      noc_ifs[i].set_core(&cores[i]);
    }
  }

  auto run(uint32_t max_tick = 0) -> double {
    auto start = std::chrono::steady_clock::now();

    if (global_params.num_workers <= 1)
      run_single_thread(max_tick);
    else
      run_multi_thread(max_tick);

    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end - start).count();
  }

  auto run_single_thread(uint32_t max_tick) -> void {
    while (max_tick ? tick_count < max_tick : true) {
      for (auto i = 0; i < domains.size(); i++) {
        if (tick_count % domains[i].cycle_period == 0) {
          domains[i].cycle_count++;
          domains[i].cycle_arrival = true;
        } else {
          domains[i].cycle_arrival = false;
        }
      }

      for (auto i = 0; i < domains.size(); i++)
        for (auto &sig : domains[i].signals)
          sig();

      for (auto i = 0; i < domains.size(); i++)
        if (domains[i].cycle_arrival)
          for (auto &mod : domains[i].modules)
            mod->cycle();

      tick_count++;

      auto all_stopped = true;
      for (auto i = 0; i < domains.size(); i++) {
        if (!domains[i].is_stop()) {
          all_stopped = false;
          break;
        }
      }
      if (all_stopped)
        break;
    }
  }

  auto run_multi_thread(uint32_t max_tick) -> void {
    // prepare worker items
    // printf("prepare worker items\n");
    auto tot_mods = 0;
    auto tot_sigs = 0;
    for (auto i = 0; i < domains.size(); i++) {
      tot_mods += domains[i].modules.size();
      tot_sigs += domains[i].signals.size();
    }
    auto mods_per_worker = tot_mods / global_params.num_workers;
    auto mods_remain = tot_mods % global_params.num_workers;
    auto sigs_per_worker = tot_sigs / global_params.num_workers;
    auto sigs_remain = tot_sigs % global_params.num_workers;
    for (int i = 0; i < global_params.num_workers; ++i) {
      worker_params.emplace_back();
      worker_params[i].num_modules = mods_per_worker;
      if (i < mods_remain)
        worker_params[i].num_modules++;

      worker_params[i].num_signals = sigs_per_worker;
      if (i < sigs_remain)
        worker_params[i].num_signals++;
    }
    auto cur_wid = 0;
    for (auto i = 0; i < domains.size(); i++) {
      for (auto j = 0; j < domains[i].modules.size(); ++j) {
        if (worker_params[cur_wid].modules.size() ==
            worker_params[cur_wid].num_modules)
          cur_wid++;
        worker_params[cur_wid].modules.emplace_back((uint32_t)i, (uint32_t)j);
      }
    }
    cur_wid = 0;
    for (auto i = 0; i < domains.size(); i++) {
      for (auto j = 0; j < domains[i].signals.size(); ++j) {
        if (worker_params[cur_wid].signals.size() ==
            worker_params[cur_wid].num_signals)
          cur_wid++;
        worker_params[cur_wid].signals.emplace_back((uint32_t)i, (uint32_t)j);
      }
    }
    // for (int i = 0; i < global_params.num_workers; ++i) {
    //   printf(
    //     "worker %d: %ld modules, %ld signals\n", i,
    //     worker_params[i].modules.size(), worker_params[i].signals.size()
    //   );
    // }

    // launch workers
    // printf("launch workers\n");
    auto workers = std::vector<std::jthread> {};
    for (int i = 0; i < global_params.num_workers; ++i) {
      workers.emplace_back(&TopBase::worker_loop, this, i);
    }

    while (max_tick ? tick_count < max_tick : true) {
      for (auto i = 0; i < domains.size(); i++) {
        if (tick_count % domains[i].cycle_period == 0) {
          domains[i].cycle_count++;
          domains[i].cycle_arrival = true;
        } else {
          domains[i].cycle_arrival = false;
        }
      }

      barrier.arrive_and_wait();
      // workers doing flush()

      barrier.arrive_and_wait();
      // workers doing cycle()

      barrier.arrive_and_wait();
      tick_count++;

      auto all_stopped = true;
      for (auto i = 0; i < domains.size(); i++) {
        if (!domains[i].is_stop()) {
          all_stopped = false;
          break;
        }
      }
      if (all_stopped)
        break;
    }

    for (auto &w : workers) {
      w.request_stop();
    }

    // printf("done\n");
    barrier.arrive_and_wait();
  }
};
