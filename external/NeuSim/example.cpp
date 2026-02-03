#include "neu-sim/core/PacketInjector.hpp"
#include "neu-sim/params.hpp"
#include "neu-sim/power_params.hpp"
#include "neu-sim/top.hpp"
#include <algorithm>
#include <chrono>
#include <numeric>
#include <ranges>
#include <thread>
#include <vector>

class Top : public TopBase<PacketInjector> {
public:
  auto load_test() -> bool {
    auto filename = global_params.traffic_table_file;
    std::printf("load traffic table %s", filename.c_str());
    std::ifstream fin(filename, std::fstream::in);
    if (!fin)
      return false;

    int num_cores;
    int num_core_conns;
    int tot_num_packets = 0;
    fin >> num_cores >> num_core_conns;
    assert(num_cores == topo.size());

    for (int i = 0; i < num_cores; ++i) {
      cores[i].init_traffic(num_core_conns);
    }

    for (int i = 0; i < num_core_conns; i++) {
      int src, dst, num_packets;
      fin >> src >> dst >> num_packets;
      cores[src].add_traffic(dst, num_packets);
      tot_num_packets += num_packets;
    }
    std::printf(" done\n");
    if (tot_num_packets == 0) {
      tot_num_packets = 1;
      cores[0].add_traffic(1, 1);
    }
    std::printf("load %d packets\n", tot_num_packets);

    domains[0].stats.tot_packets = tot_num_packets;

    return true;
  }

  auto save_stats() -> void {
    std::printf("Total cycles: %d\n", domains[0].cycle_count);
    std::printf(
      "Total received packets: %d\n", domains[0].stats.num_recv_packets.load()
    );
    std::printf(
      "Total received flits: %d\n", domains[0].stats.num_recv_flits.load()
    );

    auto total_delay = std::transform_reduce(
      routers.begin(),
      routers.end(),
      (u_int64_t)0,
      std::plus<>(),
      [](const auto &r) { return r.stats.total_delay; }
    );
    std::printf(
      "Global average delay (cycles): %f\n",
      domains[0].stats.num_recv_flits == 0
        ? 0
        : (double)total_delay / domains[0].stats.num_recv_flits
    );

    auto max_delay =
      std::ranges::max_element(routers, std::ranges::less {}, [](auto &r) {
        return r.stats.max_delay;
      })->stats.max_delay;
    std::printf("Max delay (cycles): %d\n", max_delay);

    auto throughput =
      (double)domains[0].stats.num_recv_flits / domains[0].cycle_count;
    auto throughputPerIP = throughput / routers.size();
    std::printf("Network throughput (flits/cycle): %f\n", throughput);
    std::printf(
      "Average IP throughput (flits/cycle/IP): %f\n", throughputPerIP
    );

    auto static_power = std::transform_reduce(
      routers.begin(),
      routers.end(),
      0.0,
      std::plus<>(),
      [&](const auto &r) { 
        return r.stats.power.getStaticEnergy(
          domains[0].cycle_count,
          global_params.tick_noc * 1e-9 // TODO, the frequency is unknown here
        ); 
      }
    );
    auto dynamic_power = std::transform_reduce(
      routers.begin(),
      routers.end(),
      0.0,
      std::plus<>(),
      [&](const auto &r) { return r.stats.power.getDynamicEnergy(); }
    );
    auto total_power = static_power + dynamic_power;
    std::printf("Total Power (J): %.6E\n", total_power);
    std::printf("\tStatic Power (J): %.6E\n", static_power);
    std::printf("\tDynamic Power (J): %.6E\n", dynamic_power);

    std::printf("\ntotal flits/total packets/send packets/recv packets\n");
    for (auto &r : routers) {
      std::printf("%d ", r.stats.total_flits);
    }
    std::printf("\n");

    for (auto &r : routers) {
      std::printf("%d ", r.stats.total_packets);
    }
    std::printf("\n");

    for (auto &r : routers) {
      std::printf("%d ", r.stats.sent_packets);
    }
    std::printf("\n");

    for (auto &r : routers) {
      std::printf("%d ", r.stats.recv_packets);
    }
    std::printf("\n");
  }
};

int main(int argc, char const *argv[]) {
  if (argc > 1) {
    global_params.parse_config(argv[1]);
  }

  uint32_t num_threads = 1;
  if (argc > 2) {
    num_threads = std::stoi(argv[2]);
  }
  global_params.num_workers = num_threads;

  if (argc > 3) {
    global_power_params.parse_config(argv[3]);
  }

  auto top = Top {};
  top.load_test();

  uint32_t max_tick = 0;
  if (argc > 4) {
    max_tick = std::stoi(argv[4]);
  }
  top.run(max_tick);
  top.save_stats();
}
