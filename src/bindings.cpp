#include "neu-sim/core/PacketInjector.hpp"
#include "neu-sim/params.hpp"
#include "neu-sim/top.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // allow automatic conversion between Python lists and std::vector
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;
using namespace std;

struct SimResult {
  int retcode;
  uint32_t total_cycles;
  uint32_t total_received_flits;
};

class Top : public TopBase<PacketInjector> {
public:
  auto load_test() -> bool {
    auto filename = global_params.traffic_table_file;
    std::printf("load traffic table %s", filename.c_str());
    std::ifstream fin(filename, std::fstream::in);
    if (!fin) {
      std::printf(" failed\n");
      return false;
    }

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
    std::printf("total cycles: %d\n", cores[0].clk.cycle_count);
    std::printf(
      "total received flits: %d\n", routers[0].clk.stats.num_recv_flits.load()
    );

    std::printf("total flits/total packets/send packtes/recv packets\n");
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

  auto result_stats() -> SimResult {
    SimResult res;
    res.retcode = 0;
    res.total_cycles = cores[0].clk.cycle_count;
    res.total_received_flits = routers[0].clk.stats.num_recv_flits.load();
    return res;
  }
};

SimResult run_sim(
  py::array_t<uint32_t> position, py::array_t<bool> core_conns,
  py::array_t<bool> spikes, py::kwargs kwargs
) {
  if (kwargs.contains("num_threads")) {
    global_params.num_workers = kwargs["num_threads"].cast<uint32_t>();
  }

  if (kwargs.contains("topology")) {
    global_params.topology = kwargs["topology"].cast<std::string>();
  }

  if (kwargs.contains("topology_size")) {
    global_params.topo_size =
      kwargs["topology_size"].cast<std::vector<uint32_t>>();
  }

  if (kwargs.contains("routing")) {
    global_params.routing = kwargs["routing"].cast<std::string>();
  }

  if (kwargs.contains("traffic_table")) {
    global_params.traffic_table_file =
      std::filesystem::path(kwargs["traffic_table"].cast<std::string>())
        .lexically_normal();
  }

  if (kwargs.contains("max_packet_size")) {
    global_params.packet_size = kwargs["max_packet_size"].cast<uint32_t>();
  }

  uint32_t max_tick = 0;
  if (kwargs.contains("max_tick")) {
    max_tick = kwargs["max_tick"].cast<uint32_t>();
  }

  auto top = Top {};
  top.load_test();

  top.run(max_tick);
  return top.result_stats();
}

PYBIND11_MODULE(sim, m) {
  m.doc() = "Wrapper for NeuSim simulator";

  py::class_<SimResult>(m, "SimResult")
    .def(py::init<int, uint32_t, uint32_t>()) // 绑定构造函数
    .def_readwrite("retcode", &SimResult::retcode)
    .def_readwrite("total_cycles", &SimResult::total_cycles)
    .def_readwrite("total_received_flits", &SimResult::total_received_flits)
    .def("__repr__", [](const SimResult &r) { // 可选：定义打印格式
      return "<example.SimResult retcode=" + std::to_string(r.retcode) + ">";
    });

  m.def(
    "run", &run_sim, "NeuSim", py::arg("position"), py::arg("core_conns"),
    py::arg("spikes")
  );
}
