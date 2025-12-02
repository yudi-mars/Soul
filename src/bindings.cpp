#include "neu-sim/core/SyncCore.hpp"
#include "neu-sim/params.hpp"
#include "neu-sim/top.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
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
  uint32_t total_recv_flits;
  uint32_t total_recv_spikes;
  uint32_t total_sent_spikes;
  uint32_t total_firing_cnt;
};

class Top : public TopBase<SyncCore> {
public:
  auto load_snn(
    py::array_t<uint32_t> &position,
    py::array_t<uint8_t, py::array::c_style> &core_conns,
    py::array_t<uint8_t, py::array::c_style> &spikes
  ) {
    for (int i = 0; i < topo.size(); ++i) {
      auto &core = cores[i];
      auto [y, x] = topo.id2pos(i);
      if (x > 0) {
        core.parents_id.emplace_back(cores[i - 1].id);
      }
      if (x < topo.x - 1) {
        core.children_id.emplace_back(cores[i + 1].id);
      }
      if (y > 0) {
        core.parents_id.emplace_back(cores[i - topo.x].id);
      }
      if (y < topo.y - 1) {
        core.children_id.emplace_back(cores[i + topo.x].id);
      }
    }

    auto rp = position.unchecked<1>();
    auto r1 = core_conns.unchecked<2>();
    auto r2 = spikes.unchecked<2>();
    auto num_neurons = r1.shape(0);
    auto num_cores = r1.shape(1);
    auto max_ts = r2.shape(1);
    printf(
      "Loading SNN: %ld neurons, %ld cores, %ld timesteps\n",
      num_neurons,
      num_cores,
      max_ts
    );
    // printf(
    //   "Neuron 0 conn: %d %d %d %d\n", r1(0, 0), r1(0, 1), r1(0, 2), r1(0, 3)
    // );
    for (py::ssize_t i = 0; i < num_neurons; i++) {
      const uint8_t *axon_ptr = r1.data(i, 0);
      const uint8_t *spike_ptr = r2.data(i, 0);
      // if (i == 0) {
      //   printf("Neuron %ld ts: %d\n", i, spike_ptr[0]);
      // }
      // printf(
      //   "Neuron %ld conn: %d %d %d %d\n",
      //   i,
      //   axon_ptr[0],
      //   axon_ptr[1],
      //   axon_ptr[2],
      //   axon_ptr[3]
      // );
      cores[rp(i)].add_neuron(
        std::span<const uint8_t>(axon_ptr, num_cores),
        std::span<const uint8_t>(spike_ptr, max_ts)
      );
    }

    for (int i = 0; i < topo.size(); ++i) {
      cores[i].init(max_ts);
      cores[i].launch();
    }
  }

  auto result_stats() -> SimResult {
    SimResult res;
    res.retcode = 0;
    res.total_cycles = cores[0].clk.cycle_count;
    res.total_recv_flits = routers[0].clk.stats.num_recv_flits.load();

    res.total_recv_spikes = 0;
    for (auto &core : cores) {
      res.total_recv_spikes += core.stats.total_recv_spikes;
    }

    res.total_sent_spikes = 0;
    for (auto &core : cores) {
      res.total_sent_spikes += core.stats.total_sent_spikes;
    }

    res.total_firing_cnt = 0;
    for (auto &core : cores) {
      res.total_firing_cnt += core.stats.firing_cnt;
    }
    return res;
  }
};

SimResult run_sim(
  py::array_t<uint32_t> position,
  py::array_t<uint8_t, py::array::c_style> core_conns,
  py::array_t<uint8_t, py::array::c_style> spikes,
  py::kwargs kwargs
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

  if (kwargs.contains("packet_size")) {
    global_params.packet_size = kwargs["packet_size"].cast<uint32_t>();
  }

  uint32_t max_tick = 0;
  if (kwargs.contains("max_tick")) {
    max_tick = kwargs["max_tick"].cast<uint32_t>();
  }

  auto r = core_conns.unchecked<2>();
  auto num_cores = r.shape(1);
  auto total_cores = 1;
  for (auto dim : global_params.topo_size) {
    total_cores *= dim;
  }
  if (num_cores > total_cores) {
    throw std::invalid_argument(
      std::format(
        "Number of cores ({}) exceeds topology size ({})",
        num_cores,
        total_cores
      )
    );
  }

  auto top = Top {};
  top.load_snn(position, core_conns, spikes);

  top.run(max_tick);
  return top.result_stats();
}

PYBIND11_MODULE(sim, m) {
  m.doc() = "Wrapper for NeuSim simulator";

  py::class_<SimResult>(m, "SimResult")
    .def(py::init<int, uint32_t, uint32_t>()) // 绑定构造函数
    .def_readwrite("retcode", &SimResult::retcode)
    .def_readwrite("total_cycles", &SimResult::total_cycles)
    .def_readwrite("total_recv_flits", &SimResult::total_recv_flits)
    .def_readwrite("total_recv_spikes", &SimResult::total_recv_spikes)
    .def_readwrite("total_sent_spikes", &SimResult::total_sent_spikes)
    .def_readwrite("total_firing_cnt", &SimResult::total_firing_cnt)
    .def("__repr__", [](const SimResult &r) { // 可选：定义打印格式
      return "<example.SimResult retcode=" + std::to_string(r.retcode) + ">";
    });

  m.def(
    "run",
    &run_sim,
    "NeuSim",
    py::arg("position"),
    py::arg("core_conns"),
    py::arg("spikes")
  );
}
