#pragma once

#include "neu-sim/module.hpp"
#include "neu-sim/noc/flit.hpp"
#include "neu-sim/noc/port.hpp"
#include "neu-sim/noc/power.hpp"
#include "neu-sim/noc/topo/topology.hpp"
#include "neu-sim/params.hpp"
#include "neu-sim/power_params.hpp"
#include "neu-sim/types.hpp"
#include <optional>

class Router : public Module {
private:
  std::function<uint32_t(const RouteData &)> route_func;
  std::vector<Port> ports;
  std::vector<std::unique_ptr<Flit>> xbar;
  std::vector<uint32_t> rr_xbar;

  auto stats_used_ch_cnt() -> void;

public:
  uint32_t id;

  struct Stats {
    uint32_t used_ch_cnt;
    double used_ch_ratio;
    uint32_t total_flits { 0 };   // total flits passed through this router
    uint32_t sent_flits { 0 };    // total flits sent from this router
    uint32_t recv_flits { 0 };    // total flits received by this router
    uint32_t total_packets { 0 }; // total packets passed through this router
    uint32_t sent_packets { 0 };  // total packets sent from this router
    uint32_t recv_packets { 0 };  // total packets received by this router
    uint32_t max_delay { 0 };     // max delay of flits received by this router
    uint64_t total_delay { 0 }; // total delay of flits received by this router
    Power power {};
  } stats;

  Router(ClockDomain &clk, uint32_t id) : Module(clk), id(id) {}

  auto route(const RouteData &r) -> uint32_t { return route_func(r); }

  auto init_ports(
    uint32_t n_port, std::function<uint32_t(const RouteData &)> route_func
  ) {
    this->route_func = route_func;
    ports.reserve(n_port);
    for (int i = 0; i < n_port; ++i) {
      ports.emplace_back(clk, i, global_params.n_ch, global_params.buf_size);
      ports[i].router = this;
      xbar.emplace_back(nullptr);
    }
    rr_xbar = std::vector<uint32_t>(n_port, 0);
  }

  auto init_power() {
    stats.power.configure(
      global_power_params,
      32, // TODO, assume flit width is 32 bits for now
      global_params.buf_size,
      ports.size(),
      global_params.routing,
      "default", // TODO，use default selection algorithm for now
      1.0        // TODO, assume link length is 1.0 mm for now
    );
  }

  auto get_port(uint32_t port_id) -> Port & { return ports[port_id]; }

  auto get_local_port() -> Port & { return ports[0]; }

  auto cycle() -> void override;
};
