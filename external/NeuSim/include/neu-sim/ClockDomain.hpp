#pragma once

#include "neu-sim/module.hpp"
#include "neu-sim/params.hpp"
#include <atomic>
#include <functional>

class ClockDomain {
public:
  uint32_t cycle_period { 1 };
  uint32_t cycle_count { 0 };
  std::vector<Module *> modules;
  std::vector<std::function<void()>> signals;

  enum State { Running, Stopped };

  std::atomic<State> state { Running };
  bool cycle_arrival { false };

  struct Stats {
    ClockDomain &clk;
    uint32_t tot_packets { 0 };
    std::atomic<uint32_t> num_sent_packets { 0 };
    std::atomic<uint32_t> num_recv_packets { 0 };
    std::atomic<uint32_t> num_recv_flits { 0 };

    Stats(ClockDomain &clk) : clk(clk) {}

    auto send_packet() -> void { num_sent_packets++; }

    auto recv_flit() -> void { auto num = ++num_recv_flits; }

    auto recv_packet() -> void { num_recv_packets++; }
  } stats { *this };

  ClockDomain(uint32_t cycle_period) : cycle_period(cycle_period) {}

  ClockDomain(const ClockDomain &) {
    throw std::runtime_error("ClockDomain cannot be copied");
  }

  auto is_stop() -> bool {
    if (stats.num_recv_packets != stats.num_sent_packets)
      return false;
    for (auto module : modules) {
      if (!module->is_stop()) {
        return false;
      }
    }
    return true;
  }

  auto add_module(Module *module) { modules.push_back(module); }

  auto add_signal(std::function<void()> signal) { signals.push_back(signal); }
};
