#pragma once

#include "neu-sim/ClockDomain.hpp"
#include "neu-sim/core/BaseCore.hpp"
#include "neu-sim/noc/interface.hpp"

class PacketInjector : public BaseCore {
private:
  std::vector<uint32_t> traffic_table;
  uint32_t traffic_p;

  auto get_next_dst() -> int {
    int dst = traffic_p;
    while (dst < traffic_table.size()) {
      if (traffic_table[dst] > 0)
        break;
      dst += 1;
    }
    if (dst < traffic_table.size()) {
      traffic_p = dst;
      return dst;
    }
    return -1;
  }

public:
  PacketInjector(ClockDomain &clk, uint32_t id) : BaseCore(clk, id) {}

  auto core_cycle() -> void override {
    consume_packet();

    if (send_packet_ready()) {
      auto dst = get_next_dst();
      if (dst >= 0) {
        traffic_table[dst] -= 1;
        send_packet(Packet::new_packet(id, dst, global_params.packet_size));
      }
    }
  }

  auto init_traffic(uint32_t num_dst) -> void {
    traffic_table = std::vector<uint32_t>(num_dst, 0);
    traffic_p = 0;
  }

  auto add_traffic(uint32_t dst, uint32_t num) -> void {
    traffic_table[dst] += num;
  }
};
