#pragma once

#include "neu-sim/iobuffer.hpp"
#include "neu-sim/module.hpp"
#include "neu-sim/noc/flit.hpp"
#include "neu-sim/noc/packet.hpp"
#include "neu-sim/noc/port.hpp"

class BaseCore;

class Interface : public Module {
private:
  BaseCore *core { nullptr };
  Port port_noc;
  std::deque<std::unique_ptr<Flit>> flit_out_buf;
  IOBuffer<Packet> packet_in;

public:
  uint32_t id;
  uint32_t buf_size;

  Interface(ClockDomain &clk, uint32_t id)
    : Module(clk), id(id), buf_size(global_params.buf_size),
      port_noc(clk, 0, 1, global_params.buf_size),
      packet_in(global_params.buf_size) {
    clk.add_signal([&] { packet_in.flush(); });
  }

  auto set_core(BaseCore *c) -> void { core = c; }

  auto get_port() -> Port & { return port_noc; }

  auto cycle() -> void override;

  auto send_packet_ready() -> bool;

  auto send_packet(std::unique_ptr<Packet> packet) -> void;

  auto fill_buffer(std::unique_ptr<Packet> packet) -> void;

  // auto flush() -> void {
  //   if (!core_in_buf.empty()) {
  //     fill_buffer(std::move(core_in_buf.front()));
  //     core_in_buf.pop_front();
  //   }
  // }
};
