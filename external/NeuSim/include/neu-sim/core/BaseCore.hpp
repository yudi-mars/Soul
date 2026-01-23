#pragma once

#include "neu-sim/iobuffer.hpp"
#include "neu-sim/noc/interface.hpp"
#include "neu-sim/noc/packet.hpp"

class BaseCore : public Module {
private:
  IOBuffer<Packet> packet_in;

  Interface *noc_if { nullptr };

  virtual auto core_cycle() -> void = 0;

public:
  uint32_t id;

  BaseCore(ClockDomain &clk, uint32_t id)
    : Module(clk), id(id), packet_in(global_params.buf_size) {
    clk.add_signal([&] { packet_in.flush(); });
  }

  auto set_noc_if(Interface *iface) -> void { noc_if = iface; }

  auto cycle() -> void final { core_cycle(); }

  auto consume_packet() -> void { packet_in.take(); }

  auto get_packet() -> Packet * { return packet_in.read(); }

  auto recv_packet(std::unique_ptr<Packet> packet) -> void {
    packet_in.write(std::move(packet));
  }

  auto recv_packet_ready() -> bool { return packet_in.write_ready(); }

  auto send_packet(std::unique_ptr<Packet> packet) -> void {
    noc_if->send_packet(std::move(packet));
  }

  auto send_packet_ready() -> bool { return noc_if->send_packet_ready(); }

  auto is_stop() -> bool override { return packet_in.is_clear(); }
};
