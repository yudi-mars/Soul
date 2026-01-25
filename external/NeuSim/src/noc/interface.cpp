#include "neu-sim/noc/interface.hpp"
#include "neu-sim/core/BaseCore.hpp"

auto Interface::cycle() -> void {
  // receive from NoC
  // if core is not ready, the recv is blocked
  // otherwise read a flit from the NoC
  if (core->recv_packet_ready()) {
    if (!port_noc.is_empty(0) && port_noc.send_credit_ready()) {
      auto flit = port_noc.pop_flit(0);
      if (flit->is_tail) {
        clk.stats.recv_packet();
        core->recv_packet(std::move(flit->payload));
      }
      port_noc.send(Flit::new_credit(0));
    }
  }

  // send to noc
  if (!packet_in.empty())
    fill_buffer(packet_in.take());

  if (!flit_out_buf.empty() && port_noc.send_flit_ready()) {
    auto flit = flit_out_buf.front().get();
    auto valid = true;
    auto dst_p = flit->dst_p;
    auto ovc = port_noc.get_ovc(0);
    if (ovc.has_value()) {
      if (!port_noc.has_credit(ovc.value()))
        valid = false;
    } else {
      if (!port_noc.has_free_ovc())
        valid = false;
    }

    if (valid) {
      // read_buffer();
      auto flit_valid = std::move(flit_out_buf.front());
      flit_out_buf.pop_front();
      if (flit_valid->is_head) {
        // if flit is a head, it must be allocated only at this stage
        assert(!ovc.has_value());
        auto ovc = port_noc.alloc_ovc();
        assert(ovc.has_value());
        port_noc.grant_ovc(0, ovc.value());
      } else {
        // else it must have already been allocated
        assert(ovc.has_value());
      }

      auto ovc_valid = port_noc.get_ovc(0).value();
      flit_valid->set_vc(ovc_valid);
      port_noc.dec_credit(ovc_valid);

      if (flit_valid->is_tail) {
        port_noc.ungrant_ovc(0);
        port_noc.free_ovc(ovc_valid);
      }

      port_noc.send(std::move(flit_valid));
    }
  }
}

auto Interface::send_packet_ready() -> bool { return packet_in.write_ready(); }

auto Interface::send_packet(std::unique_ptr<Packet> packet) -> void {
  // printf("Interface %d send packet to NoC\n", id);
  clk.stats.send_packet();
  packet_in.write(std::move(packet));
}

auto Interface::fill_buffer(std::unique_ptr<Packet> packet) -> void {
  auto n = packet->len;
  auto src = packet->src_pid;
  auto dst = packet->dst_pid;
  // auto ptype = packet->ptype;
  // auto ts = packet->ts;
  // auto debug = packet->debug;
  assert(src != dst);

  for (int i = 0; i < n; ++i) {
    auto flit = Flit::new_flit(src, dst);
    // flit->ptype = ptype;
    // flit->ts = ts;
    // flit->debug = debug;
    if (i == 0) {
      flit->set_head();
    }
    if (i == n - 1) {
      flit->set_payload(std::move(packet));
      flit->set_tail();
    }
    // write_buffer();
    flit_out_buf.emplace_back(std::move(flit));
  }
}
