#include "neu-sim/noc/port.hpp"
#include "neu-sim/noc/router.hpp"

auto Port::get_router_id() const -> uint32_t { return router->id; }

auto Port::send(std::unique_ptr<Flit> flit) -> void {
  // [Power Modeling]
  if (router) {
    router->stats.power.linkTraversal();
  }
  if (!flit->is_credit) {
    if (router && port_id == 0) {
      router->stats.recv_flits++;
      if (flit->is_head) {
        router->stats.recv_packets++;
      }

      uint32_t latency = clk.cycle_count - flit->stats.inject_cycle;
      router->stats.total_delay += latency;
      if (latency > router->stats.max_delay) {
        router->stats.max_delay = latency;
      }
    }
  }
  linked_port->recv(std::move(flit));
}

auto Port::flush() -> void {
  credit_in.flush();
  if (!credit_in.empty()) {
    auto credit = credit_in.take();
    inc_credit(credit->vc);
  }

  flit_in.flush();
  if (!flit_in.empty()) {
    auto flit = flit_in.take();

    if (router != nullptr) {
      router->stats.power.bufferPush();
      if (flit->is_head) {
        router->stats.power.routing();
      }
      do_routing(flit.get());
      router->stats.total_flits++;
      if (flit->is_head) {
        router->stats.total_packets++;
      }
      if (port_id == 0) {
        router->stats.sent_flits++;
        flit->stats.inject_cycle = clk.cycle_count;
        if (flit->is_head) {
          router->stats.sent_packets++;
        }
      }
    } else {
      clk.stats.recv_flit();
    }
    buf_flit(std::move(flit));
  }
}

auto Port::route(const Flit *flit) const -> uint32_t {
  return router->route(
    { get_router_id(), flit->src_id, flit->dst_id, port_id, flit->vc }
  );
}

auto Port::arbitrate_ivc() -> std::optional<uint32_t> {
  // for (int i = 0; i < n_ch; ++i) {
  //   if (!vc[i].is_empty())
  //     read_buffer();
  // }
  for (int i = 0; i < n_ch; ++i) {
    auto ivc = (rr_ivc + i) % n_ch;
    if (vc[ivc].is_empty()) //  || ivc_wait_mask(ivc)
      continue;
    // [Power Modeling]
    if (router && !vc[ivc].is_empty()) {
      router->stats.power.bufferFront();
    }
    auto flit = top_flit(ivc);

    auto &dst_port = router->get_port(flit->dst_p);

    auto ovc = vc[ivc].ovc;
    auto valid = true;
    if (ovc.has_value()) {
      if (!dst_port.has_credit(ovc.value()))
        valid = false;
    } else {
      if (!dst_port.has_free_ovc())
        valid = false;
    }

    if (valid)
      return ivc;
  }

  return std::nullopt;
}
