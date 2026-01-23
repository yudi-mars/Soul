#include "neu-sim/noc/router.hpp"
#include "neu-sim/noc/port.hpp"

auto Router::cycle() -> void {
  // simulate combination logics, if it is a pipeline,
  // it's better to simulate it reversely (i.e., stage N -> ... -> stage 1)
  // stage 2: crossbar, switch traversal and send
  int send_cnt = 0;
  auto n_port = ports.size();
  for (int i = 0; i < n_port; ++i) {
    if (xbar[i]) {
      if (!ports[i].send_flit_ready())
        continue;
      ports[i].send(std::move(xbar[i]));
      send_cnt += 1;
    }
  }

  // stage 1: parallel lookahead routing and VC/switch allocation
  // Lookahead routing means do routing for the next hop in this hop and store
  // it in the packet, thus the routing stage can be removed.
  // In software simulation, we just do routing once a head flit coming.
  stats_used_ch_cnt();

  // arbitrate vc
  auto vc_winner = std::vector<std::optional<uint32_t>>(ports.size());
  for (int i = 0; i < ports.size(); ++i) {
    if (ports[i].send_credit_ready())
      vc_winner[i] = ports[i].arbitrate_ivc();
  }

  // vc/sw allocation
  // auto granted = std::vector<bool>(n_port, false);
  for (int i = 0; i < n_port; ++i) {
    // for each out port
    if (xbar[i] != nullptr)
      continue;

    auto op = i;
    auto &oport = ports[op];
    for (int j = 0; j < n_port; ++j) {
      // for each in port
      auto ip = (rr_xbar[i] + j) % n_port;
      auto &iport = ports[ip];

      if (!vc_winner[ip].has_value())
        continue;

      // if (granted[ip])
      //     continue;
      // auto winner = iport->arbitrate_ivc(op);
      // if (!winner.has_value())
      //     continue;
      // auto ivc = winner.value();

      auto ivc = vc_winner[ip].value();
      if (iport.top_flit(ivc)->dst_p != op)
        continue;
      // [Power Modeling]
      this->stats.power.selection();
      vc_winner[ip].reset();

      // [Power Modeling]
      this->stats.power.bufferPop();
      auto flit = iport.pop_flit(ivc);
      auto ovc = iport.get_ovc(ivc);
      if (flit->is_head) {
        // if flit is a head, it must be allocated only at this stage
        assert(!ovc.has_value());
        auto ovc = oport.alloc_ovc();
        assert(ovc.has_value());
        iport.grant_ovc(ivc, ovc.value());
      } else {
        // else it must have already been allocated
        assert(ovc.has_value());
      }

      auto ovc_valid = iport.get_ovc(ivc).value();
      flit->set_vc(ovc_valid);
      oport.dec_credit(ovc_valid);

      if (flit->is_tail) {
        iport.ungrant_ovc(ivc);
        oport.free_ovc(ovc_valid);
      }
      iport.send(Flit::new_credit(ivc));

      rr_xbar[op] = (ip + 1) % n_port;
      iport.grant_xbar(ivc);
      // granted[ip] = true;
      // [Power Modeling]
      this->stats.power.crossBar();
      xbar[op] = std::move(flit);
      break;
    }
  }
}

auto Router::stats_used_ch_cnt() -> void {
  stats.used_ch_cnt = 0;
  auto n_ch = 0;
  for (auto &port : ports) {
    for (int j = 0; j < port.n_ch; ++j) {
      if (!port.is_empty(j))
        stats.used_ch_cnt += 1;
    }
    n_ch += port.n_ch;
  }
  stats.used_ch_ratio = (double)stats.used_ch_cnt / n_ch;
}
