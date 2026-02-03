#pragma once

#include "neu-sim/ClockDomain.hpp"
#include "neu-sim/iobuffer.hpp"
#include "neu-sim/module.hpp"
#include "neu-sim/noc/flit.hpp"
#include <cassert>
#include <deque>
#include <format>
#include <optional>

class Router;

class VC {
public:
  // for input vc
  uint32_t buf_size;
  std::deque<std::unique_ptr<Flit>> buf;
  std::optional<uint32_t> dst_p;
  std::optional<uint32_t> ovc;

  VC(uint32_t buf_size) : buf_size(buf_size) {}

  VC(VC &&vc) = default;

  bool is_full() const { return buf.size() == buf_size; }

  bool is_empty() const { return buf.empty(); }
};

class Port : public Module {
private:
  IOBuffer<Flit> flit_in;
  IOBuffer<Flit> credit_in;

  std::vector<VC> vc;

  std::vector<uint32_t> credit;
  std::vector<bool> ovc_is_assigned;

  uint32_t rr_ivc { 0 }; // round-robin
  uint32_t rr_ovc { 0 }; // round-robin

public:
  Router *router { nullptr };
  Port *linked_port { nullptr };
  uint32_t port_id;
  uint32_t n_ch;
  uint32_t buf_size;

  Port(ClockDomain &clk, uint32_t port_id, uint32_t n_ch, uint32_t buf_size)
    : Module(clk), port_id(port_id), n_ch(n_ch), buf_size(buf_size) {

    for (int i = 0; i < n_ch; ++i) {
      vc.emplace_back(buf_size);
    }

    clk.add_signal([&] { flush(); });
  }

  auto link_to(Port &port) -> void {
    linked_port = &port;
    for (int i = 0; i < linked_port->n_ch; ++i) {
      credit.emplace_back(linked_port->buf_size);
      ovc_is_assigned.emplace_back(false);
    }
  }

  auto send_flit_ready() -> bool {
    return linked_port && linked_port->recv_flit_ready();
  }

  auto send_credit_ready() -> bool {
    return linked_port && linked_port->recv_credit_ready();
  }

  auto send(std::unique_ptr<Flit> flit) -> void;

  auto dec_credit(uint32_t ovc) -> void {
    assert(credit[ovc] > 0);
    credit[ovc] -= 1;
  }

  auto inc_credit(uint32_t ovc) -> void { credit[ovc] += 1; }

  auto is_empty(uint32_t ivc) const -> bool { return vc[ivc].is_empty(); }

  auto arbitrate_ivc() -> std::optional<uint32_t>;

  auto top_flit(uint32_t ivc) const -> const Flit * {
    return vc[ivc].buf[0].get();
  }

  auto pop_flit(uint32_t ivc) -> std::unique_ptr<Flit> {
    auto res = std::move(vc[ivc].buf[0]);
    vc[ivc].buf.pop_front();
    return res;
  }

  auto has_credit(uint32_t ovc) const -> bool { return credit[ovc] > 0; }

  auto has_free_ovc() const -> bool {
    assert(linked_port != nullptr);
    for (int i = 0; i < linked_port->n_ch; ++i) {
      if (!ovc_is_assigned[i] && has_credit(i)) {
        return true;
      }
    }
    return false;
  }

  auto get_ovc(uint32_t ivc) const -> std::optional<uint32_t> {
    return vc[ivc].ovc;
  }

  auto grant_ovc(uint32_t ivc, uint32_t ovc) -> void { vc[ivc].ovc = ovc; }

  auto ungrant_ovc(uint32_t ivc) -> void { vc[ivc].ovc.reset(); }

  auto grant_xbar(uint32_t ivc) -> void { rr_ivc = (ivc + 1) % n_ch; }

  auto alloc_ovc() -> std::optional<uint32_t> {
    assert(linked_port != nullptr);
    for (int i = 0; i < linked_port->n_ch; ++i) {
      auto ovc = (rr_ovc + i) % linked_port->n_ch;
      if (!ovc_is_assigned[ovc] && has_credit(ovc)) {
        ovc_is_assigned[ovc] = true;
        rr_ovc = (ovc + 1) % linked_port->n_ch;
        return ovc;
      }
    }

    return std::nullopt;
  }

  void free_ovc(uint32_t ovc) {
    assert(linked_port != nullptr);
    ovc_is_assigned[ovc] = false;
  }

private:
  auto buf_flit(std::unique_ptr<Flit> flit) -> void {
    auto idx = flit->vc;
    assert(!this->vc[idx].is_full());

    // write_buffer();
    this->vc[idx].buf.emplace_back(std::move(flit));
  }

  auto flush() -> void;

  auto recv_flit_ready() -> bool { return flit_in.write_ready(); }

  auto recv_credit_ready() -> bool { return credit_in.write_ready(); }

  auto recv(std::unique_ptr<Flit> flit) -> void {
    if (flit->is_credit) {
      if (!credit_in.write_ready())
        throw std::runtime_error(
          std::format("Port {}: credit conflict!", port_id)
        );
      credit_in.write(std::move(flit));
    } else {
      if (!flit_in.write_ready())
        throw std::runtime_error(
          std::format("Port {}: flit conflict!", port_id)
        );
      flit_in.write(std::move(flit));
    }
  }

  auto get_router_id() const -> uint32_t;

  auto route(const Flit *flit) const -> uint32_t;

  auto do_routing(Flit *flit) -> void {
    if (flit->is_head) {
      flit->set_dst_port(route(flit));
      if (vc[flit->vc].dst_p.has_value()) {
        throw std::runtime_error(
          std::format(
            "{} {}: Received a head flit before the last tail flit.",
            get_router_id(),
            port_id
          )
        );
      } else {
        if (!flit->is_tail) {
          vc[flit->vc].dst_p = flit->dst_p;
        }
      }
    } else {
      if (vc[flit->vc].dst_p.has_value()) {
        flit->set_dst_port(vc[flit->vc].dst_p.value());
        if (flit->is_tail) {
          vc[flit->vc].dst_p.reset();
        }
      } else {
        throw std::runtime_error(
          std::format(
            "{} {}: Drop a flit({}).",
            get_router_id(),
            port_id,
            flit->get_type()
          )
        );
      }
    }
  }
};
