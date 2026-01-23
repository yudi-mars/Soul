#pragma once

#include "neu-sim/noc/packet.hpp"
#include "neu-sim/types.hpp"

class Flit {
public:
  uint32_t src_id;
  uint32_t dst_id;
  uint32_t dst_p;
  uint32_t vc { 0 };

  std::unique_ptr<Packet> payload;

  bool is_head { false };
  bool is_tail { false };

  // credit flit
  bool is_credit { false };

  // statistics
  struct {
    uint32_t inject_cycle { 0 };
  } stats;

  Flit(uint32_t src, uint32_t dst) : src_id(src), dst_id(dst) {}

  Flit(uint32_t vc) : vc(vc), is_credit(true) {}

  static std::unique_ptr<Flit> new_flit(uint32_t src, uint32_t dst) {
    return std::make_unique<Flit>(src, dst);
  }

  static std::unique_ptr<Flit> new_credit(uint32_t vc) {
    return std::make_unique<Flit>(vc);
  }

  void set_head() { is_head = true; }

  void set_body() {
    is_head = false;
    is_tail = false;
  }

  void set_tail() { is_tail = true; }

  void set_headtail() {
    is_head = true;
    is_tail = true;
  }

  void set_payload(std::unique_ptr<Packet> payload) {
    this->payload = std::move(payload);
  }

  void set_dst_port(uint32_t dst_p) { this->dst_p = dst_p; }

  void set_vc(uint32_t vc) { this->vc = vc; }

  bool same_place_with(const Flit *that) const {
    return this->src_id == that->src_id && this->dst_id == that->dst_id;
  }

  std::string_view get_type() const {
    if (is_head) {
      if (is_tail) {
        return "HEADTAIL";
      } else {
        return "HEAD";
      }
    } else if (is_tail) {
      return "TAIL";
    } else {
      return "BODY";
    }
  }
};
