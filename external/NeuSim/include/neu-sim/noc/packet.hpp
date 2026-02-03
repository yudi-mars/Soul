#pragma once

#include "neu-sim/types.hpp"

class Packet {
public:
  uint32_t src_pid;
  uint32_t dst_pid;
  uint32_t len;

  Packet(uint32_t src, uint32_t dst, uint32_t len)
    : src_pid(src), dst_pid(dst), len(len) {}

  static std::unique_ptr<Packet>
  new_packet(uint32_t src, uint32_t dst, uint32_t len) {
    return std::make_unique<Packet>(src, dst, len);
  }

  virtual ~Packet() = default;
};
