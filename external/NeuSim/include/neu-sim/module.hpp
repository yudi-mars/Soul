#pragma once

#include "neu-sim/types.hpp"
#include <memory>
#include <vector>

class ClockDomain;

class Module {
public:
  ClockDomain &clk;

  Module(ClockDomain &clk);

  virtual auto cycle() -> void {}

  virtual auto is_stop() -> bool { return true; }

  virtual ~Module() = default;
};
