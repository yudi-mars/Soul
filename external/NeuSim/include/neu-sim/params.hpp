#pragma once

#include "neu-sim/types.hpp"
#include <filesystem>

struct Params {
  uint32_t num_workers { 1 };

  // config file
  std::filesystem::path config_file;
  auto parse_config(std::string_view file) -> void;
  auto print() -> void;

  // NoC
  std::string topology { "Mesh2D" };
  std::vector<uint32_t> topo_size { 4, 4 };
  std::string routing { "YX" };
  uint32_t n_ch { 1 };
  uint32_t buf_size { 4 };
  uint32_t packet_size { 1 };
  uint32_t tick_noc { 1 };
  uint32_t tick_core { 1 };

  // data
  std::filesystem::path traffic_table_file;
};

extern Params global_params;
