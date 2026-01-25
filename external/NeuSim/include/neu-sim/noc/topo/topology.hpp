#pragma once

#include "neu-sim/types.hpp"
#include <concepts>
#include <vector>

class Router;

struct RouteData {
  uint32_t cur_id;
  uint32_t src_id;
  uint32_t dst_id;
  uint32_t port_id;
  uint32_t vc;
};

template <typename TopoT, typename RouterT>
concept Topology =
  std::derived_from<RouterT, Router> &&
  requires(TopoT t, const RouteData &r, std::vector<RouterT> &rv) {
    { t.size() } -> std::convertible_to<uint32_t>;
    { t.route(r) } -> std::convertible_to<uint32_t>;
    { t.link(rv) } -> std::same_as<void>;
  };
