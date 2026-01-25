#pragma once

#include "neu-sim/noc/topo/topology.hpp"

template <typename Derived>
class Mesh2D {
public:
  uint32_t y;
  uint32_t x;
  uint32_t n_port { 1 + 4 };

  Mesh2D(uint32_t y, uint32_t x) : y(y), x(x) {}

  enum Direction { Local, East, West, North, South };

  auto size() -> uint32_t { return y * x; }

  template <std::derived_from<Router> RouterT>
  auto link(std::vector<RouterT> &routers) -> void {
    auto n = routers.size();
    for (auto &router : routers) {
      router.init_ports(n_port, [this](const RouteData &r) {
        return this->route_fn(r);
      });
    }
    for (int i = 0; i < y; ++i) {
      for (int j = 0; j < x; ++j) {
        auto idx = i * x + j;
        if (i > 0)
          routers[idx]
            .get_port(Direction::North)
            .link_to(routers[idx - x].get_port(Direction::South));
        if (i < y - 1)
          routers[idx]
            .get_port(Direction::South)
            .link_to(routers[idx + x].get_port(Direction::North));
        if (j > 0)
          routers[idx]
            .get_port(Direction::West)
            .link_to(routers[idx - 1].get_port(Direction::East));
        if (j < x - 1)
          routers[idx]
            .get_port(Direction::East)
            .link_to(routers[idx + 1].get_port(Direction::West));
      }
    }
  }

  std::tuple<uint32_t, uint32_t> id2pos(uint32_t id) const {
    return { id / x, id % x };
  }

  auto route_fn(const RouteData &r) -> uint32_t {
    return static_cast<Derived *>(this)->route(r);
  };
};

class Mesh2DXY : public Mesh2D<Mesh2DXY> {
public:
  Mesh2DXY(uint32_t y, uint32_t x) : Mesh2D(y, x) {}

  auto route(const RouteData &r) -> uint32_t {
    auto [cur_y, cur_x] = id2pos(r.cur_id);
    auto [dst_y, dst_x] = id2pos(r.dst_id);

    auto res = Direction::Local;
    if (cur_x < dst_x) {
      res = Direction::East;
    } else if (cur_x > dst_x) {
      res = Direction::West;
    } else if (cur_y < dst_y) {
      res = Direction::South;
    } else if (cur_y > dst_y) {
      res = Direction::North;
    }
    return res;
  }
};
