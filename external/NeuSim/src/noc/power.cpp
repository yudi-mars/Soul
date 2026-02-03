#include "neu-sim/noc/power.hpp"
#include <cassert>
#include <limits>

using namespace std;

Power::Power() {
  counters = PowerCounters {};
  costs = UnitPowerCosts {};
}

void Power::configure(
  const PowerParams &params,
  uint32_t flit_width,
  uint32_t buffer_depth,
  uint32_t n_ports,
  std::string routing_algo,
  std::string selection_algo,
  double link_length_mm
) {
  bool found_buffer = false;
  // 遍历查找最匹配的 buffer 配置
  for (const auto &entry : params.buffer_energies) {
    if (entry.depth == buffer_depth && entry.width == flit_width) {
      costs.buffer_push_j = entry.push;
      costs.buffer_pop_j = entry.pop;
      costs.buffer_front_j = entry.front;

      // 静态功耗计算：假设每个端口都有 Input Buffer，且可能每个 VC 都有
      // 这里我们简化假设：N_PORTS 个输入缓冲
      // 注意：Leakage 在配置里通常是单个 Buffer 实例的
      costs.total_leakage_w += entry.leakage * n_ports;

      found_buffer = true;
      break;
    }
  }
  // if (!found_buffer) {
  //   cerr << "[Power] Warning: No matching buffer config for depth="
  //        << buffer_depth << " width=" << flit_width << ". Using 0." << endl;
  // }

  // Link 配置通常是离散的长度 (0.5mm, 1.0mm...)，我们需要找到最近的或者插值
  double min_diff = std::numeric_limits<double>::max();
  const LinkEnergy *best_link = nullptr;

  for (const auto &entry : params.link_energies) {
    double diff = std::abs(entry.length - link_length_mm);
    if (diff < min_diff) {
      min_diff = diff;
      best_link = &entry;
    }
  }

  if (best_link) {
    // 注意：配置文件里的 Link 能耗通常是 Per Bit 的
    // 所以 dynamic 需要乘以位宽
    costs.link_j = best_link->dynamic * flit_width;

    // 静态功耗：所有输出链路的总漏电
    // Link Leakage 也是 Per Bit，且每个输出端口都有一条链路
    costs.total_leakage_w += best_link->leakage * flit_width * n_ports;
  } else {
    // cerr << "[Power] Warning: No link config found." << endl;
  }

  // 3. Crossbar
  bool found_xbar = false;
  for (const auto &entry : params.crossbar_energies) {
    // 简单匹配端口数和位宽，或者找最接近的
    if (entry.ports == n_ports && entry.width == flit_width) {
      costs.crossbar_j = entry.dynamic;
      costs.total_leakage_w += entry.leakage; // Crossbar 通常只有一个中心实例
      found_xbar = true;
      break;
    }
  }
  // if (!found_xbar) {
  //   cerr << "[Power] Warning: No matching crossbar config found." << endl;
  // }

  // 4. Routing Logic
  bool found_routing = false;
  for (const auto &entry : params.routing_energies) {
    if (entry.algorithm == routing_algo) {
      costs.routing_j = entry.dynamic;
      costs.total_leakage_w += entry.static_pwr;
      found_routing = true;
      break;
    }
  }
  // Fallback to "default" if specific algo not found
  if (!found_routing) {
    for (const auto &entry : params.routing_energies) {
      if (entry.algorithm == "default") {
        costs.routing_j = entry.dynamic;
        costs.total_leakage_w += entry.static_pwr;
        break;
      }
    }
  }

  // 5. Selection (Arbitration) Logic
  bool found_sel = false;
  for (const auto &entry : params.selection_energies) {
    if (entry.algorithm == selection_algo) {
      costs.selection_j = entry.dynamic;
      costs.total_leakage_w += entry.static_pwr;
      found_sel = true;
      break;
    }
  }
  if (!found_sel) {
    for (const auto &entry : params.selection_energies) {
      if (entry.algorithm == "default") {
        costs.selection_j = entry.dynamic;
        costs.total_leakage_w += entry.static_pwr;
        break;
      }
    }
  }
}

// --- Accumulators ---

void Power::bufferPush() { counters.buffer_push++; }

void Power::bufferPop() { counters.buffer_pop++; }

void Power::bufferFront() { counters.buffer_front++; }

void Power::routing() { counters.routing++; }

void Power::selection() { counters.selection++; }

void Power::crossBar() { counters.crossbar++; }

void Power::linkTraversal() { counters.link++; }

// --- Calculators ---

double Power::getDynamicEnergy() const {
  double energy = 0.0;
  energy += counters.buffer_push * costs.buffer_push_j;
  energy += counters.buffer_pop * costs.buffer_pop_j;
  energy += counters.buffer_front * costs.buffer_front_j;
  energy += counters.routing * costs.routing_j;
  energy += counters.selection * costs.selection_j;
  energy += counters.crossbar * costs.crossbar_j;
  energy += counters.link * costs.link_j;
  return energy;
}

double
Power::getStaticEnergy(uint64_t total_cycles, double period_seconds) const {
  double total_time = total_cycles * period_seconds;
  return costs.total_leakage_w * total_time;
}

double
Power::getTotalEnergy(uint64_t total_cycles, double period_seconds) const {
  return getDynamicEnergy() + getStaticEnergy(total_cycles, period_seconds);
}

void Power::printBreakdown(
  std::ostream &out, uint64_t total_cycles, double period_seconds
) const {
  double total_time = total_cycles * period_seconds;
  double e_dynamic = getDynamicEnergy();
  double e_static = getStaticEnergy(total_cycles, period_seconds);

  out << "--- Power Breakdown (Energy in Joules) ---" << endl;
  out << "Counts:" << endl;
  out << "  Buffer Push: " << counters.buffer_push << endl;
  out << "  Buffer Pop:  " << counters.buffer_pop << endl;
  out << "  Routing:     " << counters.routing << endl;
  out << "  Selection:   " << counters.selection << endl;
  out << "  Crossbar:    " << counters.crossbar << endl;
  out << "  Link:        " << counters.link << endl;

  out << "Dynamic Energy: " << e_dynamic << " J" << endl;
  out << "  Buffer:   "
      << (counters.buffer_push * costs.buffer_push_j +
          counters.buffer_pop * costs.buffer_pop_j)
      << " J" << endl;
  out << "  Crossbar: " << (counters.crossbar * costs.crossbar_j) << " J"
      << endl;
  out << "  Link:     " << (counters.link * costs.link_j) << " J" << endl;
  out << "  Logic:    "
      << (counters.routing * costs.routing_j +
          counters.selection * costs.selection_j)
      << " J" << endl;

  out << "Static Energy: " << e_static << " J (" << costs.total_leakage_w
      << " W * " << total_time << " s)" << endl;
  out << "Total Energy:  " << (e_dynamic + e_static) << " J" << endl;
  out << "------------------------------------------" << endl;
}
