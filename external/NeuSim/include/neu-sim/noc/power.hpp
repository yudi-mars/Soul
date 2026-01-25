#pragma once

#include "neu-sim/power_params.hpp"
#include <cmath>
#include <iostream>
#include <string>

// 用于统计各类事件发生的次数
struct PowerCounters {
  uint64_t buffer_push = 0;
  uint64_t buffer_pop = 0;
  uint64_t buffer_front = 0;
  uint64_t routing = 0;
  uint64_t selection = 0;
  uint64_t crossbar = 0;
  uint64_t link = 0;
  // uint64_t ni = 0;        // Network Interface, Not yet characterized (TBD)
  // in noxim
};

struct UnitPowerCosts {
  // Dynamic (J)
  double buffer_push_j = 0.0;
  double buffer_pop_j = 0.0;
  double buffer_front_j = 0.0;
  double routing_j = 0.0;
  double selection_j = 0.0;
  double crossbar_j = 0.0;
  double link_j = 0.0;
  double ni_j = 0.0;

  // Static (W)
  double total_leakage_w = 0.0;
};

class Power {
public:
  Power();

  // 配置函数：根据路由器的物理参数，从全局配置中提取能耗系数
  void configure(
    const PowerParams &params,
    uint32_t flit_width,
    uint32_t buffer_depth,
    uint32_t n_ports,
    std::string routing_algo,
    std::string selection_algo,
    double link_length_mm
  );

  // --- 事件触发接口 ---
  void bufferPush();
  void bufferPop();
  void bufferFront();

  void routing();
  void selection();
  void crossBar();
  void linkTraversal();

  // --- 统计获取接口 ---

  // 获取当前累积的动态能耗 (Joules)
  double getDynamicEnergy() const;

  // 获取静态能耗 (Joules) = 漏电功率 * 周期数 * 周期时间(秒)
  double getStaticEnergy(uint64_t total_cycles, double period_seconds) const;

  // 获取总能耗
  double getTotalEnergy(uint64_t total_cycles, double period_seconds) const;

  // 打印能耗明细
  void printBreakdown(
    std::ostream &out, uint64_t total_cycles, double period_seconds
  ) const;

private:
  PowerCounters counters;
  UnitPowerCosts costs;
};
