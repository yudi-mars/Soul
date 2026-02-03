#pragma once

#include "neu-sim/types.hpp"
#include <filesystem>
#include <vector>
#include <string>

// 定义数据结构以匹配 TOML 中的表结构
struct BufferEnergy {
    uint32_t depth;
    uint32_t width;
    double leakage;
    double push;
    double front;
    double pop;
};

struct LinkEnergy {
    double length;
    double leakage;
    double dynamic;
};

struct CrossbarEnergy {
    uint32_t ports;
    uint32_t width;
    double leakage;
    double dynamic;
};

struct LogicEnergy {
    std::string algorithm;
    double static_pwr;
    double dynamic;
};

struct PowerParams {
    // 存储解析后的能耗库数据
    std::vector<BufferEnergy> buffer_energies;
    std::vector<LinkEnergy> link_energies;
    std::vector<CrossbarEnergy> crossbar_energies;
    std::vector<LogicEnergy> routing_energies;
    std::vector<LogicEnergy> selection_energies;

    // 配置文件路径
    std::filesystem::path config_file;

    // 解析函数
    auto parse_config(std::string_view file) -> void;
    
    // 打印调试信息
    auto print() -> void;
};

extern PowerParams global_power_params;