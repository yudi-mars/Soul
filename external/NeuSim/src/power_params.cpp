#include "neu-sim/power_params.hpp"
#include "toml.hpp" // 确保这里包含的是 toml++ v3.4.0 的头文件
#include <iostream>

PowerParams global_power_params;

auto PowerParams::parse_config(std::string_view file) -> void {
    config_file = std::filesystem::path(file);
    
    // 解析 TOML 文件
    toml::table config;
    try {
        config = toml::parse_file(config_file.string());
    } catch (const toml::parse_error& err) {
        std::cerr << "Parsing failed:\n" << err << "\n";
        exit(EXIT_FAILURE);
    }

    // 获取 energy 节点，这是一个 table
    auto energy_node = config["energy"];
    if (!energy_node) return; // 如果不存在 energy 节点，直接返回

    // 1. Parse Buffer Energy
    // 注意：在 toml++ v3 中，通过 key 获取的是 node_view
    // 需要调用 as_array() 获取指向数组的指针
    if (auto arr = energy_node["buffer"].as_array()) {
        for (auto&& elem : *arr) {
            // elem 是一个 node (可能是 table)
            // 需要将其转换为 table view 才能方便取值
            if (auto tbl = elem.as_table()) {
                BufferEnergy item;
                // 使用 value_or 直接获取值，简化 optional 处理
                item.depth   = tbl->get("depth")->value_or<uint32_t>(0);
                item.width   = tbl->get("width")->value_or<uint32_t>(0);
                item.leakage = tbl->get("leakage")->value_or<double>(0.0);
                item.push    = tbl->get("push")->value_or<double>(0.0);
                item.front   = tbl->get("front")->value_or<double>(0.0);
                item.pop     = tbl->get("pop")->value_or<double>(0.0);
                buffer_energies.push_back(item);
            }
        }
    }

    // 2. Parse Link BitLine Energy
    if (auto arr = energy_node["link_bit_line"].as_array()) {
        for (auto&& elem : *arr) {
            if (auto tbl = elem.as_table()) {
                LinkEnergy item;
                item.length  = tbl->get("length")->value_or<double>(0.0);
                item.leakage = tbl->get("leakage")->value_or<double>(0.0);
                item.dynamic = tbl->get("dynamic")->value_or<double>(0.0);
                link_energies.push_back(item);
            }
        }
    }

    // 3. Parse Router Crossbar Energy
    // 路径: energy -> router -> crossbar
    // 注意中间的节点可能为空，as_array() 会处理空指针链
    if (auto arr = energy_node["router"]["crossbar"].as_array()) {
        for (auto&& elem : *arr) {
            if (auto tbl = elem.as_table()) {
                CrossbarEnergy item;
                item.ports   = tbl->get("ports")->value_or<uint32_t>(0);
                item.width   = tbl->get("width")->value_or<uint32_t>(0);
                item.leakage = tbl->get("leakage")->value_or<double>(0.0);
                item.dynamic = tbl->get("dynamic")->value_or<double>(0.0);
                crossbar_energies.push_back(item);
            }
        }
    }

    // 4. Parse Routing Logic Energy (Table format in TOML)
    // 这是一个 key-value 对的集合，例如: XY = { ... }
    if (auto tbl = energy_node["router"]["routing"].as_table()) {
        for (auto&& [key, val] : *tbl) {
            // val 是一个 node，我们需要把它当做 table 来看待
            if (auto val_tbl = val.as_table()) {
                LogicEnergy item;
                item.algorithm  = std::string(key.str()); 
                item.static_pwr = val_tbl->get("static")->value_or<double>(0.0);
                item.dynamic    = val_tbl->get("dynamic")->value_or<double>(0.0);
                routing_energies.push_back(item);
            }
        }
    }

    // 5. Parse Selection Logic Energy
    if (auto tbl = energy_node["router"]["selection"].as_table()) {
        for (auto&& [key, val] : *tbl) {
            if (auto val_tbl = val.as_table()) {
                LogicEnergy item;
                item.algorithm  = std::string(key.str());
                item.static_pwr = val_tbl->get("static")->value_or<double>(0.0);
                item.dynamic    = val_tbl->get("dynamic")->value_or<double>(0.0);
                selection_energies.push_back(item);
            }
        }
    }
}

auto PowerParams::print() -> void {
    printf("========================================================\n");
    printf("Power Configuration Loaded from: %s\n", config_file.c_str());
    printf("========================================================\n");

    printf("\n[Buffer Energy] (%zu entries)\n", buffer_energies.size());
    printf("%-8s %-8s %-12s %-12s %-12s %-12s\n", "Depth", "Width", "Leakage", "Push", "Front", "Pop");
    printf("----------------------------------------------------------------------\n");
    for (const auto& item : buffer_energies) {
        printf("%-8u %-8u %-12.2e %-12.2e %-12.2e %-12.2e\n", 
            item.depth, item.width, item.leakage, item.push, item.front, item.pop);
    }

    printf("\n[Link BitLine Energy] (%zu entries)\n", link_energies.size());
    printf("%-12s %-12s %-12s\n", "Length(mm)", "Leakage", "Dynamic");
    printf("--------------------------------------\n");
    for (const auto& item : link_energies) {
        printf("%-12.2f %-12.2e %-12.2e\n", 
            item.length, item.leakage, item.dynamic);
    }

    printf("\n[Crossbar Energy] (%zu entries)\n", crossbar_energies.size());
    printf("%-8s %-8s %-12s %-12s\n", "Ports", "Width", "Leakage", "Dynamic");
    printf("------------------------------------------------\n");
    for (const auto& item : crossbar_energies) {
        printf("%-8u %-8u %-12.2e %-12.2e\n", 
            item.ports, item.width, item.leakage, item.dynamic);
    }

    printf("\n[Routing Logic Energy] (%zu entries)\n", routing_energies.size());
    printf("%-20s %-12s %-12s\n", "Algorithm", "Static", "Dynamic");
    printf("------------------------------------------------\n");
    for (const auto& item : routing_energies) {
        printf("%-20s %-12.2e %-12.2e\n", 
            item.algorithm.c_str(), item.static_pwr, item.dynamic);
    }

    printf("\n[Selection Logic Energy] (%zu entries)\n", selection_energies.size());
    printf("%-20s %-12s %-12s\n", "Strategy", "Static", "Dynamic");
    printf("------------------------------------------------\n");
    for (const auto& item : selection_energies) {
        printf("%-20s %-12.2e %-12.2e\n", 
            item.algorithm.c_str(), item.static_pwr, item.dynamic);
    }
    printf("========================================================\n");
}