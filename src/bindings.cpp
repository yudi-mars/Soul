#include "neu-sim/core/PacketInjector.hpp"
#include "neu-sim/params.hpp"
#include "neu-sim/top.hpp"
#include <algorithm>
#include <chrono>
#include <thread>
#include <vector>
#include <string>
#include <fstream>
#include <cassert>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // allow automatic conversion between Python lists and std::vector

class Top : public TopBase<PacketInjector>
{
public:
    auto load_test() -> bool
    {
        auto filename = global_params.traffic_table_file;
        std::printf("load traffic table %s", filename.c_str());
        std::ifstream fin(filename, std::fstream::in);
        if (!fin)
            return false;

        int num_cores;
        int num_core_conns;
        int tot_num_packets = 0;
        fin >> num_cores >> num_core_conns;
        assert(num_cores == topo.size());

        for (int i = 0; i < num_cores; ++i)
        {
            cores[i].init_traffic(num_core_conns);
        }

        for (int i = 0; i < num_core_conns; i++)
        {
            int src, dst, num_packets;
            fin >> src >> dst >> num_packets;
            cores[src].add_traffic(dst, num_packets);
            tot_num_packets += num_packets;
        }
        std::printf(" done\n");
        if (tot_num_packets == 0)
        {
            tot_num_packets = 1;
            cores[0].add_traffic(1, 1);
        }
        std::printf("load %d packets\n", tot_num_packets);

        domains[0].stats.tot_packets = tot_num_packets;

        return true;
    }

    auto save_stats() -> void
    {
        std::printf("total cycles: %d\n", cores[0].clk.cycle_count);
        std::printf(
            "total received flits: %d\n", routers[0].clk.stats.num_recv_flits.load());

        std::printf("total flits/total packets/send packtes/recv packets\n");
        for (auto &r : routers)
        {
            std::printf("%d ", r.stats.total_flits);
        }
        std::printf("\n");

        for (auto &r : routers)
        {
            std::printf("%d ", r.stats.total_packets);
        }
        std::printf("\n");

        for (auto &r : routers)
        {
            std::printf("%d ", r.stats.sent_packets);
        }
        std::printf("\n");

        for (auto &r : routers)
        {
            std::printf("%d ", r.stats.recv_packets);
        }
        std::printf("\n");
    }
};

// Explicit-run function with named parameters exposed to Python.
// Parameters:
//   config_file: path to config file (empty means no config parsed)
//   num_threads: number of worker threads
//   max_tick: maximum ticks to run (0 means no explicit limit)
int run_sim(const std::string &config_file, uint32_t num_threads = 1,
            uint32_t max_tick = 0)
{
    // config_file is required
    if (config_file.empty())
    {
        std::fprintf(stderr, "error: config_file is required\n");
        return -1;
    }

    // check file exists
    std::ifstream fin(config_file);
    if (!fin)
    {
        std::fprintf(stderr, "error: config_file not found: %s\n", config_file.c_str());
        return -1;
    }
    fin.close();

    global_params.parse_config(config_file);

    global_params.num_workers = num_threads;

    auto top = Top{};
    top.load_test();

    top.run(max_tick);
    top.save_stats();
    return 0;
}

// Keep a normal CLI entrypoint which delegates to run_sim with explicit args
int main(int argc, char const *argv[])
{
    std::string config_file;
    uint32_t num_threads = 1;
    uint32_t max_tick = 0;

    if (argc > 1)
    {
        config_file = argv[1];
    }
    else
    {
        std::fprintf(stderr, "error: config_file is required\n");
        std::fprintf(stderr, "usage: %s <config_file> [num_threads] [max_tick]\n", argc > 0 ? argv[0] : "sim");
        return -1;
    }
    if (argc > 2)
    {
        try
        {
            num_threads = static_cast<uint32_t>(std::stoul(argv[2]));
        }
        catch (...)
        {
            num_threads = 1;
        }
    }
    if (argc > 3)
    {
        try
        {
            max_tick = static_cast<uint32_t>(std::stoul(argv[3]));
        }
        catch (...)
        {
            max_tick = 0;
        }
    }
    return run_sim(config_file, num_threads, max_tick);
}

namespace py = pybind11;

PYBIND11_MODULE(sim, m)
{
    m.doc() = "Wrapper for independent cpp_algo";

    // Expose the C++ runner to Python with explicit named parameters.
    // Example usage in Python:
    //   import sim
    //   sim.run(config_file='config.yaml', num_threads=4, max_tick=1000)
    m.def("run", &run_sim,
          py::arg("config_file"),
          py::arg("num_threads") = 1u,
          py::arg("max_tick") = 0u,
          "Run the simulator.\n\n"
          "Parameters:\n"
          "  config_file (str): path to config file\n"
          "  num_threads (int): number of worker threads\n"
          "  max_tick (int): maximum tick count (0 = no limit)");
}
