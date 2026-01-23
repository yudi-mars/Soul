#include "neu-sim/params.hpp"
#include "toml.hpp"

Params global_params;

auto Params::parse_config(std::string_view file) -> void {
  // print();
  config_file = std::filesystem::path(file);
  auto config = toml::parse_file(config_file.string());
  // printf("Parsing config file %s\n", config_file.data());

  topo_size = { config["mesh_dim_y"].value<uint32_t>().value(),
                config["mesh_dim_x"].value<uint32_t>().value() };
  routing = config["routing_algorithm"].value<std::string>().value();
  n_ch = config["n_virtual_channels"].value<uint32_t>().value();
  buf_size = config["buffer_depth"].value<uint32_t>().value();
  packet_size = config["max_packet_size"].value<uint32_t>().value();
  tick_noc = config["tick_noc"].value<uint32_t>().value_or(1);
  tick_core = config["tick_core"].value<uint32_t>().value_or(1);

  traffic_table_file =
    (config_file /
     config["traffic_table_filename"].value<std::string>().value())
      .lexically_normal();
  // print();
}

auto Params::print() -> void {
  printf("num_workers: %d\n", num_workers);

  printf("topology: %s\n", topology.data());
  printf("topology size: (");
  for (int i = 0; i < topo_size.size(); ++i) {
    if (i == topo_size.size() - 1)
      printf("%d", topo_size[i]);
    else
      printf("%d, ", topo_size[i]);
  }
  printf(")\n");

  printf("routing: %s\n", routing.data());
  printf("vc num: %d\n", n_ch);
  printf("buffer size: %d\n", buf_size);
  printf("packet size: %d\n", packet_size);
  printf("traffic table: %s\n", traffic_table_file.c_str());
}
