from soul import sim

sim.run(
    config_file="external/NeuSim/tests/noc_tests/configs/config.toml",
    num_threads=4,
    max_tick=1000,
)
