import argparse
import subprocess
from pathlib import Path

import numpy as np
import tomli_w
import yaml

parser = argparse.ArgumentParser()

parser.add_argument("-y", type=int, default=4)
parser.add_argument("-x", type=int, default=4)
parser.add_argument("-j", type=int, default=1)
parser.add_argument("--max-packet", type=int, default=2)
parser.add_argument("--packet-size", type=int, default=2)
parser.add_argument("--vc", type=int, default=4)
parser.add_argument("--buffer-size", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)

parser.add_argument("--gen", action="store_true")
parser.add_argument("--test", action="store_true")
parser.add_argument("--compare", action="store_true")

args = parser.parse_args()

np.random.seed(args.seed)

neusim_home = Path(__file__).parent.parent.parent
neusim_exe = neusim_home / "build" / "cmake-build-debug" / "main"

noxim_home = neusim_home.parent / "noxim"
noxim_exe = noxim_home / "bin" / "noxim"
noxim_power_file = noxim_home / "bin" / "power.yaml"
config_dir = Path(__file__).parent / "configs"
noxim_config_template = config_dir / "template.yaml"
config = yaml.safe_load(open(noxim_config_template, "r"))

y, x = args.y, args.x
num_cores = y * x
config["mesh_dim_y"] = y
config["mesh_dim_x"] = x
config["buffer_depth"] = args.buffer_size
config["max_packet_size"] = args.packet_size
config["n_virtual_channels"] = args.vc

config_dir.mkdir(parents=True, exist_ok=True)
noxim_config_file = config_dir / "config.yaml"
neusim_config_file = config_dir / "config.toml"
neusim_power_file = config_dir / "power.toml"
if args.gen:
    traffic_table_file = config_dir / "traffic_table.txt"
    config["traffic_table_filename"] = traffic_table_file.relative_to(
        neusim_config_file, walk_up=True
    ).as_posix()
    traffic_table = np.random.randint(0, args.max_packet, size=(num_cores, num_cores))
    conns = []
    for i in range(num_cores):
        for j in range(num_cores):
            if i != j and traffic_table[i, j] > 0:
                conns.append((i, j, traffic_table[i, j]))
    s = f"{num_cores} {len(conns)}"
    traffic_table_file.write_text(
        s + "\n" + "\n".join([f"{i} {j} {traffic}" for i, j, traffic in conns])
    )

    neusim_config_file.write_text(tomli_w.dumps(config))
    config["traffic_table_filename"] = traffic_table_file.as_posix()
    yaml.safe_dump(config, open(noxim_config_file, "w"))


def read_noxim_res_one(file):
    file = Path(file)
    ss = file.read_text().split("\n")

    latency = int(float(ss[8].split("(")[1].split(" ")[0]))
    num_packets = int(ss[10].split(":")[1].strip())
    num_flits = int(ss[11].split(":")[1].strip())
    avg_delay = float(ss[14].split(":")[1].strip())
    max_delay = int(float(ss[15].split(":")[1].strip()))
    throughput = float(ss[16].split(":")[1].strip())
    energy = float(ss[18].split(":")[1].strip())
    num_flits_core = [int(v) for v in ss[-5].strip().split(" ")]
    num_packets_core = [int(v) for v in ss[-4].strip().split(" ")]
    send_packets_core = [int(v) for v in ss[-3].strip().split(" ")]
    recv_packets_core = [int(v) for v in ss[-2].strip().split(" ")]

    return {
        "latency": latency,
        "num_packets": num_packets,
        "num_flits": num_flits,
        "avg_delay": avg_delay,
        "max_delay": max_delay,
        "throughput": throughput,
        "energy": energy,
        "num_flits_core": num_flits_core,
        "num_packets_core": num_packets_core,
        "send_packets_core": send_packets_core,
        "recv_packets_core": recv_packets_core,
    }


def read_neusim_res_one(file):
    file = Path(file)
    ss = file.read_text().split("\n")

    num_packets = int(ss[1].split(" ")[1].strip())
    latency = int(ss[4].split(":")[1])
    num_packets = int(ss[5].split(":")[1].strip())
    num_flits = int(ss[6].split(":")[1].strip())
    avg_delay = float(ss[7].split(":")[1].strip())
    max_delay = int(float(ss[8].split(":")[1].strip()))
    throughput = float(ss[9].split(":")[1].strip())
    # energy = float(ss[18].split(":")[1].strip())
    num_flits_core = [int(v) for v in ss[-5].strip().split(" ")]
    num_packets_core = [int(v) for v in ss[-4].strip().split(" ")]
    send_packets_core = [int(v) for v in ss[-3].strip().split(" ")]
    recv_packets_core = [int(v) for v in ss[-2].strip().split(" ")]

    return {
        "latency": latency,
        "num_packets": num_packets,
        "num_flits": num_flits,
        "avg_delay": avg_delay,
        "max_delay": max_delay,
        "throughput": throughput,
        # "energy": energy,
        "num_flits_core": num_flits_core,
        "num_packets_core": num_packets_core,
        "send_packets_core": send_packets_core,
        "recv_packets_core": recv_packets_core,
    }


if args.test:
    neusim_res_file = config_dir / "neusim_res.txt"
    cmd = [
        neusim_exe.as_posix(),
        neusim_config_file.as_posix(),
        f"{args.j}",
        neusim_power_file.as_posix(),
    ]
    print(" ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    neusim_res_file.write_text(result.stdout)

    neusim_res = read_neusim_res_one(neusim_res_file)
    num_flits = neusim_res["num_flits"]
    num_packets = neusim_res["num_packets"]
    num_flits_core = neusim_res["num_flits_core"]
    num_packets_core = neusim_res["num_packets_core"]
    send_packets_core = neusim_res["send_packets_core"]
    recv_packets_core = neusim_res["recv_packets_core"]

    assert num_flits == num_packets * config["max_packet_size"]
    assert num_packets == np.sum(send_packets_core)
    assert num_packets == np.sum(recv_packets_core)

    if args.compare:
        noxim_res_file = config_dir / "noxim_res.txt"
        cmd = [
            noxim_exe.as_posix(),
            "-config",
            noxim_config_file.as_posix(),
            "-power",
            noxim_power_file.as_posix(),
        ]

        print(" ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        noxim_res_file.write_text(result.stdout)

        noxim_res = read_noxim_res_one(noxim_res_file)
        num_flits2 = noxim_res["num_flits"]
        num_packets2 = noxim_res["num_packets"]
        num_flits_core2 = noxim_res["num_flits_core"]
        num_packets_core2 = noxim_res["num_packets_core"]
        send_packets_core2 = noxim_res["send_packets_core"]
        recv_packets_core2 = noxim_res["recv_packets_core"]

        assert num_flits2 == num_packets2 * config["max_packet_size"]
        assert num_packets2 == np.sum(send_packets_core2)
        assert num_packets2 == np.sum(recv_packets_core2)

        assert np.all(num_flits_core == num_flits_core2)
        assert np.all(num_packets_core == num_packets_core2)
        assert np.all(send_packets_core == send_packets_core2)
        assert np.all(recv_packets_core == recv_packets_core2)
        print("Results match!")
