import argparse
import subprocess
import sys


def build_args(profile: str) -> dict:
    if profile == "local":
        return {
            "procs": 8,
            "batch_size": 1024,
            "up_batch_size": 200,
            "up_nnet_batch_size": 20000,
            "search_itrs": 200,
            "step_max": 20,
            "max_itrs": 20000,
            "heur": "resnet_fc.512H_4B_bn",
        }
    if profile == "colab":
        return {
            "procs": 12,
            "batch_size": 4096,
            "up_batch_size": 400,
            "up_nnet_batch_size": 100000,
            "search_itrs": 400,
            "step_max": 30,
            "max_itrs": 50000,
            "heur": "resnet_fc.1024H_6B_bn",
        }
    raise ValueError(f"Unknown profile: {profile}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["local", "colab"], default="local")
    parser.add_argument(
        "--domain",
        default="numberlink.8x8x7_random_walk",
        help="Domain string, e.g. numberlink.8x8x7_random_walk",
    )
    parser.add_argument("--out_dir", default="runs/numberlink_8x8x7")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_d", type=float, default=0.9999993)
    args = parser.parse_args()

    cfg = build_args(args.profile)

    cmd = [
        sys.executable,
        "-m",
        "deepxube._cli",
        "train",
        "--domain",
        args.domain,
        "--heur",
        cfg["heur"],
        "--heur_type",
        "V",
        "--pathfind",
        "bwas.1_1.0_0.0",
        "--dir",
        args.out_dir,
        "--batch_size",
        str(cfg["batch_size"]),
        "--lr",
        str(args.lr),
        "--lr_d",
        str(args.lr_d),
        "--max_itrs",
        str(cfg["max_itrs"]),
        "--procs",
        str(cfg["procs"]),
        "--step_max",
        str(cfg["step_max"]),
        "--search_itrs",
        str(cfg["search_itrs"]),
        "--up_batch_size",
        str(cfg["up_batch_size"]),
        "--up_nnet_batch_size",
        str(cfg["up_nnet_batch_size"]),
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
