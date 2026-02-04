import argparse
import os
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain",
        default="numberlink.4x4x3_random_walk",
        help="Domain string, e.g. numberlink.4x4x3_random_walk",
    )
    parser.add_argument("--out_dir", default="runs/numberlink_4x4_smoke")
    parser.add_argument("--heur", default="resnet_fc.256H_2B_bn")
    parser.add_argument("--procs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--up_batch_size", type=int, default=32)
    parser.add_argument("--up_nnet_batch_size", type=int, default=1024)
    parser.add_argument("--search_itrs", type=int, default=10)
    parser.add_argument("--step_max", type=int, default=5)
    parser.add_argument("--max_itrs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_d", type=float, default=0.999)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument(
        "--no_shm",
        action="store_true",
        default=False,
        help="Disable shared memory (sets DEEPXUBE_NO_SHM=1 for the subprocess).",
    )
    args = parser.parse_args()

    if args.no_shm:
        os.environ["DEEPXUBE_NO_SHM"] = "1"

    cmd = [
        sys.executable,
        "run_deepxube_cli.py",
        "train",
        "--domain",
        args.domain,
        "--heur",
        args.heur,
        "--heur_type",
        "V",
        "--pathfind",
        "bwas.1_1.0_0.0",
        "--dir",
        args.out_dir,
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--lr_d",
        str(args.lr_d),
        "--max_itrs",
        str(args.max_itrs),
        "--procs",
        str(args.procs),
        "--step_max",
        str(args.step_max),
        "--search_itrs",
        str(args.search_itrs),
        "--up_batch_size",
        str(args.up_batch_size),
        "--up_nnet_batch_size",
        str(args.up_nnet_batch_size),
    ]
    if args.debug:
        cmd.append("--debug")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
