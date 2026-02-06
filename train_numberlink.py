import argparse
import os

from deepxube.base.updater import UpArgs, UpHeurArgs
from deepxube.factories.updater_factory import get_updater
from deepxube.training.train_utils import TrainArgs
from deepxube.training.train_heur import train
from deepxube.utils.command_line_utils import (
    get_domain_from_arg,
    get_heur_nnet_par_from_arg,
    get_pathfind_from_arg,
    get_pathfind_name_kwargs,
)


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
    parser.add_argument(
        "--curriculum",
        action="store_true",
        default=False,
        help="Enable step-max curriculum (balances steps and increases difficulty when solved rate improves).",
    )
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

    domain, domain_name = get_domain_from_arg(args.domain)
    heur_nnet_par = get_heur_nnet_par_from_arg(domain, domain_name, args.heur, "V")[0]
    pathfind_name, pathfind_kwargs = get_pathfind_name_kwargs("bwas.1_1.0_0.0")
    get_pathfind_from_arg(domain, "V", "bwas.1_1.0_0.0")

    up_args = UpArgs(
        args.procs,
        100,
        args.step_max,
        args.search_itrs,
        up_batch_size=args.up_batch_size,
        nnet_batch_size=args.up_nnet_batch_size,
        sync_main=False,
        v=args.debug,
    )
    up_heur_args = UpHeurArgs(False, 1)
    updater = get_updater(domain, heur_nnet_par, pathfind_name, pathfind_kwargs, up_args, up_heur_args)

    train_args = TrainArgs(args.batch_size, args.lr, args.lr_d, args.max_itrs, args.curriculum, display=0)
    train(heur_nnet_par, "V", updater, args.out_dir, train_args, test_args=None, debug=args.debug)


if __name__ == "__main__":
    main()
