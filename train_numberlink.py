import argparse
import math
import os
import pickle
import shutil
import time
from typing import Dict

import numpy as np
import torch
from deepxube.base.updater import UpArgs, UpHeurArgs
from deepxube.factories.updater_factory import get_updater
from deepxube.pathfinding.utils.performance import PathFindPerf
from deepxube.training.train_utils import TrainArgs
from deepxube.training.train_heur import train
from deepxube.training import trainers
from deepxube.utils.command_line_utils import (
    get_domain_from_arg,
    get_heur_nnet_par_from_arg,
    get_pathfind_from_arg,
    get_pathfind_name_kwargs,
)


def _divisors(n: int) -> set[int]:
    vals: set[int] = set()
    i = 1
    while i * i <= n:
        if n % i == 0:
            vals.add(i)
            vals.add(n // i)
        i += 1
    return vals


def _nearest_divisor(n: int, target: int) -> int:
    divs = _divisors(n)
    return min(divs, key=lambda d: (abs(d - target), d))


def _install_refined_curriculum_hooks(base_search_itrs: int) -> None:
    def update_step_probs_refined(self: trainers.Status, step_to_search_perf: Dict[int, PathFindPerf]) -> None:
        ave_solve = float(np.mean([step_to_search_perf[step].per_solved() for step in step_to_search_perf.keys()]))
        if ave_solve >= 60.0:
            self.step_max_curr = min(self.step_max_curr + 2, self.step_max)

        self.step_probs = np.zeros(self.step_max + 1)
        self.step_probs[np.arange(0, self.step_max_curr + 1)] = 1 / (self.step_max_curr + 1)

    def update_step_refined(self: trainers.TrainHeur) -> None:
        self.db.clear()
        itr_init: int = self.status.itr

        base_num_gen: int = self.train_args.batch_size * self.updater.up_args.get_up_gen_itrs()
        desired_search = int(math.ceil(1.08 * (max(1, self.status.step_max_curr) ** 1.6)))
        desired_search = max(base_search_itrs, desired_search)
        search_itrs_curr = _nearest_divisor(base_num_gen, desired_search)
        self.updater.up_args.search_itrs = search_itrs_curr
        num_gen = base_num_gen

        start_info_l = [
            f"itr: {self.status.itr}",
            f"update_num: {self.status.update_num}",
            f"targ_update: {self.status.targ_update_num}",
            f"num_gen: {format(num_gen, ',')}",
            f"search_itrs (curr): {search_itrs_curr}",
        ]
        if self.train_args.balance_steps:
            start_info_l.append(f"step max (curr): {self.status.step_max_curr}")
        print(f"\nGetting Data - {', '.join(start_info_l)}")
        times = trainers.Times()

        start_time = time.time()
        self.updater.start_update(self.status.step_probs.tolist(), num_gen, self.status.targ_update_num,
                                  self.train_args.batch_size, self.device, self.on_gpu)
        times.record_time("up_start", time.time() - start_time)

        self.train_start_time = time.time()
        if not self.updater.up_args.sync_main:
            ctgs_l = self._get_update_data(num_gen, times)
            self._end_update(itr_init, ctgs_l, times)
            loss = self._train_no_sync_main(times)
        else:
            loss, ctgs_l = self._train_sync_main(num_gen, times)
            self._end_update(itr_init, ctgs_l, times)

        start_time = time.time()
        torch.save(self.nnet.state_dict(), self.nnet_file)
        times.record_time("save", time.time() - start_time)

        update_targ = False
        if loss < self.train_args.loss_thresh:
            if self.train_args.targ_up_searches <= 0:
                update_targ = True
            else:
                raise NotImplementedError

        if update_targ:
            shutil.copy(self.nnet_file, self.nnet_targ_file)
            self.status.targ_update_num = self.status.targ_update_num + 1
        self.status.update_num += 1

        pickle.dump(self.status, open(self.status_file, "wb"), protocol=-1)
        print(f"Train - itrs: {self.updater.up_args.up_itrs}, loss: {loss:.2E}, targ_updated: {update_targ}")
        print(f"Times - {times.get_time_str()}")

    trainers.Status.update_step_probs = update_step_probs_refined
    trainers.TrainHeur.update_step = update_step_refined


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
    parser.add_argument("--up_itrs", type=int, default=100)
    parser.add_argument("--up_gen_itrs", type=int, default=None)
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

    # Refined curriculum: +2 steps at 70% solve and dynamic search_itrs tied to current curriculum step.
    if args.curriculum:
        _install_refined_curriculum_hooks(args.search_itrs)

    domain, domain_name = get_domain_from_arg(args.domain)
    heur_nnet_par = get_heur_nnet_par_from_arg(domain, domain_name, args.heur, "V")[0]
    pathfind_name, pathfind_kwargs = get_pathfind_name_kwargs("bwas.1_1.0_0.0")
    get_pathfind_from_arg(domain, "V", "bwas.1_1.0_0.0")

    up_args = UpArgs(
        args.procs,
        args.up_itrs,
        args.step_max,
        args.search_itrs,
        up_gen_itrs=args.up_gen_itrs,
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
