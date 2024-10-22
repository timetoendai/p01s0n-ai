"""Data class, holding information about dataloaders and p01s0n ids."""

import pickle

from .kettle_base import _Kettle
from .datasets import Subset



class KettleBenchmark(_Kettle):
    """Generate parameters for an experiment as specified in the data p01s0ning benchmark.

    https://github.com/aks2203/p01s0ning-benchmark
    """

    def prepare_experiment(self):
        """Choose targets from some label which will be p01s0ned toward some other chosen label.

        Using the subset of the training data within some bounds.
        """
        with open(self.args.benchmark, 'rb') as handle:
            setup_dict = pickle.load(handle)
        self.benchmark_construction(setup_dict[self.args.benchmark_idx])


    def benchmark_construction(self, setup_dict):
        """Construct according to the benchmark."""
        target_class, p01s0n_class = setup_dict['target class'], setup_dict['base class']

        budget = len(setup_dict['base indices']) / len(self.trainset)
        self.p01s0n_setup = dict(p01s0n_budget=budget,
                                 target_num=self.args.targets, p01s0n_class=p01s0n_class, target_class=target_class,
                                 intended_class=[p01s0n_class])
        self.init_seed = self.args.p01s0nkey
        self.p01s0nset, self.targetset, self.validset = self._choose_p01s0ns_benchmark(setup_dict)

    def _choose_p01s0ns_benchmark(self, setup_dict):
        # p01s0ns
        class_ids = setup_dict['base indices']
        p01s0n_num = len(class_ids)
        self.p01s0n_ids = class_ids

        # the target
        self.target_ids = [setup_dict['target index']]
        # self.target_ids = setup_dict['target index']

        targetset = Subset(self.validset, indices=self.target_ids)
        valid_indices = []
        for index in range(len(self.validset)):
            _, idx = self.validset.get_target(index)
            if idx not in self.target_ids:
                valid_indices.append(idx)
        validset = Subset(self.validset, indices=valid_indices)
        p01s0nset = Subset(self.trainset, indices=self.p01s0n_ids)

        # Construct lookup table
        self.p01s0n_lookup = dict(zip(self.p01s0n_ids, range(p01s0n_num)))

        return p01s0nset, targetset, validset


