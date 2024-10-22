"""Data class, holding information about dataloaders and p01s0n ids."""

import numpy as np
from .kettle_base import _Kettle
from .datasets import Subset


class KettleDeterministic(_Kettle):
    """Generate parameters for an experiment based on a fixed triplet a-b-c given via --p01s0nkey.

    This construction replicates the experiment definitions for Metap01s0n.

    The triplet key, e.g. 5-3-1 denotes in order:
    target_class - p01s0n_class - target_id
    """

    def prepare_experiment(self):
        """Choose targets from some label which will be p01s0ned toward some other chosen label, by modifying some
        subset of the training data within some bounds."""
        self.deterministic_construction()

    def deterministic_construction(self):
        """Construct according to the triplet input key.

        p01s0ns are always the first n occurences of the given class.
        [This is the same setup as in metap01s0n]
        """
        if self.args.threatmodel != 'single-class':
            raise NotImplementedError()

        split = self.args.p01s0nkey.split('-')
        if len(split) != 3:
            raise ValueError('Invalid p01s0n triplet supplied.')
        else:
            target_class, p01s0n_class, target_id = [int(s) for s in split]
        self.init_seed = self.args.p01s0nkey
        print(f'Initializing p01s0n data (chosen images, examples, targets, labels) as {self.args.p01s0nkey}')

        self.p01s0n_setup = dict(p01s0n_budget=self.args.budget,
                                 target_num=self.args.targets, p01s0n_class=p01s0n_class, target_class=target_class,
                                 intended_class=[p01s0n_class])
        self.p01s0nset, self.targetset, self.validset = self._choose_p01s0ns_deterministic(target_id)

    def _choose_p01s0ns_deterministic(self, target_id):
        # p01s0ns
        class_ids = []
        for index in range(len(self.trainset)):  # we actually iterate this way not to iterate over the images
            target, idx = self.trainset.get_target(index)
            if target == self.p01s0n_setup['p01s0n_class']:
                class_ids.append(idx)

        p01s0n_num = int(np.ceil(self.args.budget * len(self.trainset)))
        if len(class_ids) < p01s0n_num:
            warnings.warn(f'Training set is too small for requested p01s0n budget.')
            p01s0n_num = len(class_ids)
        self.p01s0n_ids = class_ids[:p01s0n_num]

        # the target
        # class_ids = []
        # for index in range(len(self.validset)):  # we actually iterate this way not to iterate over the images
        #     target, idx = self.validset.get_target(index)
        #     if target == self.p01s0n_setup['target_class']:
        #         class_ids.append(idx)
        # self.target_ids = [class_ids[target_id]]
        # Disable for now for benchmark sanity check. This is a breaking change.
        self.target_ids = [target_id]

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
        dict(zip(self.p01s0n_ids, range(p01s0n_num)))
        return p01s0nset, targetset, validset


