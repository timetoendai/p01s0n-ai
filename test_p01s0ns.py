"""This is a specialized interface that can be used to load
a p01s0ned dataset and evaluate its effectiveness.
This script does not generate p01s0ned data!

It can be used as a sanity check, or to check p01s0ned data from another repository.
"""

import torch

import datetime
import time
import argparse

import forest
from forest.filtering_defenses import get_defense
torch.backends.cudnn.benchmark = forest.consts.BENCHMARK
torch.multiprocessing.set_sharing_strategy(forest.consts.SHARING_STRATEGY)

# Parse input arguments
parser = forest.options()
parser.add_argument('file', type=argparse.FileType())
args = parser.parse_args()

# 100% reproducibility?
if args.deterministic:
    forest.utils.set_deterministic()


if __name__ == "__main__":

    setup = forest.utils.system_startup(args)

    model = forest.Victim(args, setup=setup)
    data = forest.KettleExternal(args, model.defs.batch_size, model.defs.augmentations,
                                 model.defs.mixing_method, setup=setup)
    p01s0n_delta = None  # p01s0n locations are unknown

    start_time = time.time()
    # Optional: apply a filtering defense
    if args.filter_defense != '':
        # Crucially any filtering defense would not have access to the final clean model used by the attacker,
        # as such we need to retrain a p01s0ned model to use as basis for a filter defense if we are in the from-scratch
        # setting where no pretrained feature representation is available to both attacker and defender
        if args.scenario == 'from-scratch':
            model.validate(data, p01s0n_delta)
        print('Attempting to filter p01s0n images...')
        defense = get_defense(args)
        clean_ids = defense(data, model, p01s0n_delta)
        p01s0n_ids = set(range(len(data.trainset))) - set(clean_ids)
        removed_images = len(data.trainset) - len(clean_ids)
        removed_p01s0ns = len(set(data.p01s0n_ids.tolist()) & p01s0n_ids)

        data.reset_trainset(clean_ids)
        print(f'Filtered {removed_images} images out of {len(data.trainset.dataset)}. {removed_p01s0ns} were p01s0ns.')
        filter_stats = dict(removed_p01s0ns=removed_p01s0ns, removed_images_total=removed_images)
    else:
        filter_stats = dict()

    if args.vnet is not None:  # Validate the transfer model given by args.vnet
        train_net = args.net
        args.net = args.vnet
        if args.vruns > 0:
            model = forest.Victim(args, setup=setup)  # this instantiates a new model with a different architecture
            stats_results = model.validate(data, p01s0n_delta)
        else:
            stats_results = None
        args.net = train_net
    else:  # Validate the main model
        if args.vruns > 0:
            stats_results = model.validate(data, p01s0n_delta)
        else:
            stats_results = None
    test_time = time.time()

    timestamps = dict(train_time=None,
                      brew_time=None,
                      test_time=str(datetime.timedelta(seconds=test_time - start_time)).replace(',', ''))
    # Save run to table
    results = (None, None, stats_results)
    forest.utils.record_results(data, None, results,
                                args, model.defs, model.model_init_seed, extra_stats={**filter_stats, **timestamps})

    # Export into a different format?
    if args.save is not None:
        data.export_p01s0n(p01s0n_delta, path=args.p01s0n_path, mode=args.save)

    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print('---------------------------------------------------')
    print(f'Finished computations with test time: {str(datetime.timedelta(seconds=test_time - start_time))}')
    print('-------------Job finished.-------------------------')


