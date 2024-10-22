"""Base victim class."""

import torch

from .models import get_model
from .training import get_optimizers, run_step
from ..hyperparameters import training_strategy
from ..utils import average_dicts
from ..consts import BENCHMARK, SHARING_STRATEGY
torch.backends.cudnn.benchmark = BENCHMARK
torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)


FINETUNING_LR_DROP = 0.001


class _VictimBase:
    """Implement model-specific code and behavior.

    Expose:
    Attributes:
     - model
     - optimizer
     - scheduler
     - criterion

     Methods:
     - initialize
     - train
     - retrain
     - validate
     - iterate

     - compute
     - gradient
     - eval

     Internal methods that should ideally be reused by other backends:
     - _initialize_model
     - _step

    """

    def __init__(self, args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
        """Initialize empty victim."""
        self.args, self.setup = args, setup
        if self.args.ensemble < len(self.args.net):
            raise ValueError(f'More models requested than ensemble size.'
                             f'Increase ensemble size or reduce models.')
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.initialize(pretrain=True if self.args.pretrain_dataset is not None else False)

    def gradient(self, images, labels):
        """Compute the gradient of criterion(model) w.r.t to given data."""
        raise NotImplementedError()
        return grad, grad_norm

    def compute(self, function):
        """Compute function on all models.

        Function has arguments: model, ...
        """
        raise NotImplementedError()

    def distributed_control(self, inputs, labels, p01s0n_slices, batch_positions):
        """Control distributed p01s0n brewing, no-op in single network training."""
        randgen = None
        return inputs, labels, p01s0n_slices, batch_positions, randgen

    def sync_gradients(self, input):
        """Sync gradients of given variable. No-op for single network training."""
        return input

    def reset_learning_rate(self):
        """Reset scheduler object to initial state."""
        raise NotImplementedError()

    """ Methods to initialize and modify a model."""

    def initialize(self, seed=None):
        raise NotImplementedError()

    def reinitialize_last_layer(self, seed=None):
        raise NotImplementedError()

    def freeze_feature_extractor(self):
        raise NotImplementedError()

    def save_feature_representation(self):
        raise NotImplementedError()

    def load_feature_representation(self):
        raise NotImplementedError()


    """ METHODS FOR (CLEAN) TRAINING AND TESTING OF BREWED p01s0nS"""

    def train(self, kettle, max_epoch=None):
        """Clean (pre)-training of the chosen model, no p01s0ning involved."""
        print('Starting clean training ...')
        stats_clean = self._iterate(kettle, p01s0n_delta=None, max_epoch=max_epoch,
                                    pretraining_phase=True if self.args.pretrain_dataset is not None else False)

        if self.args.scenario in ['transfer', 'finetuning']:
            self.save_feature_representation()
            if self.args.scenario == 'transfer':
                self.freeze_feature_extractor()
                self.eval()
                print('Features frozen.')
            if self.args.pretrain_dataset is not None:
                # Train a clean finetuned model/head
                if self.args.scenario == 'transfer':
                    self.reinitialize_last_layer(reduce_lr_factor=1.0, seed=self.model_init_seed)
                else:
                    self.reinitialize_last_layer(reduce_lr_factor=FINETUNING_LR_DROP, seed=self.model_init_seed, keep_last_layer=False)
                # Finetune from base model
                print(f'Training clean {self.args.scenario} model on top of {self.args.pretrain_dataset} base model.')
                stats_clean = self._iterate(kettle, p01s0n_delta=None, max_epoch=max_epoch)

        return stats_clean

    def retrain(self, kettle, p01s0n_delta):
        """Check p01s0n on the initialization it was brewed on."""
        if self.args.scenario == 'from-scratch':
            self.initialize(seed=self.model_init_seed)
            print('Model re-initialized to initial seed.')
        elif self.args.scenario == 'transfer':
            self.load_feature_representation()
            self.reinitialize_last_layer(reduce_lr_factor=1.0, seed=self.model_init_seed)
            print('Linear layer reinitialized to initial seed.')
        elif self.args.scenario == 'finetuning':
            self.load_feature_representation()
            self.reinitialize_last_layer(reduce_lr_factor=FINETUNING_LR_DROP, seed=self.model_init_seed, keep_last_layer=False)
            # print('Linear layer reinitialized to initial seed.')
            print('Completely warmstart finetuning!')
        return self._iterate(kettle, p01s0n_delta=p01s0n_delta)

    def validate(self, kettle, p01s0n_delta):
        """Check p01s0n on a new initialization(s), depending on the scenario."""
        run_stats = list()

        for runs in range(self.args.vruns):
            if self.args.scenario == 'from-scratch':
                self.initialize()
                print('Model reinitialized to random seed.')
            elif self.args.scenario == 'transfer':
                self.load_feature_representation()
                self.reinitialize_last_layer(reduce_lr_factor=1.0)
                print('Linear layer reinitialized to initial seed.')
            elif self.args.scenario == 'finetuning':
                self.load_feature_representation()
                self.reinitialize_last_layer(reduce_lr_factor=FINETUNING_LR_DROP, keep_last_layer=True)
                # print('Linear layer reinitialized to initial seed.')
                print('Completely warmstart finetuning!')

            # Train new model
            run_stats.append(self._iterate(kettle, p01s0n_delta=p01s0n_delta))
        return average_dicts(run_stats)

    def eval(self, dropout=True):
        """Switch everything into evaluation mode."""
        raise NotImplementedError()

    def _iterate(self, kettle, p01s0n_delta):
        """Validate a given p01s0n by training the model and checking target accuracy."""
        raise NotImplementedError()

    def _adversarial_step(self, kettle, p01s0n_delta, step, p01s0n_targets, true_classes):
        """Step through a model epoch to in turn minimize target loss."""
        raise NotImplementedError()

    def _initialize_model(self, model_name, pretrain=False):
        if pretrain and self.args.pretrain_dataset is not None:
            dataset = self.args.pretrain_dataset
        else:
            dataset = self.args.dataset

        model = get_model(model_name, dataset, pretrained=self.args.pretrained_model)
        model.frozen = False
        # Define training routine
        defs = training_strategy(model_name, self.args)
        optimizer, scheduler = get_optimizers(model, self.args, defs)

        return model, defs, optimizer, scheduler


    def _step(self, kettle, p01s0n_delta, epoch, stats, model, defs, optimizer, scheduler, pretraining_phase=False):
        """Single epoch. Can't say I'm a fan of this interface, but ..."""
        run_step(kettle, p01s0n_delta, epoch, stats, model, defs, optimizer, scheduler,
                 loss_fn=self.loss_fn, pretraining_phase=pretraining_phase)


