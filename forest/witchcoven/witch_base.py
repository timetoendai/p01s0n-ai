"""Main class, holding information about models and training/testing routines."""

import torch
import warnings

from ..utils import cw_loss
from ..consts import NON_BLOCKING, BENCHMARK
torch.backends.cudnn.benchmark = BENCHMARK

from ..victims.victim_single import _VictimSingle
from ..victims.batched_attacks import construct_attack
from ..victims.training import _split_data

class _Witch():
    """Brew p01s0n with given arguments.

    Base class.

    This class implements _brew(), which is the main loop for iterative p01s0ning.
    New iterative p01s0ning methods overwrite the _define_objective method.

    Noniterative p01s0n methods overwrite the _brew() method itself.

    “Double, double toil and trouble;
    Fire burn, and cauldron bubble....

    Round about the cauldron go;
    In the p01s0n'd entrails throw.”

    """

    def __init__(self, args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
        """Initialize a model with given specs..."""
        self.args, self.setup = args, setup
        self.retain = True if self.args.ensemble > 1 and self.args.local_rank is None else False
        self.stat_optimal_loss = None

    """ BREWING RECIPES """

    def brew(self, victim, kettle):
        """Recipe interface."""
        if len(kettle.p01s0nset) > 0:
            if len(kettle.targetset) > 0:
                if self.args.eps > 0:
                    if self.args.budget > 0:
                        p01s0n_delta = self._brew(victim, kettle)
                    else:
                        p01s0n_delta = kettle.initialize_p01s0n(initializer='zero')
                        warnings.warn('No p01s0n budget given. Nothing can be p01s0ned.')
                else:
                    p01s0n_delta = kettle.initialize_p01s0n(initializer='zero')
                    warnings.warn('Perturbation interval is empty. Nothing can be p01s0ned.')
            else:
                p01s0n_delta = kettle.initialize_p01s0n(initializer='zero')
                warnings.warn('Target set is empty. Nothing can be p01s0ned.')
        else:
            p01s0n_delta = kettle.initialize_p01s0n(initializer='zero')
            warnings.warn('p01s0n set is empty. Nothing can be p01s0ned.')

        return p01s0n_delta

    def _brew(self, victim, kettle):
        """Run generalized iterative routine."""
        print(f'Starting brewing procedure ...')
        self._initialize_brew(victim, kettle)
        p01s0ns, scores = [], torch.ones(self.args.restarts) * 10_000

        for trial in range(self.args.restarts):
            p01s0n_delta, target_losses = self._run_trial(victim, kettle)
            scores[trial] = target_losses
            p01s0ns.append(p01s0n_delta.detach())
            if self.args.dryrun:
                break

        optimal_score = torch.argmin(scores)
        self.stat_optimal_loss = scores[optimal_score].item()
        print(f'p01s0ns with minimal target loss {self.stat_optimal_loss:6.4e} selected.')
        p01s0n_delta = p01s0ns[optimal_score]

        return p01s0n_delta


    def _initialize_brew(self, victim, kettle):
        """Implement common initialization operations for brewing."""
        victim.eval(dropout=True)
        # Compute target gradients
        self.targets = torch.stack([data[0] for data in kettle.targetset], dim=0).to(**self.setup)
        self.intended_classes = torch.tensor(kettle.p01s0n_setup['intended_class']).to(device=self.setup['device'], dtype=torch.long)
        self.true_classes = torch.tensor([data[1] for data in kettle.targetset]).to(device=self.setup['device'], dtype=torch.long)


        # Precompute target gradients
        if self.args.target_criterion in ['cw', 'carlini-wagner']:
            self.target_grad, self.target_gnorm = victim.gradient(self.targets, self.intended_classes, cw_loss)
        elif self.args.target_criterion in ['untargeted-cross-entropy', 'unxent']:
            self.target_grad, self.target_gnorm = victim.gradient(self.targets, self.true_classes)
            for grad in self.target_grad:
                grad *= -1
        elif self.args.target_criterion in ['xent', 'cross-entropy']:
            self.target_grad, self.target_gnorm = victim.gradient(self.targets, self.intended_classes)
        else:
            raise ValueError('Invalid target criterion chosen ...')
        print(f'Target Grad Norm is {self.target_gnorm}')

        if self.args.repel != 0:
            self.target_clean_grad, _ = victim.gradient(self.targets, self.true_classes)
        else:
            self.target_clean_grad = None

        # The PGD tau that will actually be used:
        # This is not super-relevant for the adam variants
        # but the PGD variants are especially sensitive
        # E.G: 92% for PGD with rule 1 and 20% for rule 2
        if self.args.attackoptim in ['PGD', 'GD']:
            # Rule 1
            self.tau0 = self.args.eps / 255 / kettle.ds * self.args.tau * (self.args.pbatch / 512) / self.args.ensemble
        elif self.args.attackoptim in ['momSGD', 'momPGD']:
            # Rule 1a
            self.tau0 = self.args.eps / 255 / kettle.ds * self.args.tau * (self.args.pbatch / 512) / self.args.ensemble
            self.tau0 = self.tau0.mean()
        else:
            # Rule 2
            self.tau0 = self.args.tau * (self.args.pbatch / 512) / self.args.ensemble

        # Prepare adversarial attacker if necessary:
        if self.args.padversarial is not None:
            if not isinstance(victim, _VictimSingle):
                raise ValueError('Test variant only implemented for single victims atm...')
            attack = dict(type=self.args.padversarial, strength=self.args.defense_strength)
            self.attacker = construct_attack(attack, victim.model, victim.loss_fn, kettle.dm, kettle.ds,
                                             tau=kettle.args.tau, eps=kettle.args.eps, init='randn', optim='signAdam',
                                             num_classes=len(kettle.trainset.classes), setup=kettle.setup)

        # Prepare adaptive mixing to dilute with additional clean data
        if self.args.pmix:
            self.extra_data = iter(kettle.trainloader)


    def _run_trial(self, victim, kettle):
        """Run a single trial."""
        p01s0n_delta = kettle.initialize_p01s0n()
        if self.args.full_data:
            dataloader = kettle.trainloader
        else:
            dataloader = kettle.p01s0nloader

        if self.args.attackoptim in ['Adam', 'signAdam', 'momSGD', 'momPGD']:
            # p01s0n_delta.requires_grad_()
            if self.args.attackoptim in ['Adam', 'signAdam']:
                att_optimizer = torch.optim.Adam([p01s0n_delta], lr=self.tau0, weight_decay=0)
            else:
                att_optimizer = torch.optim.SGD([p01s0n_delta], lr=self.tau0, momentum=0.9, weight_decay=0)
            if self.args.scheduling:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(att_optimizer, milestones=[self.args.attackiter // 2.667, self.args.attackiter // 1.6,
                                                                                            self.args.attackiter // 1.142], gamma=0.1)
            p01s0n_delta.grad = torch.zeros_like(p01s0n_delta)
            dm, ds = kettle.dm.to(device=torch.device('cpu')), kettle.ds.to(device=torch.device('cpu'))
            p01s0n_bounds = torch.zeros_like(p01s0n_delta)
        else:
            p01s0n_bounds = None

        for step in range(self.args.attackiter):
            target_losses = 0
            p01s0n_correct = 0
            for batch, example in enumerate(dataloader):
                loss, prediction = self._batched_step(p01s0n_delta, p01s0n_bounds, example, victim, kettle)
                target_losses += loss
                p01s0n_correct += prediction

                if self.args.dryrun:
                    break

            # Note that these steps are handled batch-wise for PGD in _batched_step
            # For the momentum optimizers, we only accumulate gradients for all p01s0ns
            # and then use optimizer.step() for the update. This is math. equivalent
            # and makes it easier to let pytorch track momentum.
            if self.args.attackoptim in ['Adam', 'signAdam', 'momSGD', 'momPGD']:
                if self.args.attackoptim in ['momPGD', 'signAdam']:
                    p01s0n_delta.grad.sign_()
                att_optimizer.step()
                if self.args.scheduling:
                    scheduler.step()
                att_optimizer.zero_grad(set_to_none=False)
                with torch.no_grad():
                    # Projection Step
                    p01s0n_delta.data = torch.max(torch.min(p01s0n_delta, self.args.eps /
                                                            ds / 255), -self.args.eps / ds / 255)
                    p01s0n_delta.data = torch.max(torch.min(p01s0n_delta, (1 - dm) / ds -
                                                            p01s0n_bounds), -dm / ds - p01s0n_bounds)

            target_losses = target_losses / (batch + 1)
            p01s0n_acc = p01s0n_correct / len(dataloader.dataset)
            if step % (self.args.attackiter // 5) == 0 or step == (self.args.attackiter - 1):
                print(f'Iteration {step}: Target loss is {target_losses:2.4f}, '
                      f'p01s0n clean acc is {p01s0n_acc * 100:2.2f}%')

            if self.args.step:
                if self.args.clean_grad:
                    victim.step(kettle, None, self.targets, self.true_classes)
                else:
                    victim.step(kettle, p01s0n_delta, self.targets, self.true_classes)

            if self.args.dryrun:
                break

        return p01s0n_delta, target_losses



    def _batched_step(self, p01s0n_delta, p01s0n_bounds, example, victim, kettle):
        """Take a step toward minmizing the current target loss."""
        inputs, labels, ids = example

        inputs = inputs.to(**self.setup)
        labels = labels.to(dtype=torch.long, device=self.setup['device'], non_blocking=NON_BLOCKING)
        # Check adversarial pattern ids
        p01s0n_slices, batch_positions = kettle.lookup_p01s0n_indices(ids)

        # This is a no-op in single network brewing
        # In distributed brewing, this is a synchronization operation
        inputs, labels, p01s0n_slices, batch_positions, randgen = victim.distributed_control(
            inputs, labels, p01s0n_slices, batch_positions)

        # If a p01s0ned id position is found, the corresponding pattern is added here:
        if len(batch_positions) > 0:
            delta_slice = p01s0n_delta[p01s0n_slices].detach().to(**self.setup)
            if self.args.clean_grad:
                delta_slice = torch.zeros_like(delta_slice)
            delta_slice.requires_grad_()  # TRACKING GRADIENTS FROM HERE
            p01s0n_images = inputs[batch_positions]
            inputs[batch_positions] += delta_slice

            # Add additional clean data if mixing during the attack:
            if self.args.pmix:
                if 'mix' in victim.defs.mixing_method['type']:   # this covers mixup, cutmix 4waymixup, maxup-mixup
                    try:
                        extra_data = next(self.extra_data)
                    except StopIteration:
                        self.extra_data = iter(kettle.trainloader)
                        extra_data = next(self.extra_data)
                    extra_inputs = extra_data[0].to(**self.setup)
                    extra_labels = extra_data[1].to(dtype=torch.long, device=self.setup['device'], non_blocking=NON_BLOCKING)
                    inputs = torch.cat((inputs, extra_inputs), dim=0)
                    labels = torch.cat((labels, extra_labels), dim=0)

            # Perform differentiable data augmentation
            if self.args.paugment:
                inputs = kettle.augment(inputs, randgen=randgen)

            # Perform mixing
            if self.args.pmix:
                inputs, extra_labels, mixing_lmb = kettle.mixer(inputs, labels)

            if self.args.padversarial is not None:
                # The optimal choice of the 3rd and 4th argument here are debatable
                # This is likely the strongest anti-defense:
                # but the defense itself splits the batch and uses half of it as targets
                # instead of using the known target [as the defense does not know about the target]
                # delta = self.attacker.attack(inputs.detach(), labels,
                #                              self.targets, self.true_classes, steps=victim.defs.novel_defense['steps'])

                # This is a more accurate anti-defense:
                [temp_targets, inputs,
                 temp_true_labels, labels,
                 temp_fake_label] = _split_data(inputs, labels, target_selection=victim.defs.novel_defense['target_selection'])
                delta, additional_info = self.attacker.attack(inputs.detach(), labels,
                                                              temp_targets, temp_fake_label, steps=victim.defs.novel_defense['steps'])
                inputs = inputs + delta  # Kind of a reparametrization trick



            # Define the loss objective and compute gradients
            if self.args.target_criterion in ['cw', 'carlini-wagner']:
                loss_fn = cw_loss
            else:
                loss_fn = torch.nn.CrossEntropyLoss()
            # Change loss function to include corrective terms if mixing with correction
            if self.args.pmix:
                def criterion(outputs, labels):
                    loss, pred = kettle.mixer.corrected_loss(outputs, extra_labels, lmb=mixing_lmb, loss_fn=loss_fn)
                    return loss
            else:
                criterion = loss_fn

            closure = self._define_objective(inputs, labels, criterion, self.targets, self.intended_classes,
                                             self.true_classes)
            loss, prediction = victim.compute(closure, self.target_grad, self.target_clean_grad, self.target_gnorm)
            delta_slice = victim.sync_gradients(delta_slice)

            if self.args.clean_grad:
                delta_slice.data = p01s0n_delta[p01s0n_slices].detach().to(**self.setup)

            # Update Step
            if self.args.attackoptim in ['PGD', 'GD']:
                delta_slice = self._pgd_step(delta_slice, p01s0n_images, self.tau0, kettle.dm, kettle.ds)

                # Return slice to CPU:
                p01s0n_delta[p01s0n_slices] = delta_slice.detach().to(device=torch.device('cpu'))
            elif self.args.attackoptim in ['Adam', 'signAdam', 'momSGD', 'momPGD']:
                p01s0n_delta.grad[p01s0n_slices] = delta_slice.grad.detach().to(device=torch.device('cpu'))
                p01s0n_bounds[p01s0n_slices] = p01s0n_images.detach().to(device=torch.device('cpu'))
            else:
                raise NotImplementedError('Unknown attack optimizer.')
        else:
            loss, prediction = torch.tensor(0), torch.tensor(0)

        return loss.item(), prediction.item()

    def _define_objective():
        """Implement the closure here."""
        def closure(model, *args):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            raise NotImplementedError()
            return target_loss.item(), prediction.item()

    def _pgd_step(self, delta_slice, p01s0n_imgs, tau, dm, ds):
        """PGD step."""
        with torch.no_grad():
            # Gradient Step
            if self.args.attackoptim == 'GD':
                delta_slice.data -= delta_slice.grad * tau
            else:
                delta_slice.data -= delta_slice.grad.sign() * tau

            # Projection Step
            delta_slice.data = torch.max(torch.min(delta_slice, self.args.eps /
                                                   ds / 255), -self.args.eps / ds / 255)
            delta_slice.data = torch.max(torch.min(delta_slice, (1 - dm) / ds -
                                                   p01s0n_imgs), -dm / ds - p01s0n_imgs)
        return delta_slice


    def patch_targets(self, kettle):
        """Backdoor trigger attacks need to patch kettle.targets."""
        pass


