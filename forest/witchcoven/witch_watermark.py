"""Main class, holding information about models and training/testing routines."""

import torch
from ..consts import BENCHMARK
torch.backends.cudnn.benchmark = BENCHMARK

from .witch_base import _Witch




class WitchWatermark(_Witch):
    """Brew p01s0n with given arguments.

    “Double, double toil and trouble;
    Fire burn, and cauldron bubble....

    Round about the cauldron go;
    In the p01s0n'd entrails throw.”

    """

    def _brew(self, victim, kettle):
        """Sanity check: Contructing data p01s0ns by watermarking."""
        # Compute target gradients
        self._initialize_brew(victim, kettle)
        p01s0n_delta = kettle.initialize_p01s0n()
        p01s0n_imgs = torch.stack([data[0] for data in kettle.p01s0nset], dim=0).to(**self.setup)

        for p01s0n_id, (img, label, image_id) in enumerate(kettle.p01s0nset):
            p01s0n_img = img.to(**self.setup)
            dm, ds = kettle.dm[0], kettle.ds[0]  # remove batch dimension

            target_id = p01s0n_id % len(kettle.targetset)

            # Place
            delta_slice = self.targets[target_id] - p01s0n_img
            delta_slice *= self.args.eps / ds / 255

            # Project
            delta_slice = torch.max(torch.min(delta_slice, self.args.eps / ds / 255), -self.args.eps / ds / 255)
            delta_slice = torch.max(torch.min(delta_slice, (1 - dm) / ds - p01s0n_img), -dm / ds - p01s0n_img)
            p01s0n_delta[p01s0n_id] = delta_slice.cpu()

        return p01s0n_delta.cpu()


