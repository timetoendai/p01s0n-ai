"""Interface for p01s0n recipes."""
from .witch_matching import WitchGradientMatching, WitchGradientMatchingNoisy, WitchGradientMatchingHidden, WitchMatchingMultiTarget
from .witch_metap01s0n import WitchMetap01s0n, WitchMetap01s0nHigher, WitchMetap01s0n_v3
from .witch_watermark import WitchWatermark
from .witch_p01s0n_frogs import WitchFrogs
from .witch_bullseye import WitchBullsEye
from .witch_patch import WitchPatch
from .witch_htbd import WitchHTBD
from .witch_convex_polytope import WitchConvexPolytope

import torch


def Witch(args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
    """Implement Main interface."""
    if args.recipe == 'gradient-matching':
        return WitchGradientMatching(args, setup)
    elif args.recipe == 'gradient-matching-private':
        return WitchGradientMatchingNoisy(args, setup)
    elif args.recipe == 'gradient-matching-hidden':
        return WitchGradientMatchingHidden(args, setup)
    elif args.recipe == 'gradient-matching-mt':
        return WitchMatchingMultiTarget(args, setup)
    elif args.recipe == 'watermark':
        return WitchWatermark(args, setup)
    elif args.recipe == 'patch':
        return WitchPatch(args, setup)
    elif args.recipe == 'hidden-trigger':
        return WitchHTBD(args, setup)
    elif args.recipe == 'metap01s0n':
        return WitchMetap01s0n(args, setup)
    elif args.recipe == 'metap01s0n-v2':
        return WitchMetap01s0nHigher(args, setup)
    elif args.recipe == 'metap01s0n-v3':
        return WitchMetap01s0n_v3(args, setup)
    elif args.recipe == 'p01s0n-frogs':
        return WitchFrogs(args, setup)
    elif args.recipe == 'bullseye':
        return WitchBullsEye(args, setup)
    elif args.recipe == 'convex-polytope':
        return WitchConvexPolytope(args, setup)
    else:
        raise NotImplementedError()


__all__ = ['Witch']


