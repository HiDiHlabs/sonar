# This file provides utils function for sonar co-occurrence calculations.
#
#

import torch as t

def rescale_topographic_tensor(topographic_tensor, scaling_factor):
    """Rescales a topographic tensor by a given factor along the spatial axes [1,2].
    Args:
        topographic_tensor (torch.Tensor): Topographic tensor of shape (n_classes, x, y).
        scaling_factor (float): Scaling factor.
    Returns:
        topographic_tensor (torch.Tensor): Rescaled topographic tensor."""
    
    topographic_tensor = t.tensor(topographic_tensor, dtype=float)

    
    return t.nn.functional.interpolate(topographic_tensor.unsqueeze(0), scale_factor=scaling_factor, mode='nearest').squeeze(0)

