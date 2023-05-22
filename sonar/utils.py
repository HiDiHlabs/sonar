# This file provides utils function for sonar co-occurrence calculations.
#
#

import torch as t
import numpy as np

def rescale_topographic_tensor(topographic_tensor, scaling_factor):
    """Rescales a topographic tensor by a given factor along the spatial axes [1,2].
    Args:
        topographic_tensor (torch.Tensor): Topographic tensor of shape (n_classes, x, y).
        scaling_factor (float): Scaling factor.
    Returns:
        topographic_tensor (torch.Tensor): Rescaled topographic tensor."""
    
    topographic_tensor = t.tensor(topographic_tensor, dtype=float)
    
    return t.nn.functional.interpolate(topographic_tensor.unsqueeze(0), scale_factor=scaling_factor, mode='nearest').squeeze(0)


class PointProcess():
    """
    Class for storing point process data. The can be rasterized greedily when needed for co-occurrence calculations.
    :param x: x-coordinates of the point process.
    :param y: y-coordinates of the point process.
    :param class_id: class id of the point process.
    :param rescale_factor: rescale factor for the point process.
    :param device: torch device indicator.
    """

    def __init__(self,x,y,class_id=None,rescale_factor=1.0,device='cpu'):


        if class_id is None:
            class_id = np.zeros((len(x),)).astype(int)

        self.x = np.array(x*rescale_factor).astype(int)
        self.y = np.array(y*rescale_factor).astype(int)
        self.class_id = np.array(class_id).astype(int)
        
        self.device = device

        self.shape = (int(self.class_id.max()+1), int(self.x.max()+1), int(self.y.max()+1))

        self.bins = (np.arange(self.shape[1]+1), np.arange(self.shape[2]+1))

    def __len__(self):

        return len(self.class_id)
    
    def _generate_histogram(self, class_id):

        # histogram = np.zeros(self.shape[1:], dtype=int)

        class_mask = self.class_id == class_id

        histogram = np.histogram2d(self.x[class_mask], self.y[class_mask], bins=self.bins)[0]

        return t.tensor(histogram,device=self.device, dtype=t.float)

    def __getitem__(self, index):
        
        if isinstance(index, int):
            return self._generate_histogram(index)
        