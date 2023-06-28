"""Sonar module. Generates co-occurrence curves from topographic maps."""	

import torch
from sonar.utils import PointProcess

if torch.cuda.is_available():
    import torch as t
    device = 'cuda:0'
else:
    import torch as t
    device='cpu'

import numpy as np
from scipy import fft as sp_fft

import tqdm

def _get_kernels(max_radius, linear_steps, normalize=True):
    """Generates a range of circular kernels used to calculate co-occurrence curves.
    
    Args:
        max_radius (int): Maximum radius of the kernels.
        linear_steps (int): Number of linear steps in the kernels.
        
    Returns:
        kernels (np.array): Array of kernels.
        vals (np.array): Array of radii corresponding to the kernels.
    """

    if linear_steps > max_radius:
        linear_steps = max_radius

    vals = np.ones((max_radius)+1)
    vals[0] = 0
    vals[linear_steps+1:] += np.arange(max_radius-linear_steps)
    vals = np.cumsum(vals)
    vals = vals[:(vals < max_radius).sum()+1]
    vals[-1] = max_radius

    span = np.arange(-max_radius, max_radius + 1)
    X, Y = np.meshgrid(span, span)
    dists = (X**2+Y**2)**0.5

    kernel_1 = np.zeros_like(X)
    kernel_1[max_radius, max_radius] = 1
    kernels = [kernel_1]

    for i in range(len(vals)-1):

        r1 = vals[i]
        r2 = vals[i+1]

        kernel_1 = (dists-r1)
        kernel_1 = -(kernel_1-1)*(kernel_1 < 1)
        kernel_1[kernel_1 > 1] = 1
        kernel_1 = 1-kernel_1

        kernel_2 = (dists-r2)
        kernel_2 = -(kernel_2-1)*(kernel_2 < 1)
        kernel_2[kernel_2 > 1] = 1
        kernel_2[kernel_2 == 1] = kernel_1[kernel_2 == 1]

        kernels.append(kernel_2)

    kernels = np.array(kernels,dtype=float)

    if normalize:
        kernels = kernels/kernels.sum(axis=(1,2))[:,None,None]

    return (kernels,vals)


def _interpolate(x,y,step):

    from scipy import interpolate 
    """_interpolat generates linear interpolation from the x,y to create mutual distance curves in um units
    :param x: distances with available values
    :type x: np.array
    :param y: values at x
    :type y: np array
    :param step: step size of x,y
    :type step: float
    """

    # return lambda x_ :  interpolate.BSpline(*interpolate.splrep(x,y,s=0,k=5), extrapolate=False)

    return np.apply_along_axis(lambda x_ :  interpolate.BSpline(*interpolate.splrep(x,x_,s=0,k=5), extrapolate=False)(np.arange(0,x.max()*step)),2,y)

class Sonar():
    """Generates co-occurrence curves from topographic maps.
    
    Args:
        max_radius (int): Maximum radius of the kernels.
        linear_steps (int): Number of linear steps in the kernels.
        n_bins (int): Number of bins in the co-occurrence curves.
        min_val (float): Minimum value of the co-occurrence curves.
        max_val (float): Maximum value of the co-occurrence curves.
        normalize (bool): Whether to normalize the co-occurrence curves.
    """
    
    def __init__(self, max_radius=20, linear_steps=20, normalize=True, device=device,edge_correction=False):
        
        self.max_radius = max_radius
        self.linear_steps = linear_steps
        self.normalize = normalize
        self.device = device
        self.kernels, self.vals = _get_kernels(max_radius, linear_steps,normalize=True)
        self.kernels = t.tensor(self.kernels,dtype=torch.float32,device=device)
        self.edge_correction = edge_correction
        
    def co_occurrence_from_map(self, topographic_map):
        """Calculates co-occurrence curves for a topographic map.
        
        Args:
            topographic_map (nd-iterable): Topographic map.
            
        Returns:
            co_occurrence (nd-iterable): Co-occurrence curves.
        """
        
        topographic_tensor = np.zeros((topographic_map.max()+1,)+topographic_map.shape,dtype=bool)

        print(topographic_tensor.shape)
        X,Y = np.meshgrid(np.arange(topographic_tensor.shape[1]),np.arange(topographic_tensor.shape[2]))
        topographic_tensor[topographic_map.T,X,Y]=True            

        return self.co_occurrence_from_tensor(topographic_tensor)
    
    def co_occurrence_from_tensor(self, hists, interpolate=True,  progbar=False, normalize=True):
        """Calculates co-occurrence curves for a topographic tensor.
        
        Args:
            topographic_tensor (nd-iterable): Topographic tensor.
            
        Returns:
            co_occurrence (nd-iterable): Co-occurrence curves.
        """

        if not (type(hists) is PointProcess):
            hists = t.tensor(hists,dtype=torch.float32,device=self.device)

        # Determine dimensions of the input/output/intermediate variables
        n_classes = hists.shape[0]      
        kernels,radii = self.kernels,self.vals
        map_size = hists.shape[1:]
        kernel_size = self.kernels.shape[1:]
            

        co_occurrences = np.empty((n_classes, n_classes,self.kernels.shape[0]))

        shape = [kernel_size[i]+map_size[i] for i in range(2)]
        fshape = [sp_fft.next_fast_len(shape[a], True) for a in [0,1]]

        kernels_fft = (t.fft.rfftn(kernels.float(), fshape,dim=[1,2]))

        width_kernel=kernels[0].shape[0]
        
        if self.edge_correction:
            bg_mask = hists.sum(dim=0)
            bg_fft = t.fft.rfftn(bg_mask.float(), fshape,dim=[0,1])
            bg_fftprod =  (bg_fft*kernels_fft)

            bg_conv = t.fft.irfftn(bg_fftprod,fshape,dim=[1,2]).float()
            bg_conv =  bg_conv[:,width_kernel//2:width_kernel//2+hists[0].shape[0],
                            width_kernel//2:width_kernel//2+hists[0].shape[1]] 
            bg_conv[bg_conv<=0]=1

        total_computations = (n_classes**2+n_classes)/2
        n_computations = 0


        if progbar:
            pbar = tqdm.tqdm(total=total_computations)

        # with tqdm.tqdm(total=total_computations, disable=~progbar) as _:
        for i in range(n_classes):
            # print(i)
            h1_fft = t.fft.rfftn(hists[i].float(), fshape,dim=[0,1])
            h1_fftprod =  (h1_fft*kernels_fft)

            h1_conv = t.fft.irfftn(h1_fftprod,fshape,dim=[1,2]).float()
            h1_conv =  h1_conv[:,width_kernel//2:width_kernel//2+hists[0].shape[0],
                            width_kernel//2:width_kernel//2+hists[0].shape[1]] #signal._signaltools._centered(h1_conv,[len(kernels)]+fshape).copy()

            if self.edge_correction:
                h1_conv = h1_conv/bg_conv
                # h1_conv[h1_conv<0]=0

            h1_product=h1_conv*hists[i]#/np.sum(kernels,axis=(1,2))[:,None,None]
            co_occurrences[i,i]=h1_product.sum(dim=(1,2)).cpu()

            for j in range(i+1,n_classes):
                h2_product=h1_conv*hists[j]#/np.sum(kernels,axis=(1,2))[:,None,None]
                co_occurrences[i,j] = h2_product.sum(dim=(1,2)).cpu()
                co_occurrences[j,i]= co_occurrences[i,j]
            n_computations += n_classes-i-1

            if progbar:
                pbar.update(n_classes-i)

        if interpolate: 
            co_occurrences = _interpolate(radii, co_occurrences, 1)
            if normalize:
                co_occurrences = co_occurrences/(co_occurrences[:,:,0].diagonal()[:,None,None])   
            return co_occurrences
        
        else:
            if normalize:
                co_occurrences = co_occurrences/(co_occurrences[:,:,0].diagonal()[:,None,None])
            return radii, co_occurrences
