### This is a script for generating a topographic tensor from a given set of co-occurrence curves.

import numpy as np
import torch as t
from scipy import fft as sp_fft
import tqdm

class Generator():
    """Generator class that provides methods for generating topographic tensors from co-occurrence curves."""

    def __init__(self, sonar, shape=(400,400),border =100 ) -> None:
        self.sonar = sonar
        self.kernels = sonar.kernels
        self.device=sonar.device
        self.border=border
        self.shape = shape
        
        shape = [self.kernels.shape[i]+shape[i] for i in range(2)]
        self.fshape = [sp_fft.next_fast_len(shape[a], True) for a in [0,1]]


    def determine_loss(self, generative_map, template_co_occurrence):
        """Calculates the loss between a generative map's co-occurrence and a template co-occurrence curve.
        
        Args:
            generative_map (torch.Tensor): Generative map.
            template_co_occurrence (torch.Tensor): Template co-occurrence curve.
        
        Returns:
            loss (torch.Tensor): Loss between the generative map's co-occurrence and the template co-occurrence curve.
            """

        kernel_width = self.kernels.shape[1]
        
        d_generative_map = t.zeros((generative_map.shape[0],self.kernels.shape[0],self.kernels.shape[0]), device=self.device)

        kernels_fft = (t.fft.rfftn(self.kernels, self.fshape,dim=[1,2]))
        
        for i in range(generative_map.shape[0]):
            # print(i)
            h1_fft = t.fft.rfftn(generative_map[i], self.fshape,dim=[0,1])
            h1_fftprod =  (h1_fft*kernels_fft)
            h1_conv = t.fft.irfftn(h1_fftprod,self.fshape,dim=[1,2])
            h1_conv_ =  h1_conv[:,kernel_width//2:kernel_width//2+generative_map.shape[1],
                            kernel_width//2:kernel_width//2+generative_map.shape[2]] #signal._signaltools._centered(h1_conv,[len(kernels)]+fshape).copy()

            h1_product=h1_conv_*generative_map[i]#/np.sum(kernels,axis=(1,2))[:,None,None]
            co_occurrence=h1_product.sum(dim=(1,2))

            d_cooc = template_co_occurrence[i,i]-co_occurrence

            d_generative_map[i] += (h1_product*d_cooc[:,None,None])[:,self.border:-self.border,self.border:-self.border].mean(0)

            for j in range(i+1,generative_map.shape[0]):
                h2_product=h1_conv_*generative_map[j]#/np.sum(kernels,axis=(1,2))[:,None,None]
                co_occurrence = h2_product.sum(dim=(1,2))
                d_cooc = template_co_occurrence[i,j]-co_occurrence
                d_update = (h2_product*d_cooc[:,None,None])[:,self.border:-self.border,self.border:-self.border].mean(0)
                d_generative_map[i] += d_update
                d_generative_map[j] += d_update

        return d_generative_map
    
    def update_step(self):
        pass

    def generate(self, template_co_occurrence, iterations=100, lr=5, momentum=0.99, verbose=True):
        """Generates a topographic tensor from a template co-occurrence curve.
        
        Args:
            template_co_occurrence (torch.Tensor): Template co-occurrence curve.
            iterations (int, optional): Number of iterations to run the generator. Defaults to 100.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            verbose (bool, optional): Whether to print the loss at each iteration. Defaults to True.
        
        Returns:
            generative_map (torch.Tensor): Generated topographic tensor.
            """

        counts = template_co_occurrence[:,:,0].diagonal()
        generative_map = t.rand((template_co_occurrence.shape[0],self.shape[0],self.shape[1]), device=self.device)
        generative_map *= counts[:,None,None]/generative_map.sum(dim=(1,2))[:,None,None]

        for i in tqdm.tqdm(range(iterations)):
            
            temperature = i/iterations

            generative_map_old = generative_map.clone()
            d_generative_map = self.determine_loss(generative_map, template_co_occurrence)

            # update generative map
            generative_map+=lr*d_generative_map

            # turn into a proper probability distribution, with a temperature parameter 
            # determining the 
            generative_map = t.nn.functional.softmax(generative_map*(0.0001+temperature*3),dim=0)
            generative_map = generative_map*(1-momentum)+generative_map_old*momentum

        return generative_map
