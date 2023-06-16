### This is a script for generating a topographic tensor from a given set of co-occurrence curves.

import numpy as np
import torch as t
from scipy import fft as sp_fft
import tqdm

class Generator():
    """Generator class that provides methods for generating topographic tensors from co-occurrence curves."""

    def __init__(self, sonar, shape=(500,500) ) -> None:
        self.sonar = sonar
        self.kernels = sonar.kernels
        self.device=sonar.device
        self.shape = shape
        
        shape = [self.kernels.shape[i+1]+shape[i] for i in range(2)]
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
        
        d_generative_map = t.zeros((generative_map.shape[0],self.shape[0],self.shape[1]), dtype=t.float32, device=self.device)

        kernels_fft = (t.fft.rfftn(self.kernels.float(), self.fshape,dim=[1,2]))
        
        d_auto_coocs=[]

        for i in range(generative_map.shape[0]):
            # print(i)
            h1_fft = t.fft.rfftn(generative_map[i].float(), self.fshape,dim=[0,1])
            h1_fftprod =  (h1_fft*kernels_fft)
            h1_conv = t.fft.irfftn(h1_fftprod,self.fshape,dim=[1,2]).float()
            h1_conv_ =  h1_conv[:,kernel_width//2:kernel_width//2+generative_map.shape[1],
                            kernel_width//2:kernel_width//2+generative_map.shape[2]] #signal._signaltools._centered(h1_conv,[len(kernels)]+fshape).copy()

            h1_product=h1_conv_*generative_map[i]#/np.sum(kernels,axis=(1,2))[:,None,None]
            co_occurrence=h1_product.sum(dim=(1,2))

            d_cooc = template_co_occurrence[i,i]-co_occurrence

            d_auto_coocs.append(d_cooc.cpu().numpy())

            d_generative_map[i] += (h1_product*d_cooc[:,None,None]).mean(0)

            for j in range(i+1,generative_map.shape[0]):
                h2_product=h1_conv_*generative_map[j]#/np.sum(kernels,axis=(1,2))[:,None,None]
                co_occurrence = h2_product.sum(dim=(1,2))
                d_cooc = template_co_occurrence[i,j]-co_occurrence
                d_update = (h2_product*d_cooc[:,None,None]).mean(0)
                d_generative_map[i] += d_update
                d_generative_map[j] += d_update

        return d_generative_map,np.array(d_auto_coocs)
    
    def update_step(self):
        pass

    def generate(self, template_co_occurrence, iterations=100, lr=5, momentum=0.99, verbose=True, render_args=None):
        """Generates a topographic tensor from a template co-occurrence curve.
        
        Args:
            template_co_occurrence (torch.Tensor): Template co-occurrence curve.
            iterations (int, optional): Number of iterations to run the generator. Defaults to 100.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            verbose (bool, optional): Whether to print the loss at each iteration. Defaults to True.
        
        Returns:
            generative_map (torch.Tensor): Generated topographic tensor.
            """

        if render_args is not None:
            import matplotlib.pyplot as plt
            import os
            os.makedirs(render_args['save_dir'], exist_ok=True)


        template_co_occurrence = t.tensor(template_co_occurrence, dtype=t.float32, device=self.device)
        counts = template_co_occurrence[:,:,0].diagonal()
        generative_map = t.rand((template_co_occurrence.shape[0],self.shape[0],self.shape[1]), dtype=t.float32, device=self.device)
        # generative_map *= counts[:,None,None]/generative_map.sum(dim=(1,2))[:,None,None]

        # print(generative_map.shape, template_co_occurrence.shape)

        for i in tqdm.tqdm(range(iterations)):
            
            temperature = i/iterations

            generative_map_old = generative_map.clone()
            d_generative_map,d_auto_coocs = self.determine_loss(generative_map, template_co_occurrence)


            d_map_stds = d_generative_map.std(dim=(1,2))
            d_map_stds[d_map_stds<2] = 2
            d_generative_map/=d_map_stds[:,None,None]


            # update generative map
            generative_map+=lr*d_generative_map

            # turn into a proper probability distribution, with a temperature parameter 
            # determining the 
            generative_map = t.nn.functional.softmax(generative_map*(0.0001+temperature*3),dim=0)
            generative_map = generative_map*(1-momentum)+generative_map_old*momentum

            if (render_args is not None) and (i%render_args['steps_per_frame']==0):
                pass
                
        return generative_map

    def render_map(self,generative_map):
        """ """
        fig, ax = plt.subplots(figsize=(15,8))

        plt.subplot(1,2,1)
        plt.imshow(generative_map.argmax(0).cpu().numpy(),cmap='nipy_spectral',interpolation='none',vmin=0,vmax=generative_map.shape[0]-1)

        plt.subplot(1,2,2)
        plt.plot(d_auto_coocs.T)

        plt.savefig(f'{render_args["save_dir"]}/{i:0>4d}.png')
        plt.close()

