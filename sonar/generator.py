### This is a script for generating a topographic tensor from a given set of co-occurrence curves.

import numpy as np
import torch as t
from scipy import fft as sp_fft
import tqdm
import matplotlib.pyplot as plt
#import cm

from matplotlib.cm import nipy_spectral

class Generator():
    """Generator class that provides methods for generating topographic tensors from co-occurrence curves."""

    def __init__(self, sonar, shape=(500,500) ) -> None:
        self.sonar = sonar
        self.kernels = sonar.kernels
        self.radii = (sonar.kernels>0.0).sum(dim=(1,2)).float()
        self.radius_steps = t.cat([t.tensor([1],device=sonar.device),t.diff(self.radii)])
        self.device=sonar.device
        self.shape = shape
        self.losses = []
        self.area = shape[0]*shape[1]
        
        shape = [self.kernels.shape[i+1]+shape[i] for i in range(2)]
        self.fshape = [sp_fft.next_fast_len(shape[a], True) for a in [0,1]]

    def optimize_combinations(self, tissue_tensor, counts, iterations=5, exponent=1.1):
        
        tissue_tensor = t.nn.functional.softmax(tissue_tensor,dim=0)

        for i in range(iterations):
            # tissue_tensor = t.nn.functional.softmax(tissue_tensor,dim=0)
            quotient=(tissue_tensor.sum(dim=(1,2)))[:,None,None].clone()
            quotient[quotient<=0]=0.001

            # print("counts:",quotient)            
            tissue_tensor/= quotient
            tissue_tensor*=counts[:,None,None]  
            tissue_tensor = tissue_tensor**exponent
            tissue_tensor/=tissue_tensor.sum(dim=0,keepdim=True)

        tissue_tensor = t.nn.functional.softmax(tissue_tensor,dim=0)
        return tissue_tensor


    def initiate_map(self, template_co_occurrence,mode="random_uniform"):
        """
        initiates a generative map from a template co-occurrence curve
        
        Args:
            template_co_occurrence (torch.Tensor): Template co-occurrence
            mode (str, optional): Mode of initialization. Can be "random_uniform", "random_int", "constant". Defaults to "random_uniform".       
        """

        surface_counts = template_co_occurrence[:,:,0].diagonal()

        if mode =="random_uniform":
            generative_tensor = t.rand((template_co_occurrence.shape[0],self.shape[0],self.shape[1]), dtype=t.float32, device=self.device)
            # generative_tensor *= surface_counts[:,None,None]/generative_tensor.sum(dim=(1,2))[:,None,None]
        if mode == "random_weighted":
            generative_tensor = t.ones((template_co_occurrence.shape[0],self.shape[0],self.shape[1]), dtype=t.float32, device=self.device)
            generative_tensor *= template_co_occurrence[:,:,0].diagonal()[:,None,None]/template_co_occurrence[:,:,0].diagonal().sum()
            generative_tensor += t.normal(0,0.01,generative_tensor.shape, dtype=t.float32, device=self.device)
            generative_tensor = generative_tensor.clamp(min=0,max=1)
        elif mode=="random_int":
            probabilities = surface_counts/surface_counts.sum()
            cdf = probabilities.cumsum(0)
            generative_map = t.rand((self.shape[0],self.shape[1]), dtype=t.float32, device=self.device)
            generative_tensor = t.zeros((template_co_occurrence.shape[0],self.shape[0],self.shape[1]), dtype=t.float32, device=self.device)

            for p,i in enumerate(cdf):
                generative_tensor[p][generative_map<i] = 1

        elif mode=="constant":
            generative_tensor = t.ones((self.shape[0],self.shape[1]), dtype=t.float32, device=self.device)
            generative_tensor *= template_co_occurrence[:,:,0].diagonal()[:,None,None]/generative_tensor.sum(dim=(1,2))[:,None,None]
        else:
            raise ValueError("mode must be one of 'random_uniform', 'random_int', or 'constant'")

        return generative_tensor


    def determine_loss(self, generative_map, template_co_occurrence,momentum=0,temperature=0):
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
        
        radius_steps = t.tensor(self.radius_steps, dtype=t.float32, device=self.device)+1

        for i in range(generative_map.shape[0]):

            h1_fft = t.fft.rfftn(generative_map[i].float(), self.fshape,dim=[0,1])
            h1_fftprod =  (h1_fft*kernels_fft)
            h1_conv = t.fft.irfftn(h1_fftprod,self.fshape,dim=[1,2]).float()
            h1_conv_ =  h1_conv[:,kernel_width//2:kernel_width//2+generative_map.shape[1],
                            kernel_width//2:kernel_width//2+generative_map.shape[2]] #signal._signaltools._centered(h1_conv,[len(kernels)]+fshape).copy()

            h1_product=h1_conv_*generative_map[i]#/np.sum(kernels,axis=(1,2))[:,None,None]
            co_occurrence=h1_product.sum(dim=(1,2))
            co_occurrence/=self.area

            d_cooc = template_co_occurrence[i,i]-co_occurrence
            self.d_co_occurrence[i,i] = (self.d_co_occurrence[i,i]*momentum+d_cooc*(1-momentum)*t.nan_to_num(radius_steps**(temperature/2)))

            d_auto_coocs.append(self.d_co_occurrence[i,i].cpu().numpy())


            d_generative_map[i] += (h1_product*self.d_co_occurrence[i,i][:,None,None]).mean(0)

            for j in range(i+1,generative_map.shape[0]):
                h2_product=h1_conv_*generative_map[j]#/np.sum(kernels,axis=(1,2))[:,None,None]
                co_occurrence = h2_product.sum(dim=(1,2))
                co_occurrence/=self.area
                d_cooc = template_co_occurrence[i,j]-co_occurrence
                self.d_co_occurrence[i,j] = (self.d_co_occurrence[i,j]*momentum+d_cooc*(1-momentum)*t.nan_to_num(radius_steps**(temperature/2)))
                d_update = (h2_product*self.d_co_occurrence[i,j][:,None,None]).mean(0)
                d_generative_map[i] += d_update
                d_generative_map[j] += d_update

        return d_generative_map,np.array(d_auto_coocs)
    
    def update_step(self):
        pass

    def generate(self, template_co_occurrence, iterations=100, lr=5, lr_decay=0.99, 
                 momentum=0.99, momentum_map=0.9, 
                 verbose=True, render_args=None, noise_injection=0.1,init_mode="random_uniform"):
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
            if not "steps_per_frame" in render_args:
                render_args['steps_per_frame'] = iterations//100
            if not "suffix" in render_args:
                render_args['suffix'] = ""
            


        template_co_occurrence = t.tensor(template_co_occurrence, dtype=t.float32, device=self.device)
        # template_co_occurrence/=template_co_occurrence.sum(dim=(1,2))[:,None,None]
        
        counts = template_co_occurrence[:,:,0].diagonal().clone()

        total_counts = counts.sum()
        self.d_co_occurrence = t.zeros_like(template_co_occurrence)
        template_co_occurrence/=total_counts
        generative_tensor = self.initiate_map(template_co_occurrence,mode=init_mode)
        
        self.losses = []
        
        
        for i in tqdm.tqdm(range(iterations)):
            
            temperature = 1-(i/iterations)*0.9

            d_sample_old = generative_tensor.clone()
            d_sample,d_auto_coocs = self.determine_loss(generative_tensor.clone(), template_co_occurrence.clone(),momentum=momentum,temperature=temperature)
            
            self.losses.append(d_sample.abs().mean().item())

            d_sample_stds = d_sample.std(dim=(1,2))
            d_sample_stds[d_sample_stds<0.00001] = 0.00001
            d_sample/=d_sample_stds[:,None,None]
            # print(d_sample.min(),d_sample.max()) 

            d_sample = d_sample*(momentum_map)+d_sample_old*(1-momentum_map)
            generative_tensor += d_sample*lr*(temperature)**(lr_decay)
            
            if temperature>0.2:
                # produce a mask of noise-injection candidates:
                noise_injection_mask = t.rand(generative_tensor.shape[1:])<(noise_injection)
                
                # apply noise to the generative tensor at the noise-injection candidates:
                generative_tensor[:,noise_injection_mask] += t.normal(0,temperature**2/5,(generative_tensor.shape[0],noise_injection_mask.sum()),dtype=t.float32, device=self.device)
                # clean momentum at noise injection candidates:
                d_sample[:,noise_injection_mask] = 0
            
            # if np.random.uniform()<(temperature*noise_injection):
            #     x = np.random.randint(0,generative_tensor.shape[1]-50)
            #     y = np.random.randint(0,generative_tensor.shape[2]-50)
            #     generative_tensor[:,x:x+50,y:y+50] += t.normal(0,0.001,(generative_tensor.shape[0],50,50),dtype=t.float32, device=self.device)
            
            generative_tensor[generative_tensor<0] = 0
            generative_tensor[generative_tensor>1] = generative_tensor[generative_tensor>1]**(0.1*temperature)
            
            # generative_tensor = generative_tensor/generative_tensor.sum(dim=0,keepdim=True).clamp(min=0.00001)
            
            if (render_args is not None) and (i%render_args['steps_per_frame']==0):
                if 'render_stats' in render_args and render_args['render_stats']:
                    self.render_map(generative_tensor,d_sample,d_auto_coocs, i, i//render_args['steps_per_frame'], render_args)
                    
                else:
                    self.render_image(generative_tensor, d_auto_coocs, i, i//render_args['steps_per_frame'], render_args)
                # self.render_map(generative_tensor,d_sample,d_auto_coocs, i, render_args),
                
        return generative_tensor

    def render_map(self,generative_map,d_generative_map,d_auto_coocs, n_frame, i, render_args):
        """ """

        fig, ax = plt.subplots(figsize=(16,12))

        plt.subplot(2,3,1)
        plt.imshow(generative_map.argmax(0).cpu().numpy(),
                   cmap=nipy_spectral,interpolation='none',vmin=0,vmax=d_auto_coocs.shape[0]-1)

        plt.subplot(2,3,2)
        for j in range(d_auto_coocs.shape[0]):
            plt.plot(d_auto_coocs[j],color=nipy_spectral(j/d_auto_coocs.shape[0]))

        plt.subplot(2,3,3)
        plt.title("max_values")
        plt.plot(sorted(generative_map.max(dim=0)[0].cpu().numpy().flatten()))

        plt.subplot(2,3,4)
        plt.title("std delta")
        plt.imshow(generative_map.std(dim=0).cpu().numpy(),vmin=0,vmax=2)
        plt.colorbar()

        plt.subplot(2,3,5)
        plt.title("max delta")
        plt.imshow(d_generative_map.max(dim=0)[0].cpu().numpy())
        plt.colorbar()

        sums = generative_map.sum(dim=(1,2)).cpu().numpy()
        plt.subplot(2,3,6)
        for j in range(d_auto_coocs.shape[0]):
            plt.bar([j],[sums[j]],color=nipy_spectral(j/d_auto_coocs.shape[0]))


        plt.savefig(f'{render_args["save_dir"]}/{render_args["suffix"]}{i:0>4d}.svg')
        plt.close()

    def render_image(self,generative_map, d_auto_coocs, n_frame, i, render_args):
        """ """
        fig, ax = plt.subplots(figsize=(12,12))
        plt.imshow(generative_map.argmax(0).cpu().numpy(),
                   cmap=nipy_spectral,interpolation='none',vmin=0,vmax=d_auto_coocs.shape[0]-1)
        plt.savefig(f'{render_args["save_dir"]}/{render_args["suffix"]}{i:0>4d}.svg',dpi=300)
        plt.close()
