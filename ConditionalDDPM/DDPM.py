import torch
import torch.nn as nn
import torch.nn.functional as F
from ResUNet import ConditionalUnet
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConditionalDDPM(nn.Module):
    def __init__(self, dmconfig):
        super().__init__()
        self.dmconfig = dmconfig
        self.loss_fn = nn.MSELoss()
        self.network = ConditionalUnet(1, self.dmconfig.num_feat, self.dmconfig.num_classes)

    def scheduler(self, t_s):
        beta_1, beta_T, T = self.dmconfig.beta_1, self.dmconfig.beta_T, self.dmconfig.T
        # ==================================================== #
        # YOUR CODE HERE:
        #   Inputs:
        #       t_s: the input time steps, with shape (B,1). 
        #   Outputs:
        #       one dictionary containing the variance schedule
        #       $\beta_t$ along with other potentially useful constants.       
        
        # Normalize time step to range [0, 1], noting that we start at t=1 and not t=0
        t_normalized = (t_s - 1) / (T - 1)

        beta_t = beta_1 + (beta_T - beta_1) * t_normalized #beta has a constant positive linear slope
        sqrt_beta_t = torch.sqrt(beta_t)
        alpha_t = 1 - beta_t

        oneover_sqrt_alpha = 1/torch.sqrt(alpha_t)
        
        t_s_tensor = t_s.unsqueeze(0) if t_s.dim() == 0 else t_s
        alpha_t_bar = torch.zeros_like(t_s_tensor, dtype=torch.float32)
        i=0
        for ts_single in t_s_tensor: # need an alpha_t_bar at every timestep up to t
            alpha_1 = 1 - beta_1
            alpha_t_bar[i] = torch.prod(alpha_1 - (beta_T - beta_1) * torch.arange(ts_single.item()) / (T-1)) # ts_single is like t1 t2 t3 ...
            i+=1

        sqrt_alpha_bar = torch.sqrt(1-alpha_t_bar)
        sqrt_oneminus_alpha_bar = torch.sqrt(1-alpha_t_bar)
        # ==================================================== #
        return {
            'beta_t': beta_t,
            'sqrt_beta_t': sqrt_beta_t,
            'alpha_t': alpha_t,
            'sqrt_alpha_bar': sqrt_alpha_bar,
            'oneover_sqrt_alpha': oneover_sqrt_alpha,
            'alpha_t_bar': alpha_t_bar,
            'sqrt_oneminus_alpha_bar': sqrt_oneminus_alpha_bar
        }

    def forward(self, images, conditions):
        T = self.dmconfig.T
        noise_loss = None
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given training algorithm.
        #   Inputs:
        #       images: real images from the dataset, with size (B,1,28,28).
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #   Outputs:
        #       noise_loss: loss computed by the self.loss_fn function  .  

        p_uncond = self.dmconfig.mask_p
        num_classes = self.dmconfig.num_classes
        conditions_1hot = F.one_hot(conditions, num_classes = num_classes) #torch.nn.functional
        batch_size = images.size(0)

        # randomly discard conditioning to train unconditionally
        mask = torch.bernoulli(torch.full((batch_size,), p_uncond, device=device)).view(batch_size, 1)
        conditions_1hot = conditions_1hot * (1 - mask) + mask * self.dmconfig.condition_mask_value

        # Generate t sampled uniformly from {1, ..., T}
        t = torch.randint(low=1, high=T+1, size=(batch_size,), device=device)
        t_normalized = t / T
        
        epsilon = torch.randn_like(images, device = device)

        schedule_dict = self.scheduler(t)

        x_t = schedule_dict['alpha_t_bar'].view(-1,1,1,1).to(device)*images + schedule_dict['sqrt_oneminus_alpha_bar'].view(-1,1,1,1).to(device)*epsilon
        
        out = self.network(x_t, t_normalized, conditions_1hot)

        noise_loss = self.loss_fn(out, epsilon)
        # ==================================================== #
        
        return noise_loss

    def sample(self, conditions, omega):
        T = self.dmconfig.T
        X_t = None
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given sampling algorithm.
        #   Inputs:
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #       omega: conditional guidance weight.
        #   Outputs:
        #       generated_images  

        input_dim = self.dmconfig.input_dim
        num_channels = self.dmconfig.num_channels
        batch_size = conditions.shape[0]
        
        # conditions is already one hot encoded

        x_prev = torch.randn(batch_size, self.dmconfig.num_channels, self.dmconfig.input_dim[0], self.dmconfig.input_dim[1], device=device)
        
        with torch.no_grad():
            for t in torch.arange(T,0,-1):
                schedule_dict = self.scheduler(t)
                t_normalized = torch.full((batch_size,1,1,1),t, device=device) / T

                if t>1:
                    z = torch.randn_like(x_prev[0])
                else:
                    z = torch.zeros_like(x_prev[0])

                sigma_t = torch.sqrt(schedule_dict['beta_t'].view(-1,1,1,1).to(device))

                epsilon_t = (1 + omega)*self.network(x_prev, t_normalized, conditions) - omega*self.network(x_prev, t_normalized, (torch.zeros_like(conditions)-1))
                x_prev = schedule_dict['oneover_sqrt_alpha'].view(-1,1,1,1).to(device) * (x_prev - (1-schedule_dict['alpha_t'].view(-1,1,1,1).to(device))/schedule_dict['sqrt_oneminus_alpha_bar'].view(-1,1,1,1).to(device) *epsilon_t  ) + sigma_t*z

        X_t = x_prev

        pass

        # ==================================================== #
        generated_images = (X_t * 0.3081 + 0.1307).clamp(0,1) # denormalize the output images
        return generated_images