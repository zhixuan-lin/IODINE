import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from lib.utils.vis_logger import logger


class VAE(nn.Module):
    def __init__(self, dim_in, dim_latent):
        nn.Module.__init__(self)
        self.dim_latent = dim_latent
        self.encoder = MLP(dim_in, 256, 256, 'relu')
        self.gaussian = GaussianLayer(256, dim_latent)
        self.decoder = MLP(dim_latent, 256, dim_in, 'sigmoid')
        self.bce = nn.BCELoss(reduction='none')
        
    def forward(self, x):
        """
        :param x: (B, 1, H, W)
        """
        B = x.size(0)
        x = x.view(B, -1)
        org = x.clone()
        x = self.encoder(x)
        x = self.gaussian(x)
        x = self.decoder(x)
        bce = self.bce(x, org)
        bce = bce.sum(dim=1).mean()
        kl = self.gaussian.kl_divergence().mean()
        
        logger.update(image=org.view(B, 28, 28)[0])
        logger.update(pred=x.view(B, 28, 28)[0])
        logger.update(bce=bce)
        logger.update(kl=kl)
        # generate from unit gaussian
        z = torch.randn(1, self.dim_latent)
        gen = self.decoder(z)
        logger.update(gen=gen.view(1, 28, 28)[0])
        
        return bce + kl
        # return bce
    
class MLP(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, act):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(dim_in, dim_h)
        self.fc2 = nn.Linear(dim_h, dim_out)
        self.act = act
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.act == 'relu':
            x = F.relu(self.fc2(x))
        else:
            x = torch.sigmoid(self.fc2(x))
        
        return x
    
class GaussianLayer(nn.Module):
    def __init__(self, dim_in, dim_latent):
        nn.Module.__init__(self)
        self.mean_layer = nn.Linear(dim_in, dim_latent)
        # log variance here
        self.log_var_layer = nn.Linear(dim_in, dim_latent)
        
        # self.normal = Normal(0, 1)
        
    def forward(self, x):
        """
        :param x: input from encoder (B, D)
        """
        # (B, L)
        self.mean = self.mean_layer(x)
        # log standard deviation here
        self.log_var = self.log_var_layer(x)
        log_dev = 0.5 * self.log_var
        # standard deviation
        dev = torch.exp(log_dev)
        
        # sample
        # (B, L)
        # epsilon = self.normal.sample(sample_shape=self.mean.size())
        epsilon = torch.randn(self.mean.size())
        
        return self.mean + dev * epsilon
    
    def kl_divergence(self):
        """
        Compute KL divergence between estimated dist and standrad gaussian
        return: mean KL
        """
        var = torch.exp(self.log_var)
        # (B, L)
        kl = 0.5 * (var + self.mean ** 2 - 1 - self.log_var)
        # sum over data dimension
        kl = kl.sum(dim=1)
        return kl
        
        
