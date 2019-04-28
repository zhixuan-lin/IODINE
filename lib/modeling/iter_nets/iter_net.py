import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from lib.utils.vis_logger import logger


class IterNet(nn.Module):
    def __init__(self, dim_in, dim_latent, n_iters=5):
        nn.Module.__init__(self)
        self.dim_in = dim_in
        self.dim_latent = dim_latent
        self.n_iters = n_iters
        self.encoder = MLP(self.input_size(), 256, 256, 'relu')
        self.gaussian = GaussianLayer(256, dim_latent)
        self.decoder = MLP(dim_latent, 256, dim_in, 'sigmoid')
        self.bce = nn.BCELoss(reduction='none')
        
    def init(self, x):
        """
        For a batch x,
            - Initialize posterior from prior
            - Initialize posterior gradient using elbo
        :param x: (B, D)
        """
        # reshape
        B = x.size(0)
        x = x.view(B, -1)
        org = x.clone()
        
        # initialize from prior
        self.gaussian.init_unit((B, self.dim_latent), x.device)
        # sample z: (B, N, L)
        x = self.gaussian.sample(n_samples=1)
        # (B, N, D)
        x = self.decoder(x)
        # broadcast input
        x, org = torch.broadcast_tensors(x, org[:, None, :])
        # (B, N, D)
        bce = self.bce(x, org)
        # (B, N)
        bce = bce.sum(dim=-1)
        
        # this will be zero, so not needed
        # kl = self.gaussian.kl_divergence()
        
        # compute lambda gradient
        neg_elbo = bce.mean()
        neg_elbo.backward()
        

    def forward(self, x, n_samples=1, reduce=True):
        """
        This forward pass is used for training. Only one sample will be used.
        :param x: (B, 1, H, W)
        :return: if reduce, a single loss. Otherwise it will be (B, N)
        """
        # initialize lambda and its gradient from prior
        self.init(x)
        for i in range(self.n_iters):
            # compute elbo for this iteration
            elbo = self.elbo(x, n_samples=n_samples)
            # compute new gradient
            # note retain graph has to be true for backpropagation to all iterations
            (-elbo.mean()).backward(retain_graph=True)
            
        
        if reduce:
            elbo = elbo.mean()
        return -elbo
    
    def elbo(self, x, n_samples=1):
        """
        Evaluates elbo for each sample (not averaged)
        :param x: (B, 1, H, W)
        :return:
            elbo: (B, N)
        """
        B = x.size(0)
        x = x.view(B, -1)
        org = x.clone()
        x = self.get_input_encoding(x)
        # (B, D)
        x = self.encoder(x)
        # (B, N, L)
        x = self.gaussian(x, n_samples)
        # (B, N, D)
        x = self.decoder(x)
        # broadcast input
        x, org = torch.broadcast_tensors(x, org[:, None, :])
        # (B, N, D)
        bce = self.bce(x, org)
        # (B, N)
        bce = bce.sum(dim=-1)
        # (B,)
        kl = self.gaussian.kl_divergence()
        
        logger.update(image=org[0, 0].view(28, 28))
        logger.update(pred=x[0, 0].view(28, 28))
        logger.update(bce=bce.mean())
        logger.update(kl=kl.mean())
        # generate from unit gaussian
        z = torch.randn(1, self.dim_latent).to(x.device)
        gen = self.decoder(z)
        logger.update(gen=gen.view(1, 28, 28)[0])
        
        return -bce - kl[:, None]
    
    def input_size(self):
        return self.dim_in + 6 * self.dim_latent
    
    def get_input_encoding(self, x):
        """
        Encodings:
            - x: (B, D)
            - mean: (B, L)
            - var: (B, L)
            - mean_log_gradient (B, L)
            - log_var_log_gradient (B, L)
            - sign_mean_gradient (B, L)
            - sign_log_var_log_gradient (B, L)
        """
        eps = 1e-5
        mean = self.gaussian.mean
        log_var = self.gaussian.log_var
        
        # initial: first iteration, initialize all to zero
        # log gradients
        mean_log_gradient = torch.log(mean.grad.abs().detach() + eps)
        log_var_log_gradient = torch.log(log_var.grad.abs().detach() + eps)
        # gradient sign
        mean_sign_gradient = torch.sign(mean.grad.abs().detach())
        log_var_sign_gradient = torch.sign(log_var.grad.abs().detach())

        logger.update(mean_log_grad_mag=mean_log_gradient.mean())
        logger.update(log_var_log_grad_mag=mean_log_gradient.mean())
        
        # concatenate all inputs
        x = (
            x,
            mean, log_var,
            mean_log_gradient, log_var_log_gradient,
            mean_sign_gradient, log_var_sign_gradient
        )
        x = torch.cat(x, dim=1)
        return x


class MLP(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, act):
        nn.Module.__init__(self)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.fc1 = nn.Linear(dim_in, dim_h)
        self.fc2 = nn.Linear(dim_h, dim_out)
        self.act = act
    
    def forward(self, x):
        """
        Dimension preserving
        :param x: (B, *, D_in)
        :return: (B, *, D_out)
        """
        # A will contain any other size
        *A, _ = x.size()
        # reshape
        x = x.view(-1, self.dim_in)
        x = F.relu(self.fc1(x))
        if self.act == 'relu':
            x = F.relu(self.fc2(x))
        else:
            x = torch.sigmoid(self.fc2(x))
        
        x = x.view(*A, self.dim_out)
        
        return x


class GaussianLayer(nn.Module):
    def __init__(self, dim_in, dim_latent):
        nn.Module.__init__(self)
        self.mean_layer = nn.Linear(dim_in, dim_latent)
        # log variance here
        self.log_var_layer = nn.Linear(dim_in, dim_latent)
        
        # self.normal = Normal(0, 1)
    
    def forward(self, x, n_samples):
        """
        :param x: input from encoder (B, D)
        :return: (B, N, D), where N is the number of samples
        """
        # (B, L)
        self.mean = self.mean_layer(x)
        # log standard deviation here
        self.log_var = self.log_var_layer(x)
        # required for next iteration
        self.mean.retain_grad()
        self.log_var.retain_grad()
        return self.sample(n_samples)
    
    def init_unit(self, size, device):
        """
        Initialize to be a unit gaussian
        """
        self.mean = torch.zeros(size, device=device, requires_grad=True)
        self.log_var = torch.zeros(size, device=device, requires_grad=True)
    
    def sample(self, n_samples):
        """
        Sample from current mean and dev
        :param n_samples:
        :return:
        """
        N = n_samples
        B, D = self.mean.size()
        # log deviation
        log_dev = 0.5 * self.log_var
        # standard deviation
        dev = torch.exp(log_dev)
        mean = self.mean[:, None, :]
        dev = dev[:, None, :]
        epsilon = torch.randn(B, N, D).to(self.mean.device)

        return mean + dev * epsilon
    
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


