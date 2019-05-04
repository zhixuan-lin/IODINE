import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from lib.utils.vis_logger import logger

class IODINE(nn.Module):
    def __init__(self, K, n_iters, dim_latent):
        nn.Module.__init__(self)
        self.dim_latent = dim_latent
        self.n_iters = n_iters
        self.K = K
        self.refine = RefinementNetwork(self.get_input_size(), 64, 256, dim_latent)
        self.broadcast = SpatialBroadcast()
        self.decoder = Decoder(dim_in=dim_latent + 2, dim_hidden=64)
        
        self.posterior = Gaussian()
        
        # global variance, trainable
        self.logvar = nn.Parameter(torch.zeros(1))
        
        # lstm hidden states
        self.lstm_hidden = None
        
        
        # these states are necessary for input encoding
        self.z = [None] * K
        # output mean
        self.mean = [None] * K
        # mask logits
        self.mask_logits = [None] * K
        # mask
        self.mask = [None] * K
        # output mean
        self.mean = [None] * K
        # p(x|z_k), dimension-wise
        self.log_likelihood = [None] * K
        # kl divergence KL(p(zk|x)||p(z_k))
        self.kl = [None] * K
        
    def forward(self, x):
        """
        :param x: (B, 3, H, W)
        :return: loss
        """
        B, _, H, W = x.size()
        # initialization
        self.posterior.init_unit((B, self.K, self.dim_latent), device=x.device)
        self.lstm_hidden = None
        elbos = []
        for i in range(self.n_iters):
            # compute ELBO
            elbo = self.elbo(x)
            elbo.backward(retain_graph=True)
            elbos.append(elbo)
            # get inputs to the refinement network
            # (B, K, D, H, W), D depends on encoding
            input = self.get_input_encoding(x)
            # get refinement
            mean_delta, logvar_delta, self.lstm_hidden = self.refine(input, self.lstm_hidden)
            self.posterior.update(mean_delta, logvar_delta)
            # elbo = self.elbo(x)
            
        return -elbo
        
        
    def elbo(self, x):
        """
        Single pass ELBO computation
        :param x: input, (B, 3, H, W)
        :return: elbo, scalar
        """
        B, C, H, W = x.size()
        
        # sample K latent variables
        # (B, K, L)
        self.z = self.posterior.sample()

        # spatial broadcast
        # (B, K, L + 2, H, W)
        z_broadcast = self.broadcast(self.z, W, H)
        
        # compute mask and mean
        # (B, K, 3, H, W), (B, K, 1, H, W)
        self.mean, self.mask_logits = self.decoder(z_broadcast)
        
        # softmax mask
        # (B, K, 1, H, W)
        self.mask = F.softmax(self.mask_logits, dim=1)
        
        # compute kl divergence
        # (B, K, L)
        kl = self.posterior.kl_divergence()
        # sum over (K, L), mean over (B,)
        kl = kl.mean(0).sum()
        
        # compute pixelwise log likelihood (mixture of Gaussian)
        # refer to the formula to see why this is the case
        # (B, 3, H, W)
        # self.log_likelihood = torch.logsumexp(
        #     torch.log(self.mask + 1e-12) + gaussian_log_likelihood(x[:, None], self.mean, torch.exp(self.logvar)),
        #     dim=1
        # )
        # print(gaussian_likelihood(x[:, None], self.mean, torch.exp(self.logvar)).max())
        self.log_likelihood = (
            torch.log(
                torch.sum(
                    self.mask * gaussian_likelihood(x[:, None], self.mean, torch.exp(self.logvar)),
                    dim=1
                )
            )
        )
        
        
        # sum over (3, H, W), mean over B
        log_likelihood = self.log_likelihood.mean(0).sum()
        
        # ELBO
        elbo = log_likelihood - kl
        
        logger.update(image=x[0])
        logger.update(pred=self.mean[0, 0])
        logger.update(kl=kl)
        logger.update(likelihood=log_likelihood)
        
        return elbo
        
    def get_input_encoding(self, x):
        """
        :param x: (B, 3, H, W)
        :return: (B, K, D, H, W), D depends on encoding scheme
        """
        B, C, H, W = x.size()
        # (B, K, 3, H, W)
        x =  x[:, None].repeat(1, self.K, 1, 1, 1)
        # (B, K, L)
        mean_grad = self.posterior.mean.grad.detach()
        # (B, K, L)
        logvar_grad = self.posterior.logvar.grad.detach()
        
        mean_grad = self.layernorm(mean_grad)
        logvar_grad = self.layernorm(logvar_grad)
        # concat to (B, K, L * 2)
        lambda_grad = torch.cat((mean_grad, logvar_grad), dim=-1)
        # (B, K, L*2+2, H, W)
        lambda_grad = self.broadcast(lambda_grad, W, H)
        
        # encoding gradient
        x = torch.cat((x, lambda_grad), dim=2)
        # x = lambda_grad
        
        return x
    
    # def get_input_encoding(self, x):
    #     B, C, H, W = x.size()
    #     # (B, K, 3, H, W)
    #     x =  x[:, None].repeat(1, self.K, 1, 1, 1)
    #     return x
        
    
    def get_input_size(self):
        return 3 + 2 * self.dim_latent + 2
        # return 3
    
    @staticmethod
    def layernorm(x):
        layer_mean = x.mean(dim=0, keepdim=True)
        layer_std = x.std(dim=0, keepdim=True)
        x = (x - layer_mean) / (layer_std + 1e-5)
        return x


class Decoder(nn.Module):
    """
    Given sampled latent variable, output RGB+mask
    """
    def __init__(self, dim_in, dim_hidden):
        nn.Module.__init__(self)
        
        self.conv1 = nn.Conv2d(dim_in, dim_hidden, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(dim_hidden, dim_hidden, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(dim_hidden, dim_hidden, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(dim_hidden, dim_hidden, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(dim_hidden, 4, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        """
        :param x: (B, K, N, H, W), where N is the number of latent dimensions
        :return: (B, K, 3, H, W) (B, K, 1, H, W), where 4 is RGB + mask
        """
        B, K, *ORI = x.size()
        x = x.view(B*K, *ORI)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = self.conv5(x)
        mean, mask = torch.split(x, [3, 1], dim=1)
        mean = F.sigmoid(mean)

        BK, *ORI = mean.size()
        mean = mean.view(B, K, *ORI)
        BK, *ORI = mask.size()
        mask = mask.view(B, K, *ORI)
        return mean, mask
    
class RefinementNetwork(nn.Module):
    """
    Given input encoding, output updates to lambda.
    """
    def __init__(self, dim_in, dim_conv, dim_hidden, dim_out):
        """
        :param dim_in: input channels
        :param dim_conv: conv output channels
        :param dim_hidden: MLP and LSTM output dim
        :param dim_out: latent variable dimension
        """
        nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(dim_in, dim_conv, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(dim_conv, dim_conv, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(dim_conv, dim_conv, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(dim_conv, dim_conv, kernel_size=3, stride=2, padding=1)
        # (D, 128, 128) goes to (64, 8, 8)
        self.mlp = MLP(256, dim_hidden, n_layers=2)
        # self.mlp = MLP(4096, dim_hidden, n_layers=2)
        self.lstm = nn.LSTMCell(dim_hidden, dim_hidden)
        self.mean_update = nn.Linear(dim_hidden, dim_out)
        self.logvar_update = nn.Linear(dim_hidden, dim_out)
        
    def forward(self, x, hidden=(None, None)):
        """
        :param x: (B, K, D, H, W), where D varies for different input encodings
        :param hidden: a tuple (c, h)
        :return: (B, K, L), (B, K, L), (h, c) for mean and gaussian respectively, where L is
                 the latent dimension. And hidden state
        """
        B, K, *ORI = x.size()
        x = x.view(B*K, *ORI)
        
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = x.view(B*K, -1)
        x = F.elu(self.mlp(x))
        (c, h) = self.lstm(x, hidden)
        # update
        mean_delta = self.mean_update(h)
        logvar_delta = self.logvar_update(h)
        
        BK, *ORI = mean_delta.size()
        mean_delta = mean_delta.view(B, K, *ORI)
        BK, *ORI = logvar_delta.size()
        logvar_delta = logvar_delta.view(B, K, *ORI)
        # BK, *ORI = c.size()
        # c = c.view(B, K, *ORI)
        # BK, *ORI = h.size()
        # h = h.view(B, K, *ORI)
        
        return mean_delta, logvar_delta, (c, h)
    
class SpatialBroadcast(nn.Module):
    """
    Broadcast a 1-D variable to 3-D, plus two coordinate dimensions
    """
    def __init__(self):
        nn.Module.__init__(self)
        
    def forward(self, x, width, height):
        """
        :param x: (B, K, L)
        :return: (B, K, L + 2, W, H)
        """
        B, K, *ORI = x.size()
        x = x.view(B*K, *ORI)

        # (B, L, 1, 1)
        x = x[:, :, None, None]
        # (B, L, W, H)
        x = x.repeat(1, 1, width, height)
        # create meshgrid
        xx = np.linspace(-1, 1, width)
        yy = np.linspace(-1, 1, height)
        xx, yy = np.meshgrid(xx, yy)
        # (1, 1, W, H)
        xx, yy = [torch.from_numpy(i)[None,None,:,:] for i in [xx, yy]]
        # (B, 1, W, H)
        xx, yy = [i.repeat(B*K, 1, 1, 1).float() for i in [xx, yy]]
        xx, yy = [i.to(x.device) for i in [xx, yy]]

        # (B, L + 2, W, H)
        x = torch.cat((x, xx, yy), dim=1)
        
        
        BK, *ORI = x.size()
        x = x.view(B, K, *ORI)
        
        return x
        
    
class MLP(nn.Module):
    """
    Multi-layer perception using elu
    """
    
    def __init__(self, dim_in, dim_out, n_layers):
        nn.Module.__init__(self)
        self.dim_in = dim_in
        self.dim_out = dim_out
        
        self.layers = nn.ModuleList([])
        
        for i in range(n_layers):
            self.layers.append(nn.Linear(dim_in, dim_out))
            dim_in = dim_out
            
    def forward(self, x):
        """
        :param x: (B, Din)
        :return: (B, Dout)
        """
        for layer in self.layers:
            x = F.elu(layer(x))
            
        return x
    
class Gaussian(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.mean = None
        self.logvar = None
        
    def init_unit(self, size, device):
        """
        Initialize to be a unit gaussian
        """
        self.mean = torch.zeros(size, device=device, requires_grad=True)
        self.logvar = torch.zeros(size, device=device, requires_grad=True)
        self.mean.retain_grad()
        self.logvar.retain_grad()
    
    def sample(self):
        """
        Sample from current mean and dev
        :return: return size is the same as self.mean
        """
        # log deviation
        logdev = 0.5 * self.logvar
        
        # standard deviation
        dev = torch.exp(logdev)
        
        # add dimension
        epsilon = torch.randn_like(self.mean).to(self.mean.device)
        
        return self.mean + dev * epsilon
    
    def update(self, mean_delta, logvar_delta):
        """
        :param mean_delta: (B, L)
        :param logvar_delta: (B, L)
        :return:
        """
        self.mean = self.mean + mean_delta
        self.logvar = self.logvar + logvar_delta
        self.mean.retain_grad()
        self.logvar.retain_grad()
    
    def kl_divergence(self):
        """
        Compute dimension-wise KL divergence between estimated dist and standrad gaussian
        """
        var = torch.exp(self.logvar)
        kl = 0.5 * (var + self.mean ** 2 - 1 - self.logvar)
        return kl
    
def gaussian_log_likelihood(x, loc, scale):
    """
    Dimension-wise log likelihood
    """
    from math import pi, sqrt, log
    return -(x - loc) ** 2 / (2 * scale ** 2) - torch.log(scale) - 0.5 * log(2 * pi)

def gaussian_likelihood(x, loc, scale):
    from math import pi, sqrt, log
    return (1 / (sqrt(2*pi) * scale)) * torch.exp(-(x - loc) ** 2 / (2 * scale ** 2))

if __name__ == '__main__':
    net = IODINE(3, 3, 128)
    H, W = 32, 32
    B = 4
    x = torch.randn(B, 3, H, W)
    for i in range(5):
        loss = net(x)
        loss.backward()

