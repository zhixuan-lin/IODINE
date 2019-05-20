import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from lib.utils.vis_logger import logger

class IODINE(nn.Module):
    def __init__(self, ARCH):
        nn.Module.__init__(self)
        self.dim_latent = ARCH.DIM_LATENT
        self.n_iters = ARCH.ITERS
        self.K = ARCH.SLOTS
        self.encodings = ARCH.ENCODING
        self.img_channels = ARCH.IMG_CHANNELS
        # standard derivation for computing likelihood
        self.sigma = ARCH.SIGMA
        # use layernormalizatoin?
        self.use_layernorm = ARCH.LAYERNORM
        # use stop gradient? (not availabe now)
        self.use_stop_gradient = ARCH.STOP_GRADIENT
        
        
        # architecture
        self.refine = RefinementNetwork(
            self.get_input_size(), ARCH.REF.CONV_CHAN, ARCH.REF.MLP_UNITS,
            ARCH.DIM_LATENT, img_size=ARCH.IMG_SIZE, kernel_size=ARCH.REF.KERNEL_SIZE)
        self.broadcast = SpatialBroadcast()
        # 2 for 2 coordinate channels
        self.decoder = Decoder(dim_in=ARCH.DIM_LATENT + 2, dim_hidden=ARCH.DEC.CONV_CHAN,
                               kernel_size=ARCH.DEC.KERNEL_SIZE)
        self.posterior = Gaussian()
        
        
        
        # lstm hidden states
        self.lstm_hidden = None
        
        # these states are necessary for input encoding
        self.z = None
        # output mean
        self.mean = None
        # mask logits
        self.mask_logits = None
        # mask
        self.mask = None
        # output mean
        self.mean = None
        # p(x|z_k), dimension-wise
        self.log_likelihood = None
        # kl divergence KL(p(z_k|x)||p(z_k))
        self.kl = None
        
    def forward(self, x):
        """
        :param x: (B, 3, H, W)
        :return: loss
        """
        # logger.update(var=self.logvar.item())
        B, _, H, W = x.size()
        # initialization, set posterior to prior
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
            
        elbo = 0
        for i, e in enumerate(elbos):
            elbo = elbo + (i + 1) / self.n_iters * e
            
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
        
        self.mean.retain_grad()
        
        # softmax mask
        # (B, K, 1, H, W)
        self.mask = F.softmax(self.mask_logits, dim=1)
        
        self.mask.retain_grad()
        
        # compute kl divergence
        # (B, K, L)
        kl = self.posterior.kl_divergence()
        # sum over (K, L), mean over (B,)
        kl = kl.mean(0).sum()
        
        # compute pixelwise log likelihood (mixture of Gaussian)
        # refer to the formula to see why this is the case
        # (B, 3, H, W)
        # self.log_likelihood = torch.logsumexp(
        #     torch.log(self.mask + 1e-12) + gaussian_log_likelihood(x[:, None], self.mean, 0.2),
        #     dim=1
        # )
        # print(gaussian_likelihood(x[:, None], self.mean, torch.exp(self.logvar)).max())
        # self.likelihood (B, K, 3, H, W)
        # self.mask (B, K, 1, H, W) self.mean (B, K, 3, H, W) x (B, 3, H, W)
        # self.log_likelihood (B, 3, H, W)
        self.likelihood =  gaussian_likelihood(x[:, None], self.mean, self.sigma)
        self.log_likelihood = (
            torch.log(
                torch.sum(
                    # self.mask * gaussian_likelihood(x[:, None], self.mean, torch.exp(self.logvar)),
                    self.mask * self.likelihood,
                    dim=1
                )
            )
        )
        
        
        # sum over (3, H, W), mean over B
        log_likelihood = self.log_likelihood.mean(0).sum()
        
        # ELBO
        elbo = log_likelihood - kl
        # elbo = log_likelihood
        pred = torch.sum(self.mask * self.mean, dim=1)
        logger.update(image=x[0])
        logger.update(pred=pred[0])
        logger.update(kl=kl)
        logger.update(likelihood=log_likelihood)
        
        masks = {}
        for i in range(self.K):
            masks['mask_{}'.format(i)] = self.mask[0, i, 0]
        preds = {}
        for i in range(self.K):
            preds['pred_{}'.format(i)] = self.mean[0, i]
            
        logger.update(**masks)
        logger.update(**preds)
        
        return elbo
        
    def get_input_encoding(self, x):
        """
        :param x: (B, 3, H, W)
        :return: (B, K, D, H, W), D depends on encoding scheme
        """
        
        encoding = None
        B, C, H, W = x.size()
        
        if 'image' in self.encodings:
            # (B, K, 3, H, W)
            encoding = x[:, None].repeat(1, self.K, 1, 1, 1)
            
        if 'grad_post' in self.encodings:
            # (B, K, L)
            mean_grad = self.posterior.mean.grad.detach()
            # (B, K, L)
            logvar_grad = self.posterior.logvar.grad.detach()
        
            if self.use_layernorm:
                mean_grad = self.layernorm(mean_grad)
                logvar_grad = self.layernorm(logvar_grad)
            # concat to (B, K, L * 2)
            lambda_grad = torch.cat((mean_grad, logvar_grad), dim=-1)
            # (B, K, L*2+2, H, W)
            lambda_grad = self.broadcast(lambda_grad, W, H)
    
            encoding = torch.cat([encoding, lambda_grad], dim=2) if encoding is not None else lambda_grad
        
        if 'means' in self.encodings:
            encoding = torch.cat([encoding, self.mean], dim=2) if encoding is not None else self.mean
        if 'mask' in self.encodings:
            encoding = torch.cat([encoding, self.mask], dim=2) if encoding is not None else self.mask
        if 'mask_logits' in self.encodings:
            encoding = torch.cat([encoding, self.mask_logits], dim=2) if encoding is not None else self.mask_l
        if 'grad_means' in self.encodings:
            mean_grad = self.mean.grad.detach()
            encoding = torch.cat([encoding, mean_grad], dim=2) if encoding is not None else mean_grad
        if 'grad_mask' in self.encodings:
            mask_grad = self.mask.grad.detach()
            encoding = torch.cat([encoding, mask_grad], dim=2) if encoding is not None else mask_grad
        if 'posterior' in self.encodings:
            # current posterior
            # (B, K, L)
            mean = self.posterior.mean
            logvar = self.posterior.logvar
            # concat to (B, K, L * 2)
            lambda_post = torch.cat((mean, logvar), dim=-1)
            # (B, K, L*2+2, H, W)
            lambda_post = self.broadcast(lambda_post, W, H)

            encoding = torch.cat([encoding, lambda_post], dim=2) if encoding is not None else lambda_post
            
        if 'mask_posterior' in self.encodings:
            # not implemented
            pass
        
        if 'likelihood' in self.encodings:
            # (B, 3, H, W)
            log_likelihood = self.log_likelihood
            log_likelihood = log_likelihood[:, None].repeat(1, self.K, 1, 1, 1)
            encoding = torch.cat([encoding, log_likelihood], dim=2) if encoding is not None else log_likelihood
            
        if 'leave_one_out_likelihood' in self.encodings:
            # not implemented
            pass

        return encoding.detach()
    
    def get_input_size(self):
        size = 0
        if 'image' in self.encodings:
            size += self.img_channels
        if 'grad_post' in self.encodings:
            size += 2 * self.dim_latent + 2
        if 'means' in self.encodings:
            size += self.img_channels
        if 'mask' in self.encodings:
            size += 1
        if 'mask_logits' in self.encodings:
            size += 1
        if 'grad_means' in self.encodings:
            size += self.img_channels
        if 'grad_mask' in self.encodings:
            size += 1
        if 'posterior' in self.encodings:
            size += 2 * self.dim_latent + 2
        if 'mask_posterior' in self.encodings:
            # not implemented
            pass

        if 'likelihood' in self.encodings:
            size += self.img_channels

        if 'leave_one_out_likelihood' in self.encodings:
            # not implemented
            pass
            
        return size
    
    @staticmethod
    def layernorm(x):
        """
        :param x: (B, K, L)
        :return:
        """
        layer_mean = x.mean(dim=2, keepdim=True)
        layer_std = x.std(dim=2, keepdim=True)
        x = (x - layer_mean) / (layer_std + 1e-5)
        return x
    
    @staticmethod
    def stop_gradient(x, threshold=5.0):
        """
        :param x: (B, K, L)
        :return: (B, K, L)
        """
        
        norm = torch.norm(x, dim=2)
        indices = norm > threshold
        x[indices, :] = x[indices, :] / norm[indices][:,None] * threshold
        
        return x
        


class Decoder(nn.Module):
    """
    Given sampled latent variable, output RGB+mask
    """
    def __init__(self, dim_in, dim_hidden, kernel_size):
        nn.Module.__init__(self)
        
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(dim_in, dim_hidden, kernel_size=kernel_size, stride=1, padding=padding)
        self.conv2 = nn.Conv2d(dim_hidden, dim_hidden, kernel_size=kernel_size, stride=1, padding=padding)
        self.conv3 = nn.Conv2d(dim_hidden, dim_hidden, kernel_size=kernel_size, stride=1, padding=padding)
        self.conv4 = nn.Conv2d(dim_hidden, dim_hidden, kernel_size=kernel_size, stride=1, padding=padding)
        self.conv5 = nn.Conv2d(dim_hidden, 4, kernel_size=kernel_size, stride=1, padding=padding)
        
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
    def __init__(self, dim_in, dim_conv, dim_hidden, dim_out, img_size, kernel_size):
        """
        :param dim_in: input channels
        :param dim_conv: conv output channels
        :param dim_hidden: MLP and LSTM output dim
        :param dim_out: latent variable dimension
        """
        nn.Module.__init__(self)
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(dim_in, dim_conv, kernel_size=kernel_size, stride=2, padding=padding)
        self.conv2 = nn.Conv2d(dim_conv, dim_conv, kernel_size=kernel_size, stride=2, padding=padding)
        self.conv3 = nn.Conv2d(dim_conv, dim_conv, kernel_size=kernel_size, stride=2, padding=padding)
        self.conv4 = nn.Conv2d(dim_conv, dim_conv, kernel_size=kernel_size, stride=2, padding=padding)
        # (D, 128, 128) goes to (64, 8, 8)
        self.mlp = MLP(dim_conv * (img_size // 16) ** 2, dim_hidden, n_layers=2)
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
        :param coordinate: whether to a the coordinate dimension
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
    return -(x - loc) ** 2 / (2 * scale ** 2) - log(scale) - 0.5 * log(2 * pi)
    # return -(x - loc) ** 2

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

