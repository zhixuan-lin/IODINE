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
        self.img_size = ARCH.IMG_SIZE
        # standard derivation for computing likelihood
        self.sigma = ARCH.SIGMA
        # use layernormalizatoin?
        self.use_layernorm = ARCH.LAYERNORM
        # use stop gradient? (not availabe now)
        self.use_stop_gradient = ARCH.STOP_GRADIENT
        
        
        # architecture
        input_size, lambda_size = self.get_input_size()
        self.refine = RefinementNetwork(
            input_size, ARCH.REF.CONV_CHAN, ARCH.REF.MLP_UNITS,
            ARCH.DIM_LATENT, ARCH.REF.CONV_LAYERS, kernel_size=ARCH.REF.KERNEL_SIZE,
            stride=ARCH.REF.STRIDE)
        self.decoder = Decoder(dim_in=ARCH.DIM_LATENT, dim_hidden=ARCH.DEC.CONV_CHAN,
                               n_layers=ARCH.DEC.CONV_LAYERS, kernel_size=ARCH.DEC.KERNEL_SIZE,
                               img_size=self.img_size)
        self.posterior = Gaussian(self.dim_latent)
        
        
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
        
        # weight initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out')
        
    def decode(self, z):
        """
        Generate a scene from z
        """
        # z: (B, K, L) mean (B, K, 3, H, W) mask_logits (B, K, 1, H, W)
        mean, mask_logits = self.decoder(z)
    
        # (B, K, 1, H, W)
        mask = F.softmax(mask_logits, dim=1)
    
        pred = torch.sum(mask * mean, dim=1)
        
        return pred, mask, mean
    
    def encode(self, x):
        """
        Get z from images
        :param x: (B, 3, H, W)
        :return z: (B, K, L)
        """
        B, _, H, W = x.size()
        # self.posterior.init_unit((B, self.K, self.dim_latent), device=x.device)
        self.posterior.init_unit(B, self.K)
        self.lstm_hidden = None
        for i in range(self.n_iters):
            # compute ELBO
            elbo = self.elbo(x)
            # note this ELBO is averaged over batch, so way multiply
            # by batch size to get a summed-over-batch version of elbo
            # this ensures that inference is invariant to batch size
    
            (B * elbo).backward(retain_graph=False)
            # get inputs to the refinement network
            # (B, K, D, H, W), D depends on encoding
            input, latent = self.get_input_encoding(x)
    
            mean_delta, logvar_delta, self.lstm_hidden = self.refine(input, latent, self.lstm_hidden)
            # nasty detail: detach lambda to prevent backprop through lambda
            # this is achieve by setting lambda delta and current lambda to be detached,
            # but new lambda to require grad
            mean_delta, logvar_delta = [x.detach() for x in (mean_delta, logvar_delta)]
            self.posterior.update(mean_delta, logvar_delta)

        # finally, sample z
        z = self.posterior.sample()
        
        return z
    
    def reconstruct(self, x):
        
        z = self.encode(x)
        pred, mask, mean = self.decode(z)
        
        return pred, mask, mean
        

    def forward(self, x):
        """
        :param x: (B, 3, H, W)
        :return: loss
        """
        # logger.update(var=self.logvar.item())
        B, _, H, W = x.size()
        # self.posterior.init_unit((B, self.K, self.dim_latent), device=x.device)
        self.posterior.init_unit(B, self.K)
        self.lstm_hidden = None
        elbos = []
        for i in range(self.n_iters):
            # zero grad
            # for param in self.decoder.parameters():
            #     if param.grad is not None:
            #         param.grad.data.zero_()
            # compute ELBO
            
            elbo = self.elbo(x)
            # note this ELBO is averaged over batch, so way multiply
            # by batch size to get a summed-over-batch version of elbo
            # this ensures that inference is invariant to batch size
            (B * elbo).backward(retain_graph=True)
            elbos.append(elbo)
            
            # get inputs to the refinement network
            # (B, K, D, H, W), D depends on encoding
            input, latent = self.get_input_encoding(x)
            
            mean_delta, logvar_delta, self.lstm_hidden = self.refine(input, latent, self.lstm_hidden)
            self.posterior.update(mean_delta, logvar_delta)
            
        # final elbo
        elbo = self.elbo(x)
        elbos.append(elbo)
        
        elbo = 0
        for i, e in enumerate(elbos):
            elbo = elbo + (i + 1) / len(elbos) * e
            
        # record intial posterior guesses:
        logger.update(init_mean=self.posterior.init_mean.mean())
        logger.update(init_logvar=self.posterior.init_logvar.mean())
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
        # z_broadcast = self.broadcast(self.z, W, H)
        
        # compute mask and mean
        # (B, K, 3, H, W), (B, K, 1, H, W)
        self.mean, self.mask_logits = self.decoder(self.z)
        
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
        
        # self.likelihood (B, K, 3, H, W)
        # self.mask (B, K, 1, H, W) self.mean (B, K, 3, H, W) x (B, 3, H, W)
        # self.log_likelihood (B, 3, H, W)
        # self.likelihood =  gaussian_likelihood(x[:, None], self.mean, self.sigma)
        # self.log_likelihood = (
        #     torch.log(
        #         torch.sum(
        #             # self.mask * gaussian_likelihood(x[:, None], self.mean, torch.exp(self.logvar)),
        #             self.mask * self.likelihood,
        #             dim=1
        #         )
        #     )
        # )
        
        # compute pixelwise log likelihood (mixture of Gaussian)
        self.K_log_likelihood= gaussian_log_likelihood(x[:, None], self.mean, self.sigma)
        # refer to the formula to see why this is the case
        # (B, 3, H, W)
        self.log_likelihood = torch.logsumexp(
            torch.log(self.mask + 1e-12) + self.K_log_likelihood,
            dim=1
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
        latent = None
        B, C, H, W = x.size()

        if 'posterior' in self.encodings:
            # current posterior
            # (B, K, L)
            mean = self.posterior.mean
            logvar = self.posterior.logvar
            # concat to (B, K, L * 2)
            lambda_post = torch.cat((mean, logvar), dim=-1)
    
            latent = torch.cat((latent, lambda_post), dim=2) if latent is not None else lambda_post

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
    
            latent = torch.cat([latent, lambda_grad], dim=2) if latent is not None else lambda_grad

        if 'image' in self.encodings:
            # (B, K, 3, H, W)
            encoding = x[:, None].repeat(1, self.K, 1, 1, 1)
        if 'means' in self.encodings:
            encoding = torch.cat([encoding, self.mean], dim=2) if encoding is not None else self.mean
        if 'mask' in self.encodings:
            encoding = torch.cat([encoding, self.mask], dim=2) if encoding is not None else self.mask
        if 'mask_logits' in self.encodings:
            encoding = torch.cat([encoding, self.mask_logits], dim=2) if encoding is not None else self.mask_logits
        if 'mask_posterior' in self.encodings:
            # self.K_log_likelihood (B, K, 3, H, W)
            # likelihood (B, K, 1, H, W)
            K_log_likelihood = self.K_log_likelihood.sum(dim=2, keepdim=True)
            K_likelihood = torch.exp(K_log_likelihood)
            # normalize
            K_likelihood = K_likelihood / K_likelihood.sum(dim=1, keepdim=True)
            encoding = torch.cat([encoding, K_likelihood], dim=2) if encoding is not None else K_likelihood
            
        if 'grad_means' in self.encodings:
            mean_grad = self.mean.grad.detach()
            if self.use_layernorm:
                mean_grad = self.layernorm(mean_grad)
            encoding = torch.cat([encoding, mean_grad], dim=2) if encoding is not None else mean_grad
        if 'grad_mask' in self.encodings:
            mask_grad = self.mask.grad.detach()
            if self.use_layernorm:
                mask_grad = self.layernorm(mask_grad)
            encoding = torch.cat([encoding, mask_grad], dim=2) if encoding is not None else mask_grad
        
        if 'likelihood' in self.encodings:
            # self.log_likelihood (B, 3, H, W)
            # log_likelihood (B, 1, H, W)
            log_likelihood = torch.sum(self.log_likelihood, dim=1, keepdim=True).detach()
            likelihood = torch.exp(log_likelihood)
            # (B, K, 1, H, W)
            likelihood = likelihood[:, None].repeat(1, self.K, 1, 1, 1)
            if self.use_layernorm:
                likelihood = self.layernorm(likelihood)
            encoding = torch.cat([encoding, likelihood], dim=2) if encoding is not None else likelihood
            
        if 'leave_one_out_likelihood' in self.encodings:
            # This computation is a little weird. Basically we do not count one of the slot.
            # self.K_log_likelihood (B, K, 3, H, W)
            # K_likelihood = (B, K, 1, H, W)
            K_log_likelihood = self.K_log_likelihood.sum(dim=2, keepdim=True)
            K_likelihood = torch.exp(K_log_likelihood)
            # likelihood = (B, 1, 1, H, W), self.mask (B, K, 1, H, W)
            likelihood = (self.mask * K_likelihood).sum(dim=1, keepdim=True)
            # leave_one_out (B, K, 1, H, W)
            leave_one_out = likelihood - self.mask * K_likelihood
            # finally, normalize
            leave_one_out = leave_one_out / (1 - self.mask + 1e-5)
            if self.use_layernorm:
                leave_one_out = self.layernorm(leave_one_out)
            encoding = torch.cat([encoding, leave_one_out], dim=2) if encoding is not None else leave_one_out
        
        if 'coordinate' in self.encodings:
            xx = torch.linspace(-1, 1, W, device=x.device)
            yy = torch.linspace(-1, 1, H, device=x.device)
            yy, xx = torch.meshgrid((yy, xx))
            # (2, H, W)
            coords = torch.stack((xx, yy), dim=0)
            coords = coords[None, None].repeat(B, self.K, 1, 1, 1).detach()
            encoding = torch.cat([encoding, coords], dim=2) if encoding is not None else coords
        

        return encoding.detach(), latent.detach()
    
    def get_input_size(self):
        size = 0
        latent = 0
        if 'grad_post' in self.encodings:
            latent += 2 * self.dim_latent
        if 'posterior' in self.encodings:
            latent += 2 * self.dim_latent
            
        if 'image' in self.encodings:
            size += self.img_channels
        if 'means' in self.encodings:
            size += self.img_channels
        if 'mask' in self.encodings:
            size += 1
        if 'mask_logits' in self.encodings:
            size += 1
        if 'mask_posterior' in self.encodings:
            size += 1
        if 'grad_means' in self.encodings:
            size += self.img_channels
        if 'grad_mask' in self.encodings:
            size += 1
        if 'likelihood' in self.encodings:
            size += 1
        if 'leave_one_out_likelihood' in self.encodings:
            size += 1
        if 'coordinate' in self.encodings:
            size += 2
            
        return size, latent
    
    @staticmethod
    def layernorm(x):
        """
        :param x: (B, K, L) or (B, K, C, H, W)
        :return:
        """
        if len(x.size()) == 3:
            layer_mean = x.mean(dim=2, keepdim=True)
            layer_std = x.std(dim=2, keepdim=True)
        elif len(x.size()) == 5:
            mean = lambda x: x.mean(2, keepdim=True).mean(3, keepdim=True).mean(4, keepdim=True)
            layer_mean = mean(x)
            # this is not implemented in some version of torch
            layer_std = torch.pow(x - layer_mean, 2)
            layer_std = torch.sqrt(mean(layer_std))
        else:
            assert False, 'invalid size for layernorm'
            
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
    def __init__(self, dim_in, dim_hidden, n_layers, kernel_size, img_size):
        nn.Module.__init__(self)
        
        padding = kernel_size // 2
        self.broadcast = SpatialBroadcast()
        self.mlc = MultiLayerConv(dim_in + 2, dim_hidden, n_layers, kernel_size)
        self.conv = nn.Conv2d(dim_hidden, 4, kernel_size=kernel_size, stride=1, padding=padding)
        self.img_size = img_size
        
    def forward(self, x):
        """
        :param x: (B, K, N, H, W), where N is the number of latent dimensions
        :return: (B, K, 3, H, W) (B, K, 1, H, W), where 4 is RGB + mask
        """
        B, K, *ORI = x.size()
        x = x.view(B*K, *ORI)
        
        x = self.broadcast(x, self.img_size, self.img_size)
        x = self.mlc(x)
        x = self.conv(x)
        
        mean, mask = torch.split(x, [3, 1], dim=1)
        mean = torch.sigmoid(mean)

        BK, *ORI = mean.size()
        mean = mean.view(B, K, *ORI)
        BK, *ORI = mask.size()
        mask = mask.view(B, K, *ORI)
        return mean, mask
    
class RefinementNetwork(nn.Module):
    """
    Given input encoding, output updates to lambda.
    """
    def __init__(self, dim_in, dim_conv, dim_hidden, dim_out, n_layers, kernel_size, stride):
        """
        :param dim_in: input channels
        :param dim_conv: conv output channels
        :param dim_hidden: MLP and LSTM output dim
        :param dim_out: latent variable dimension
        """
        nn.Module.__init__(self)
        
        self.mlc = MultiLayerConv(dim_in, dim_conv, n_layers, kernel_size, stride=stride)
        self.mlp = MLP(dim_conv, dim_hidden, n_layers=1)
        
        self.lstm = nn.LSTMCell(dim_hidden + 4 * dim_out, dim_hidden)
        self.mean_update = nn.Linear(dim_hidden, dim_out)
        self.logvar_update = nn.Linear(dim_hidden, dim_out)
        
    def forward(self, x, latent, hidden=(None, None)):
        """
        :param x: (B, K, D, H, W), where D varies for different input encodings
        :param latent: (B, K, L * 4), contains posterior parameters and gradients
        :param hidden: a tuple (c, h)
        :return: (B, K, L), (B, K, L), (h, c) for mean and gaussian respectively, where L is
                 the latent dimension. And hidden state
        """
        B, K, *ORI = x.size()
        x = x.view(B*K, *ORI)
        B, K, *ORI = latent.size()
        latent = latent.view(B*K, *ORI)
        
        # (BK, D, H, W)
        x = self.mlc(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # (BK, D)
        x = x.view(B*K, -1)
        # to uniform length
        x = F.elu(self.mlp(x))
        # concatenate
        x = torch.cat((x, latent), dim=1)
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
        :param x: (B, L)
        :param coordinate: whether to a the coordinate dimension
        :return: (B, L + 2, W, H)
        """
        # B, K, *ORI = x.size()
        # x = x.view(B*K, *ORI)

        B, L = x.size()
        # (B, L, 1, 1)
        x = x[:, :, None, None]
        # (B, L, W, H)
        x = x.repeat(1, 1, width, height)
        xx = torch.linspace(-1, 1, width, device=x.device)
        yy = torch.linspace(-1, 1, height, device=x.device)
        yy, xx = torch.meshgrid((yy, xx))
        # (2, H, W)
        coords = torch.stack((xx, yy), dim=0)
        # (B, 2, H, W)
        coords = coords[None].repeat(B, 1, 1, 1)

        # (B, L + 2, W, H)
        x = torch.cat((x, coords), dim=1)
            
        # BK, *ORI = x.size()
        # x = x.view(B, K, *ORI)
        
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
    

class MultiLayerConv(nn.Module):
    """
    Multi-layer convolutional layer
    """
    def __init__(self, dim_in, dim_out, n_layers, kernel_size, stride=1):
        nn.Module.__init__(self)
        self.dim_in = dim_in
        self.dim_out = dim_out
    
        self.layers = nn.ModuleList([])
        padding = kernel_size // 2
    
        for i in range(n_layers):
            self.layers.append(nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, padding=padding, stride=stride))
            dim_in = dim_out

    def forward(self, x):
        """
        :param x: (B, C, H, W)
        :return: (B, C, H, W)
        """
        for layer in self.layers:
            x = F.elu(layer(x))
    
        return x
    
class Gaussian(nn.Module):
    def __init__(self, dim_latent):
        nn.Module.__init__(self)
        self.mean = None
        self.logvar = None
        
        # prior mean
        self.init_mean = nn.Parameter(data=torch.zeros(dim_latent))
        self.init_logvar = nn.Parameter(data=torch.zeros(dim_latent))
        
    # def init_unit(self, size, device):
    def init_unit(self, B, K):
        """
        Initialize to be a unit gaussian
        """
        # self.mean = torch.zeros(size, device=device, requires_grad=True)
        # self.logvar = torch.zeros(size, device=device, requires_grad=True)
        # self.mean.retain_grad()
        # self.logvar.retain_grad()
        self.mean = self.init_mean[None, None].repeat(B, K, 1)
        self.logvar = self.init_logvar[None, None].repeat(B, K, 1)
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
        epsilon = torch.randn_like(self.mean, device=self.mean.device)
        
        return self.mean + dev * epsilon
    
    def update(self, mean_delta, logvar_delta):
        """
        :param mean_delta: (B, L)
        :param logvar_delta: (B, L)
        :return:
        """
        self.mean = self.mean.detach() + mean_delta
        self.logvar = self.logvar.detach() + logvar_delta
        # these are required since during inference, mean_delta and logvar_delta
        # will be detached
        if self.mean.requires_grad == False:
            self.mean.requires_grad = True
        if self.logvar.requires_grad == False:
            self.logvar.requires_grad = True
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

