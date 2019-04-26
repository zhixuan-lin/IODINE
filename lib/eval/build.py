import torch
import numpy as np
from math import sqrt

def make_evaluator(cfg):
    return ELBOEvaluator()

class Evaluator:
    def __init__(self):
        pass
    def evaluate(self, model, data):
        raise NotImplementedError
    

class ELBOEvaluator(Evaluator):
    def __init__(self, n_samples=10):
        Evaluator.__init__(self)
        self.elbo = []
        self.log_px = []
        self.n_samples = n_samples
        
    def reset(self):
        self.elbo = []
        self.log_px = []
    
    def evaluate(self, model, data):
        """
        :param model: model(data) -> -elbo
        :param data: (B, C, H, W)
        """
        B, C, H, W = data.size()
        # (B, N)
        # initialize fr
        elbos = -model(data, n_samples=self.n_samples, reduce=False)
        
        # average to get estimated elbo
        elbo = elbos.mean()
        
        # logsumexp to get an estimate of log p(x)
        # log_px: (B,)
        log_px = torch.logsumexp(elbos, dim=-1) - np.log(self.n_samples)
        log_px = log_px.mean()
        
        self.elbo.append(elbo.item())
        self.log_px.append(log_px.item())
        
    def get_results(self):
        import numpy as np
        elbo = np.array(self.elbo).mean()
        log_px = np.array(self.log_px).mean()
        
        return 'elbo: {:.2f}, log p(x): {:.2f}'.format(elbo, log_px)
    
    def get_result_dict(self):
        import numpy as np
        elbo = np.array(self.elbo).mean()
        log_px = np.array(self.log_px).mean()
        
        return dict(elbo=elbo, log_px=log_px)

# def gaussian_likelihood(x, mean, stddev):
#     from math import pi, sqrt
#     return (
#         (1 / (sqrt(2 * pi) * stddev)) *
#         torch.exp(- (x - mean) ** 2 / (2 * stddev ** 2))
#     )

def log_bernoulli_likelihood(x, phi):
    """
    Assume that each dimension of x is independent
    :param x: (B, D), target
    :param phi: (B, D)
    :return: likelihood of size (D)
    """
    cri = torch.nn.BCELoss(reduction='none')
    log_p_x = (-cri(phi, x).sum(dim=-1))
    return log_p_x

if __name__ == '__main__':
    mean = 0
    stddev = 1

