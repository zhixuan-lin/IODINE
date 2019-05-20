from .base import Evaluator
import torch
from lib.utils.ari import compute_mask_ari
import numpy as np


class ARIEvaluator(Evaluator):
    def __init__(self):
        Evaluator.__init__(self)
        
        self.aris = []
        
    def evaluate(self, model, data):
        """
        :param data: (image, mask)
            image: (B, 3, H, W)
            mask: list, each is (N, H, W)
        :return: average ari
        """
        from torch import arange as ar
        image, mask = data
        model(image)
        # (B, K, 1, H, W)
        # (B, K, H, W)
        pred_mask = model.mask
        pred_mask = pred_mask[:, :, 0]
        
        B, K, H, W = pred_mask.size()
        
        # reduced to (B, K, H, W), with 1-0 values
        
        # max_index (B, H, W)
        max_index = torch.argmax(pred_mask, dim=1)
        # get binarized masks (B, K, H, W)
        pred_mask = torch.zeros_like(pred_mask)
        pred_mask[ar(B)[:, None, None], max_index, ar(H)[None, :, None], ar(W)[None, None, :]] = 1.0

        for b in range(B):
            this_ari = compute_mask_ari(mask[b].detach().cpu(), pred_mask[b].detach().cpu())
            self.aris.append(this_ari)
        
    
    def reset(self):
        self.aris = []
    
    def get_results(self):
        return 'Ari: {}'.format(np.mean(self.aris) if self.aris else 0)
