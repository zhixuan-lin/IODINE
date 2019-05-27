"""
This file defines a logger object to keep everything. This object is like a
dictionary. To use the logger, use

    from lib.utils.vis_logger import logger

and then log any data using

    logger.update(name0=value, name1=value, ...)

Various getters should also be defined here. These are config-dependent objects,
and should wrap the logger object. It performs config-dependent operations based
on data stored in logger.

To make a getter from a configuration object, use

    getter = make_getter(cfg)

To use the getter to get data for visualization with Tensorboard,
call

    tb_data = getter.get_tensorboard_data()
"""

import torch
from torch.nn import functional as F
import numpy as np


class Logger:
    """
    Interface class
    """
    
    def __init__(self):
        self.things = dict()
    
    def __getitem__(self, key):
        return self.things[key]
    
    def update(self, **kargs):
        # detach any tensor
        # for k in kargs:
        #     if isinstance(kargs[k], torch.Tensor):
        #         kargs[k] = kargs[k].detach().cpu()
        self.things.update(kargs)


# global logger to keep literally everything
logger = Logger()


# getter maker
def make_getter(cfg):
    getter = None
    if cfg.GETTER == 'VAE':
        return VAEGetter()
    

class VAEGetter:
    """
    Designed for matterport
    """
    
    def __init__(self, logger=logger):
        self.logger = logger
    
    def get_tensorboard_data(self):
        """
        This processes the data needed for visualization. It expects the follow-
        
        Assume the following in logger:
        - image: (1, 28, 28), range (0, 1), float
        - pred: (1, 28, 28), range (0, 1), float
        - bce:
        - kl:
        """
        things = self.logger.things
        for k in things:
            if isinstance(things[k], torch.Tensor):
                things[k] = things[k].detach().cpu()
        
        # image = things['image']
        # pred = things['pred']
        #
        # image = image.squeeze()
        # pred = pred.squeeze()
        #
        # things.update(dict(
        #     image=image,
        #     pred=pred
        # ))
        #
        return things
        
        
        
        
        
