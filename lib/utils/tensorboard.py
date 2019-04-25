from tensorboardX import SummaryWriter
from easydict import EasyDict
import os


class TensorBoard:
    """
    Tensorboard wrapper for logging and visualizatoin.
    """
    
    def __init__(self, logdir, scalars, images, resume):
        """
        :param logdir: logging directory
        :param scalars: a list of scalar names
        :param images:  a list of image names
        :param resume: whether to resume logging
        """
        if os.path.exists(logdir) and not resume:
            os.system('rm -r {}'.format(logdir))
        self.logdir = logdir
        self.scalars = scalars
        self.images = images
        self.writer = SummaryWriter(log_dir=logdir)
        self.data = EasyDict()
        
    def update(self, **kargs):
        for key, value in kargs.items():
            self.data[key] = value
            
    def add(self, prefix, global_step):
        pattern = '/'.join([prefix, '{}'])
        for scalar in self.scalars:
            if scalar in self.data:
                self.writer.add_scalar(pattern.format(scalar), self.data[scalar], global_step)
        for image in self.images:
            if image in self.data:
                self.writer.add_image(pattern.format(image), self.data[image], global_step)
                
        
