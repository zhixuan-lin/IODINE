import os
import pickle

import torch

class Checkpointer:
    """
    Checkpointer that save and load model, optimizer, scheduler states, and any
    other arguments
    """
    def __init__(
            self,
            model,
            optimizer=None,
            scheduler=None,
            args={},
            max_checkpoints=10,
            save_dir="",
    ):
        """
        :param arguments: other arguments that should be saved
        :param save_dir:
            The data will be saved under 'save_dir/'. A file 'checkpoint.pkl'
            will be created to log all checkpoint files, and checkpoint files
            are themselves saved in 'save_dir/'
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        self.max_checkpoints = max_checkpoints
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.mkdir(save_dir)
        
    def save(self, name):
        """
        Save model, optimizer, scheduler and all kargs to "name.pth"
        """
        
        # save model, optimizer, scheduler and other arguments
        data = {}
        data['model'] = self.model.state_dict()
        if self.optimizer is not None:
            data['optimizer'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data['scheduler'] = self.scheduler.state_dict()
            
        # save any other arguments
        data.update(self.args)
        
        save_file = os.path.join(self.save_dir, '{}.pth'.format(name))
        torch.save(data, save_file)
        self.update_checkpoint(save_file)
    
    def load(self, f=None):
        if self.has_checkpoint():
            # there is a checkpoint
            f = self.get_checkpoint_file()
        if not f:
            print("No checkpoint found.")
            return {}
        print("Loading checkpoint from {}".format(f))
        # load the checkpoint dictionary
        checkpoint = self._load_file(f)
        
        # load model weight to the model
        self.model.load_state_dict(checkpoint.pop('model'))
        # self._load_model(checkpoint)
        if 'optimizer' in checkpoint and self.optimizer:
            # if there is a optimizer, load state dict into the optimizer
            print("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop('optimizer'))
        if 'scheduler' in checkpoint and self.scheduler:
            # if there is a optimizer, load state dict into it
            print("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop('scheduler'))
            
        # there might be other arguments to be saved
        self.args.update(checkpoint)
    
    def has_checkpoint(self):
        """
        Check whether the 'checkpoint.pkl' file exists
        :return:
        """
        save_file = os.path.join(self.save_dir, 'checkpoint.pkl')
        return os.path.exists(save_file)
    
    def get_checkpoint_file(self):
        """
        Get the last checkpoint file path.
        """
        save_file = os.path.join(self.save_dir, 'checkpoint.pkl')
        checkpoints = pickle.load(open(save_file, 'rb'))
        last_saved = os.path.join(self.save_dir, checkpoints[-1])
        return last_saved
    
    def update_checkpoint(self, last_filename):
        """
        Update 'checkpoint.pkl'. We will only keep a number of checkpoints.
        """
        save_file = os.path.join(self.save_dir, 'checkpoint.pkl')
        if os.path.exists(save_file):
            checkpoints = pickle.load(open(save_file, 'rb'))
        else:
            checkpoints = []
            
        checkpoints.append(os.path.basename(last_filename))
        if len(checkpoints) > self.max_checkpoints:
            # if number exceeds threshold, pop first file name, delete it.
            checkpoint_name = checkpoints.pop(0)
            checkpoint = os.path.join(self.save_dir, checkpoint_name)
            if os.path.exists(checkpoint) and checkpoint_name not in checkpoints:
                os.remove(checkpoint)
        pickle.dump(checkpoints, open(save_file, 'wb'))
        
    def _load_file(self, f):
        return torch.load(f, map_location=torch.device('cpu'))


