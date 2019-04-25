from .mnist import MNIST
from torch.utils.data import DataLoader
def make_dataloader(cfg, mode):
    if mode == 'train':
        batch_size = cfg.TRAIN.BATCH_SIZE
        shuffle = True
    elif mode == 'val':
        batch_size = cfg.VAL.BATCH_SIZE
        shuffle = False
    elif mode == 'test':
        batch_size = cfg.TEST.BATCH_SIZE
        shuffle = False
    
    # build dataset
    dataset = make_dataset(cfg, mode)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return dataloader
    

def make_dataset(cfg, mode):
    return MNIST('data/MNIST', mode)
