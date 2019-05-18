from .mnist import MNIST
from torch.utils.data import DataLoader
from .clevr import CLEVR
from .dsprite import MultiDSprites
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
    if cfg.DATASET.TRAIN == 'MNIST':
        return MNIST('data/MNIST', mode)
    elif cfg.DATASET.TRAIN == 'CLEVR':
        return CLEVR('data/CLEVR/images', mode)
    elif cfg.DATASET.TRAIN == 'DSPRITES':
        return MultiDSprites('data/DSPRITES', mode)
