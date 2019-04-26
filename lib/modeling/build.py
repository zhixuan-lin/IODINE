import torch
from .vae import VAE
from .iter_net import IterNet

def make_model(cfg):
    device = torch.device(cfg.MODEL.DEVICE)
    device_ids = cfg.MODEL.DEVICE_IDS
    if not device_ids: device_ids = None # use all devices
    model = _make_model(cfg)
    model = model.to(device)
    if cfg.MODEL.PARALLEL:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    return model

def _make_model(cfg):
    if cfg.MODEL.NAME == 'VAE':
        return VAE(28 * 28, 128)
    elif cfg.MODEL.NAME == 'Iter':
        return IterNet(28 * 28, 128)
