import torch
from .vae import VAE

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
    return VAE(28 * 28, 128)
