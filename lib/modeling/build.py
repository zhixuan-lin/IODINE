import torch
from .vae import VAE
from lib.modeling.iter_nets.iter_net import IterNet
from lib.modeling.iter_nets.iter_net_v1 import IterNetV1
from lib.modeling.iter_nets.iter_net_v2 import IterNetV2
from lib.modeling.iter_nets.iter_net_v3 import IterNetV3
from lib.modeling.iter_nets.iter_net_v4 import IterNetV4
from lib.modeling.iter_nets.iter_net_v5 import IterNetV5
from lib.modeling.iter_nets.iter_net_v6 import IterNetV6
from lib.modeling.iter_nets.iter_net_v7 import IterNetV7
from lib.modeling.iter_nets.iter_net_v8 import IterNetV8
from lib.modeling.iter_nets.iter_net_v9 import IterNetV9
from lib.modeling.iter_nets.iter_net_v10 import IterNetV10
from lib.modeling.iter_nets.iter_net_v11 import IterNetV11
from lib.modeling.iter_nets.iter_net_v12 import IterNetV12
from lib.modeling.iter_nets.iter_net_v13 import IterNetV13
from lib.modeling.iter_nets.iter_net_v14 import IterNetV14
from lib.modeling.iter_nets.iter_net_v15 import IterNetV15
from lib.modeling.iter_nets.iter_net_v16 import IterNetV16
from lib.modeling.iter_nets.iter_net_v17 import IterNetV17
from lib.modeling.iter_nets.iter_net_v18 import IterNetV18
from lib.modeling.iter_nets.iter_net_v19 import IterNetV19

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
    elif cfg.MODEL.NAME == 'IterV1':
        return IterNetV1(28 * 28, 128)
    elif cfg.MODEL.NAME == 'IterV2':
        return IterNetV2(28 * 28, 128)
    elif cfg.MODEL.NAME == 'IterV3':
        return IterNetV3(28 * 28, 128)
    elif cfg.MODEL.NAME == 'IterV4':
        return IterNetV4(28 * 28, 128)
    elif cfg.MODEL.NAME == 'IterV5':
        return IterNetV5(28 * 28, 128)
    elif cfg.MODEL.NAME == 'IterV6':
        return IterNetV6(28 * 28, 128)
    elif cfg.MODEL.NAME == 'IterV7':
        return IterNetV7(28 * 28, 128)
    elif cfg.MODEL.NAME == 'IterV8':
        return IterNetV8(28 * 28, 128)
    elif cfg.MODEL.NAME == 'IterV9':
        return IterNetV9(28 * 28, 128)
    elif cfg.MODEL.NAME == 'IterV10':
        return IterNetV10(28 * 28, 64)
    elif cfg.MODEL.NAME == 'IterV11':
        return IterNetV11(28 * 28, 64)
    elif cfg.MODEL.NAME == 'IterV12':
        return IterNetV12(28 * 28, 64)
    elif cfg.MODEL.NAME == 'IterV13':
        return IterNetV13(28 * 28, 64)
    elif cfg.MODEL.NAME == 'IterV14':
        return IterNetV14(28 * 28, 64)
    elif cfg.MODEL.NAME == 'IterV15':
        return IterNetV15(28 * 28, 64)
    elif cfg.MODEL.NAME == 'IterV16':
        return IterNetV16(28 * 28, 64)
    elif cfg.MODEL.NAME == 'IterV17':
        return IterNetV17(28 * 28, 64)
    elif cfg.MODEL.NAME == 'IterV18':
        return IterNetV18(28 * 28, 64)
    elif cfg.MODEL.NAME == 'IterV19':
        return IterNetV19(28 * 28, 64)
