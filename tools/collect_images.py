import os
import sys
from skimage import io
from tqdm import tqdm
import numpy as np

if not '.' in sys.path:
    sys.path.insert(0, '.')
from lib.config.parse import parse
from lib.modeling import make_model
from lib.solver import make_optimizer, make_scheduler
from lib.data import make_dataloader
from lib.utils.checkpoint import Checkpointer
from lib.config import cfg

dirs = {
    'img': '0_img',
    'recon': '1_recon',
    'fg': '2_fg',
    'bg': '3_bg',
    'img_box': '4_img_box',
    'fg_box': '5_fg_box',
    'recon_box': '6_recon_box',
    'fg_mask': '7_fg_mask',
    'masks': '8_masks',
    'comps': '9_comps',
    'masked_comps': '10_masked_comps',
}

def collect_images(cfg, num=10):
    """
    Collect result images into a directory

    :param cfg
    :return:
    """
    out = os.path.join(cfg.IMAGE_DIR, cfg.EXP.NAME)
    model = make_model(cfg)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_scheduler(cfg, model)
    valloader = make_dataloader(cfg, mode='val')

    # checkpointer
    args = {'epoch': 0, 'iter': 0}
    save_dir = os.path.join(cfg.MODEL_DIR, cfg.EXP.NAME)
    checkpointer = Checkpointer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args,
        max_checkpoints=cfg.TRAIN.NUM_CHECKPOINTS,
        save_dir=save_dir
    )
    checkpointer.load()

    if not os.path.exists(out):
        os.makedirs(out)

    model.eval()

    count = 0

    pbar = tqdm(total=num)

    for data in valloader:
        data = data[0]
        imgs = data.to(cfg.MODEL.DEVICE)
        B, C, H, W = imgs.size()

        recon, masks, comps = model.reconstruct(imgs)

        imgs = imgs.cpu().detach()
        recon = recon.cpu().detach()
        comps = comps.cpu().detach()
        masks = masks.cpu().detach().expand_as(comps)
        masked_comps = comps * masks

        results = {
            'img': imgs,
            'recon': recon,
            'masks': masks,
            'comps': comps,
            'masked_comps': masked_comps,
        }


        # Save results
        for i in range(B):
            item_root = os.path.join(out, '{:04}'.format(count))
            if not os.path.exists(item_root):
                os.mkdir(item_root)
            for key in results:
                if key in ['masks', 'comps', 'masked_comps']:
                    folder = os.path.join(item_root, dirs[key])
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    for k in range(cfg.ARCH.SLOTS):
                        img = results[key][i][k]
                        img = img.permute(1, 2, 0).detach().cpu().numpy()
                        path = os.path.join(folder, '{}.png'.format(k))
                        io.imsave(path, (img * 255.0).astype(np.uint8))
                else:
                    img = results[key][i]
                    img = img.permute(1, 2, 0).detach().cpu().numpy()
                    path = os.path.join(item_root, '{}.png'.format(dirs[key]))
                    io.imsave(path, (img * 255.0).astype(np.uint8))

            count += 1
            pbar.update(1)
            if count >= num:
                model.train()
                return

if __name__ == '__main__':
    parse(cfg)
    collect_images(cfg)
