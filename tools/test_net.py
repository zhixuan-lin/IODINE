import sys
import torch
import os
import argparse

if not '.' in sys.path:
    sys.path.insert(0, '.')

from lib.config.parse import parse
from lib.data import make_dataloader
from lib.eval import make_evaluator
from lib.modeling import make_model
from lib.engine.eval import evaluate
from lib.utils.checkpoint import Checkpointer
from lib.config import cfg


def test_net(cfg):
    """
    General test procedure
    """
    
    # model
    device = torch.device(cfg.MODEL.DEVICE)
    model = make_model(cfg)
    # model = model.to(device)
    
    dataloader = make_dataloader(cfg, mode='test')
    evaluator = make_evaluator(cfg)
    
    # checkpointer
    save_dir = os.path.join(cfg.MODEL_DIR, cfg.EXP.NAME)
    checkpointer = Checkpointer(
        model=model,
        optimizer=None,
        scheduler=None,
        args={},
        max_checkpoints=cfg.TRAIN.NUM_CHECKPOINTS,
        save_dir=save_dir
    )
    checkpointer.load()
    
    print()
    print('-' * 80)
    print('Testing model "{}" on "{}"...'.format(cfg.MODEL.NAME, cfg.DATASET.TEST))
    evaluate(
        model,
        device,
        dataloader,
        evaluator
    )
    print('-' * 80)
    

def main():
    parse(cfg)
    test_net(cfg)


if __name__ == '__main__':
    main()









