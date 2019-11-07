import sys
import torch
import os

if not '.' in sys.path:
    sys.path.insert(0, '.')

from lib.config.parse import parse
from lib.data import make_dataloader
from lib.eval import make_evaluator
from lib.modeling import make_model
from lib.solver import make_optimizer, make_scheduler
from lib.engine.train import train
# from lib.engine.evaluator import evaluate
from lib.utils.checkpoint import Checkpointer
from lib.utils.tensorboard import TensorBoard
from lib.utils.vis_logger import make_getter
from lib.config import cfg

def train_net(cfg):

    """
    General training procedure
    """

    # model
    model = make_model(cfg)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_scheduler(cfg, model)
    dataloader = make_dataloader(cfg, mode='train')

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
    if cfg.TRAIN.RESUME:
        checkpointer.load()

    # tensorboard and visualization
    tensorboard = None
    getter = None
    logdir = os.path.join(cfg.TENSORBOARD.LOG_DIR, cfg.EXP.NAME)
    if cfg.TENSORBOARD.IS_ON:
        tensorboard = TensorBoard(
            logdir=logdir,
            scalars=cfg.TENSORBOARD.TARGETS.SCALAR,
            images=cfg.TENSORBOARD.TARGETS.IMAGE,
            resume=cfg.TRAIN.RESUME
        )
        getter = make_getter(cfg)

    # validation
    dataloader_val = make_dataloader(cfg, 'val')
    evaluator = None
    # training parameters
    params = {
        'max_epochs': cfg.TRAIN.MAX_EPOCHS,
        'checkpoint_period': cfg.TRAIN.CHECKPOINT_PERIOD,
        'print_every': cfg.TRAIN.PRINT_EVERY,
        'val_every': cfg.TRAIN.VAL_EVERY,
        'val_num_batches': cfg.VAL.NUM_BATCHES
    }

    train(
        model,
        optimizer,
        dataloader,
        cfg.MODEL.DEVICE,
        params,
        checkpointer=checkpointer,
        tensorboard=tensorboard,
        getter=getter,
        dataloader_val=dataloader_val,
        evaluator=evaluator
    )

    return model

if __name__ == '__main__':
    parse(cfg)
    train_net(cfg)
