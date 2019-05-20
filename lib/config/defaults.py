import os
from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# PATH SETTING
# -----------------------------------------------------------------------------
_C.PATH = CN()
_C.PATH.CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_C.PATH.LIB_DIR = os.path.dirname(_C.PATH.CONFIG_DIR)
_C.PATH.ROOT_DIR = os.path.dirname(_C.PATH.LIB_DIR)
_C.PATH.DATA_DIR = os.path.join(_C.PATH.ROOT_DIR, 'data')

# -----------------------------------------------------------------------------
# EXP
# -----------------------------------------------------------------------------
_C.EXP = CN()
_C.EXP.NAME = 'test'

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# this specifies the model to use
_C.MODEL.NAME = "VAE"
# _C.MODEL.TEST = False
_C.MODEL.DEVICE = "cpu"
_C.MODEL.PARALLEL = False
_C.MODEL.DEVICE_IDS = []

# -----------------------------------------------------------------------------
# ARCHITECTURE
# -----------------------------------------------------------------------------
_C.ARCH = CN()

# Number of iteration
_C.ARCH.ITERS = 5
# Number of slots
_C.ARCH.SLOTS = 7
# Global, fixed sigma for evaluating data likelihood
_C.ARCH.SIGMA = 0.13
# Latent variable dimension
_C.ARCH.DIM_LATENT = 128
# Input image size (must be a multiple of 8)
_C.ARCH.IMG_SIZE = 32
# Input channels (gray or color)
_C.ARCH.IMG_CHANNELS = 3

# Layer normalization
_C.ARCH.LAYERNORM = True

# Stop gradient?
_C.ARCH.STOP_GRADIENT = False

# Input encodings
_C.ARCH.ENCODING = [
    # input
    'image',
    # predicted means
    'means',
    # mask
    'mask',
    # mask logits
    'mask_logits',
    # gradient of means
    'grad_means',
    # gradient of mask
    'grad_mask',
    # gradients of posterior parameters
    'grad_post',
    # posterior
    'posterior',
    # mask posterior (not implemented)
    'mask_posterior',
    # likelihood
    'likelihood',
    # leave-one-out likelihood
    'leave_one_out_likelihood',
]

# Refinement network
_C.ARCH.REF = CN()

# Refinement network Conv channels
_C.ARCH.REF.CONV_CHAN = 64
# Refinement network MLP units
_C.ARCH.REF.MLP_UNITS = 256
_C.ARCH.REF.KERNEL_SIZE = 3

# Decoder
_C.ARCH.DEC = CN()

# Decoder network Conv channels
_C.ARCH.DEC.CONV_CHAN = 64
_C.ARCH.DEC.KERNEL_SIZE = 3




# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.TRAIN = 'MNIST'
_C.DATASET.VAL = 'MNIST'
_C.DATASET.TEST = 'MNIST'


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4

# ---------------------------------------------------------------------------- #
# Specific train options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()

# if true, the model, optimizer, schduler will be loaded
_C.TRAIN.RESUME = False

# number of epochs
_C.TRAIN.MAX_EPOCHS = 30

# batch size
_C.TRAIN.BATCH_SIZE = 128

# use Adam as default
_C.TRAIN.BASE_LR = 0.001
_C.TRAIN.WEIGHT_DECAY = 0.0005

_C.TRAIN.CHECKPOINT_PERIOD = 2500
_C.TRAIN.NUM_CHECKPOINTS = 3
_C.TRAIN.PRINT_EVERY = 100
_C.TRAIN.VAL_EVERY = 1000

# ---------------------------------------------------------------------------- #
# Validation settings
# ---------------------------------------------------------------------------- #
_C.VAL = CN()

# validation on?
_C.VAL.IS_ON = False
_C.VAL.BATCH_SIZE = 1
_C.VAL.EVALUATOR = ''


# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 32
_C.TEST.EVALUATOR = ''

# ---------------------------------------------------------------------------- #
# Tensorboard
# ---------------------------------------------------------------------------- #
_C.TENSORBOARD = CN()
_C.TENSORBOARD.IS_ON = True
_C.TENSORBOARD.TARGETS = CN()
_C.TENSORBOARD.TARGETS.SCALAR = ["loss"]
_C.TENSORBOARD.TARGETS.IMAGE = ['image', 'pred']
_C.TENSORBOARD.LOG_DIR = os.path.join(_C.PATH.ROOT_DIR, "logs")


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# default model saving directory
_C.MODEL_DIR = os.path.join(_C.PATH.DATA_DIR, "model")
_C.GETTER = 'VAE'


# ---------------------------------------------------------------------------- #
# Path setups
# ---------------------------------------------------------------------------- #
import sys
import os

if _C.PATH.ROOT_DIR not in sys.path:
    sys.path.append(_C.PATH.ROOT_DIR)

if not os.path.exists(_C.MODEL_DIR):
    os.makedirs(_C.MODEL_DIR)

# clear log if not resume
logdir = os.path.join(_C.TENSORBOARD.LOG_DIR, _C.EXP.NAME)
if os.path.exists(logdir):
    os.system('rm -r {}'.format(logdir))

# import warnings
# warnings.filterwarnings("ignore")
