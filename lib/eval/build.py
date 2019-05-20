import torch
import numpy as np
from math import sqrt
from .ari_eval import ARIEvaluator

def make_evaluator(cfg):
    return ARIEvaluator()

