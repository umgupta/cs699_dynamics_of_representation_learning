import logging
import random

import numpy
import torch

logger = logging.getLogger()


def set_seed(seed):
    # this may still not guarantee deterministic behaviour
    # see https://discuss.pytorch.org/t/how-to-get-deterministic-behavior/18177/12

    logger.debug(f"Setting seed to {seed}")
    if isinstance(seed, list):
        torch_seed, numpy_seed, random_seed = seed
    else:
        torch_seed, numpy_seed, random_seed = seed, seed, seed

    torch.manual_seed(torch_seed)
    numpy.random.seed(numpy_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
