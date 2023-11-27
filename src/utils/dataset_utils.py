import re
import json


import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
from copy import deepcopy
from typing import TypeVar, Iterable, List, Union, Any


T = TypeVar('T')

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray) or isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NpEncoder, self).default(obj)