import torch
import numpy as np
from collections import deque
from collections import defaultdict

class SmoothedValue:
    def __init__(self, window_size=20):
        self.values = deque(maxlen=window_size)
        self.total = 0
        self.count = 0
        
    def update(self, value):
        self.values.append(value)
        self.total += value
        self.count += 1
        
    @property
    def median(self):
        values = np.array(self.values)
        return np.median(values)
    
    @property
    def global_avg(self):
        values = np.array(self.values)
        return np.mean(values)
    
class MetricLogger:
    def __init__(self, delimiter='\t'):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        
    def update(self, **kargs):
        for k, v in kargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)
            
    def __getitem__(self, key):
        return self.meters[key]
    
    def __str__(self):
        string = []
        for name in self.meters:
            string.append(
            '{}: {.4f}'.format(name, self.meters[name].median))
            
        return self.delimiter.join(string)
        
        
    
