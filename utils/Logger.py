import os
import sys
import time
import torch
import random
import numpy as np
from typing import Dict
from torch.utils.tensorboard import SummaryWriter
timestamp = time.strftime("_%Y_%m%d_%H%M%S")

class MyLogger():
    def __init__(self, outputs_dir:str='outputs', project_name:str='project',
                time:bool=True, stdlog:bool=True, tensorboard:bool=True, dirs:list[bool]=[True, False, True, True]):
        self.stdlog = stdlog
        self.tensorboard = tensorboard
        self.project_dir = os.path.join(outputs_dir, project_name)
        self.project_dir = self.project_dir + timestamp if time else self.project_dir
        self.dir_init(dirs)
        self.stdlog_init() if stdlog else None
        self.tensorboard_init() if tensorboard else None
        
    def stdlog_init(self):
        sys.stderr=open(os.path.join(self.logger_dir,'stderr.log'), 'w')
        
    def tensorboard_init(self,):
        self.tblogger = SummaryWriter(self.logger_dir, flush_secs=30)
    
    def dir_init(self, dirs:list[bool]=[True, True, True, True]):
        self.script_dir = os.path.join(self.project_dir, 'scripts')
        self.model_dir = os.path.join(self.project_dir, 'models')
        self.metric_dir = os.path.join(self.project_dir, 'metrics')
        self.logger_dir = os.path.join(self.project_dir, 'logger')
        os.makedirs(self.project_dir) if not os.path.exists(self.project_dir) else None
        os.mkdir(self.script_dir) if (not os.path.exists(self.script_dir) and dirs[0]) else None
        os.mkdir(self.model_dir) if (not os.path.exists(self.model_dir) and dirs[1]) else None
        os.mkdir(self.metric_dir) if (not os.path.exists(self.metric_dir) and dirs[2]) else None
        os.mkdir(self.logger_dir) if (not os.path.exists(self.logger_dir) and dirs[3]) else None

    def log_metrics(self, metrics_dict: Dict[str, float], iters):
        for k in metrics_dict.keys():
            self.tblogger.add_scalar(k, metrics_dict[k], iters) if self.tensorboard else None

    def close(self):
        self.tblogger.close() if self.tensorboard else None

def reproduc(seed:int=42):
    """Make experiments reproducible
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True