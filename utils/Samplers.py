import math
import torch
import pandas as pd
from torch import pi as pi
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def create_flattened_coords(shape) -> torch.Tensor:
    parameter = []
    dim = 1
    for i in range(len(shape)):
        minimum,maximum,num = shape[i]
        parameter.append(torch.linspace(minimum,maximum,num))
        dim *= num
    coords = torch.stack(torch.meshgrid(parameter, indexing='ij'),axis=-1)
    flattened_coords = coords.reshape(dim,len(shape))
    return flattened_coords

def normalize_data(data, scale_min:float=None, scale_max:float=None, data_min:float=None, data_max:float=None):
    if data_min==None or data_max==None:
        data_min, data_max = data.min(), data.max()
    if scale_min==None or scale_max==None:
        scale_min, scale_max = float(data_min), float(data_max)
    if data_max!=data_min:
        data = (data - data_min)/(data_max - data_min)
        data = data*(scale_max - scale_min) + scale_min
    side_info = {'scale_min':float(scale_min), 'scale_max':float(scale_max), 'data_min':float(data_min), 'data_max':float(data_max)}
    return data, side_info

def invnormalize_data(data, scale_min, scale_max, data_min, data_max):
    if scale_min!=None and scale_max!=None and scale_max!=scale_min:
        data = (data - scale_min)/(scale_max - scale_min)
        data = data*(data_max - data_min) + data_min
    return data

class Dataset(object):
    def __init__(self, dataset_path, input_list, output_list):
        self.dataset_path = dataset_path
        self.input_list, self.output_list = input_list, output_list
        self.df = self._load_dataset()
        self.df_train, self.df_test = self._split_dataset()

    def _load_dataset(self):
        df = pd.read_csv(self.dataset_path)
        return df
    
    def _split_dataset(self, test_size=0.2, random_state=1):
        df_train, df_test = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return df_train, df_test

    def parser(self, is_shuffle=True, random_state=0):
        X_train = self.df_train[self.input_list].to_numpy()
        y_train = self.df_train[self.output_list].to_numpy().reshape(-1,)
        X_test = self.df_test[self.input_list].to_numpy()
        y_test = self.df_test[self.output_list].to_numpy().reshape(-1,)
        X_train, y_train = shuffle(X_train, y_train, random_state=random_state)
        return torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

class BaseSampler:
    def __init__(self, batch_size: int, epochs:int, device:str='cpu'):
        self.batch_size = int(batch_size)
        self.epochs = epochs
        self.device = device
        self.evaled_epochs = []
        self.last_epochs = 0
        self.input, self.label = torch.zeros((1,1)), torch.zeros((1,1))
        self.pop_size = self.label.shape[0]

    def judge_eval(self, eval_epoch):
        if self.epochs_count%eval_epoch==0 and self.epochs_count!=self.last_epochs and not (self.epochs_count in self.evaled_epochs):
            self.evaled_epochs.append(self.epochs_count)
            return True
        elif self.index>=self.pop_size and self.epochs_count>=self.epochs-1:
            self.epochs_count = self.epochs
            return True
        else:
            return False

    def __len__(self):
        return max((self.epochs-self.last_epochs), 0)*math.ceil(self.pop_size/self.batch_size)

    def __iter__(self):
        self.index = 0
        self.epochs_count = self.last_epochs
        return self

    def __next__(self):
        if self.index < self.pop_size:
            sampled_idxs = torch.randint(0, self.pop_size, (self.batch_size,))
            sampled_input = self.input[sampled_idxs, :]
            sampled_label = self.label[sampled_idxs, :]
            self.index += self.batch_size
            return sampled_input, sampled_label
        elif self.epochs_count < self.epochs-1:
            self.epochs_count += 1
            self.index = 0
            return self.__next__()
        else:
            raise StopIteration
    
class DimlessSampler(BaseSampler):
    def __init__(self, batch_size:int=512, epochs:int=100, device:str='cpu', norm:bool=True, data_path:str='dataset/keyhole.csv', 
            input_list:list=['etaP', 'Vs', 'r0', 'alpha', 'rho', 'cp', 'Tl-T0'], output_list:list=['e*'], **kwargs) -> None:
        super().__init__(batch_size=batch_size, epochs=epochs, device=device)
        self.device = device
        self.input, self.latent, self.label = self.load_dataset(data_path, input_list, output_list)
        # output normalization for better regression
        self.output_norm = float(abs(self.label).max()) if norm else 1.
        self.label = self.label/self.output_norm
        self.pop_size = self.label.shape[0]
    
    def load_dataset(self, data_path, input_list, output_list):
        df = pd.read_csv(data_path)
        input = torch.tensor(df[input_list].to_numpy(), dtype=torch.float32).to(self.device)
        label = torch.tensor(df[output_list].to_numpy().reshape(-1,), dtype=torch.float32).unsqueeze(-1).to(self.device)
        return input, input, label
    
    def input2latent(self, latent_module, latent_norm):
        '''change z and keep x, y unchanged. y has been normalized before'''
        # latent varible: Zi=\prod Xj^Wij, W may not be the true values
        self.latent = latent_module(self.input) 
        # latent normalization for better regression
        self.latent_norm = torch.tensor([1.]*self.latent.shape[-1]) if not latent_norm else \
            torch.tensor([1./abs(self.latent[:,i]).max() for i in range(self.latent.shape[-1])]) 
        self.latent = self.latent*self.latent_norm                
    
    def invnorm(self, express_norm, pr_module):
        ''' norm: Zi'=Zi/abs(Zi).max(), y'=f'(Zi'), y'=y/abs(y).max()
            invnorm: y=f'(Zi/abs(Zi).max())*abs(y).max()=f(Zi)
        '''
        latent_scale, output_scale = self.latent_norm.unsqueeze(0), self.output_norm
        express_scale = pr_module.poly.fit_transform(latent_scale)[0]
        express = express_scale*express_norm*output_scale
        return list(express)
        
    def __next__(self):
        if self.index < self.pop_size:
            sampled_idxs = torch.randint(0, self.pop_size, (self.batch_size,))
            sampled_latent = self.latent[sampled_idxs, :]
            sampled_label = self.label[sampled_idxs, :]
            self.index += self.batch_size
            return sampled_latent, sampled_label
        elif self.epochs_count < self.epochs-1:
            self.epochs_count += 1
            self.index = 0
            return self.__next__()
        else:
            raise StopIteration