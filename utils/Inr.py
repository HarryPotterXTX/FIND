import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import PolynomialFeatures 
import copy, sys, torch, math, os

from utils.HOPE.Network import MLP
from utils.Samplers import BaseSampler
from utils.HOPE.Global import convert_derivatives
from utils.HOPE.HopeGrad import hopegrad, cleargrad

def cal_r2(y_true, y_hat):
    mean = y_true.mean()
    mse = ((y_true - y_hat)**2).mean()
    var = ((y_true - mean)**2).mean()
    r2 = 1 - mse/var
    return r2

class INRSampler(BaseSampler):
    def __init__(self, x:torch.tensor, y:torch.tensor, batch_size:int=512, epochs:int=20, device:str='cpu') -> None:
        super().__init__(batch_size=batch_size, epochs=epochs, device=device)
        self.input, self.label = x, y
        self.pop_size = self.label.shape[0]
    
    def to_device(self, device):
        self.input = self.input.to(device)
        self.label = self.label.to(device)
        
class INR():
    '''Implicit Neural Representation'''
    def __init__(self, degree:int=10) -> None:
        self.degree = degree
        self.poly = PolynomialFeatures(degree=degree) 
        self.set_gpu()
    
    def set_gpu(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    
    def epochs_thres(self, epochs_ratio, r2_thres):
        if self.sampler.epochs_count>self.sampler.epochs*epochs_ratio and self.r2<r2_thres:
            return True
        return False

    def fit(self, x, y):
        # hyper parameters
        hidden, layer, act = 64, 4, 'Sine'
        lr, batch_size, epochs = 1e-3, 65536, 4000
        device = 'cuda'
        thres_list = [[1./16, 0.001], [2./16, 0.25], [3./16, 0.3], [4./16, 0.35], [8./16, 0.5]]
        # prepare to train
        x, y = x.to(device), y.to(device)
        if x.shape[-1]>1:
            raise NotImplemented
        self.net = MLP(input=x.shape[-1], hidden=hidden, output=y.shape[-1], layer=layer, act=act, output_act=False).to(x.device)
        self.optimizer = torch.optim.Adamax([{'params':self.net.parameters()}], lr=lr)
        self.sampler = INRSampler(x, y, batch_size=batch_size, epochs=epochs, device=x.device)
        # train 
        pbar = tqdm(self.sampler, desc='Training', leave=False, file=sys.stdout)
        self.r2, self.best_net, flag = 0, copy.deepcopy(self.net), False
        for step, (sampled_x, sampled_y) in enumerate(pbar): 
            self.optimizer.zero_grad()
            y_hat = self.net(sampled_x)
            loss = torch.mean((y_hat - sampled_y)**2)
            loss.backward()
            self.optimizer.step()
            pbar.set_postfix_str("loss={:.6f}, r2={:.6f}".format(loss.item(), self.r2))
            pbar.update(1)
            if self.sampler.judge_eval(100):
                y_hat_test = self.net(self.sampler.input)
                y_test = self.sampler.label
                r2_ = cal_r2(y_test, y_hat_test)
                if r2_ > self.r2:
                    self.r2 = float(r2_.cpu().detach().numpy())
                    self.best_net = copy.deepcopy(self.net)
                for (epochs_ratio, r2_thres) in thres_list:
                    if self.epochs_thres(epochs_ratio, r2_thres):
                        flag = True
                if flag == True:
                    break
        ref_in = torch.tensor([[(self.sampler.input[:,i].min()+self.sampler.input[:,i].max())/2 for i in range(self.sampler.input.shape[-1])]]).to(x.device)
        ref_in.requires_grad = True
        ref_out = self.best_net(ref_in)
        hopegrad(y=ref_out, order=self.degree, mixed=0)
        v = {k: list(ref_in.hope_grad[k].detach().cpu().reshape(-1).numpy()) for k in ref_in.hope_grad.keys()}
        v[0] = [float(ref_out.detach().cpu().numpy()[0,0])]
        v = convert_derivatives(grad_dict=v, x0=ref_in.detach().cpu().numpy(), x1=np.zeros(ref_in.shape)) 
        self.express = []
        for k in range(self.degree+1):
            self.express = self.express + [c/math.factorial(k) for c in v[k]]
        self.express = [float(c) for c in self.express]
        # release resources
        cleargrad(ref_out)
        self.sampler.to_device('cpu')
        self.net, self.best_net = self.net.to('cpu'), self.best_net.to('cpu')
        x, y = x.to('cpu'), y.to('cpu')
        return self.express, self.r2