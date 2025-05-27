import numpy as np
import pandas as pd
from tqdm import tqdm
from sympy import Matrix
from pysr import PySRRegressor
from itertools import permutations
from sklearn.linear_model import LinearRegression 
from itertools import combinations_with_replacement
from sklearn.preprocessing import PolynomialFeatures 
import os, copy, time, sys, torch

from utils.Inr import INR, cal_r2
from utils.Samplers import DimlessSampler, create_flattened_coords

sub = {i:chr(ord('\u2080')+i) for i in range(10)}
sup =  {**{0:'\u2070'},**{1:'\u00B9',2:'\u00B2',3:'\u00B3'},**{i:chr(ord('\u2070')+i) for i in range(4,10)}}
for i in range(10,31):
    sub[i] = sub[i//10] + sub[i%10]
    sup[i] = sup[i//10] + sup[i%10]
sup[1] = ''
sup_map = str.maketrans("0123456789+-.", "\u2070\u00B9\u00B2\u00B3"+"".join([chr(ord('\u2070')+i) for i in range(4,10)])+"\u207A\u207B\u02D9")
sub_map = str.maketrans("0123456789", ''.join([chr(ord('\u2080')+i) for i in range(10)]))
def sub_sup(var:str='x', sub:str='', sup:str=''):
    if float(sup)!=0:
        sub = str(sub).translate(sub_map)
        sup = str(sup).translate(sup_map) if float(sup)!=1 else ''
        return var+sub+sup
    else:
        return ''

class LatentLayer(torch.nn.Module):
    ''' X\in R^p, Z\in R^m, W\in R^{m*p}, Zi=\prod Xj^Wij
        dimensional consistency: D*Wi=d -> Wi^T=E*li+e (D*E=0, D*e=d)
        connection relationship: E0*li+e0=0, E1*li+e1!=0 -> li=Fi*mui+fi (E0*Fi=0, E0*fi=-e0)
        in summary: Wi^T=E*(Fi*mui+fi)+e, s.t. E1*(Fi*mui+fi)+e1!=0
    '''
    def __init__(self, D:torch.tensor, d:torch.tensor, latent_dim:int=1, latent_connect:list=[], latent_ratio:list=[], coef_previous:list=[]):
        super().__init__()
        self.D = D                          # input unit matrix
        self.d = d                          # output unit matrix
        self.input_dim = self.D.shape[-1]   # input dimension: X=[X1,X2,...,Xp]
        self.latent_dim = latent_dim        # latent dimension: Z=[Z1,Z2,...,Zm]
        # DWi=d -> Wi^T=E*li+e (D*E=0, D*e=d)
        self.E, self.e = self.solver(self.D, self.d)
        print('basis matrix V:\n'+str(self.E))
        print('bias vector v:\n'+str(self.e))
        # E0*li+e0=0, E1*li+e1!=0 -> li=F*mui+f (E0*F=0, E0*f=-e0)
        self.Fs, self.fs, self.Ee1, self.mus_idxs = self.latent_prior(latent_connect, latent_ratio, coef_previous)

    def latent_prior(self, latent_connect, latent_ratio, coef_previous):
        """latent_connect: [[1,2],[3,4],[1,5]]: x1,x2->z1, x3,x4->z2, x1,x5->z3
            x1,x2->z1, weights for x3,x4,x5 are all 0."""
        self.latent_connect = latent_connect + [[i for i in range(1,self.input_dim+1)]]*(self.latent_dim-len(latent_connect)) \
            if len(latent_connect)<self.latent_dim else latent_connect
        self.latent_ratio = latent_ratio + [[1]]*(self.latent_dim-len(latent_ratio)) \
            if len(latent_ratio)<self.latent_dim else latent_ratio
        assert len(self.latent_connect)==self.latent_dim and len(self.latent_ratio)==self.latent_dim, "Incorrect latent prior!"
        Fs, fs, Ee1, mus_idxs, mu_idx = [], [], [], [], 0
        for connect in self.latent_connect:
            idx0, idx1 = np.array([i for i in range(1,self.input_dim+1) if i not in connect])-1, np.array(connect)-1
            E0, E1, e0, e1 = self.E[idx0], self.E[idx1], self.e[idx0], self.e[idx1]
            F, f = self.solver(E0, -e0)
            Fs.append(F), fs.append(f), Ee1.append([E1, e1]), mus_idxs.append([mu_idx,mu_idx+F.shape[-1]])
            mu_idx = mus_idxs[-1][-1]
        for i in range(min(len(coef_previous),len(latent_connect))):
            assert len(coef_previous[i])==Fs[i].shape[-1], "The latent variables previously discovered do not satisfy the connectivity prior!"
        self.coef_previous = list(np.array(coef_previous).reshape(-1)) if coef_previous!=[] else []
        self.coef_dim = mus_idxs[-1][-1]
        assert self.coef_dim>len(self.coef_previous), "The number of latent variables to be searched is 0!"
        return Fs, fs, Ee1, mus_idxs

    def solver(self, M, m):
        """reduced row echelon form to get solution for Mx=m: x=V*l+v (MV=0, Mv=m)"""
        # bias vector: Mv=m
        _, col = M.shape
        MA = torch.concat([M,m],dim=1)
        rM, rMA = torch.linalg.matrix_rank(M), torch.linalg.matrix_rank(MA)
        assert rM==rMA, "There's no solution for equation Mx=m."
        MA_rref, pivot_idx = Matrix(MA).rref()
        MA_rref = np.array(MA_rref.tolist()).astype(np.float32)
        bias_vector = np.zeros((col,1))
        for r in range(len(pivot_idx)): # the pivot of r-th row is pivot[r]
            bias_vector[pivot_idx[r],0] = MA_rref[r,-1]/(MA_rref[r,pivot_idx[r]])
        assert abs(np.matmul(M, bias_vector)-m).sum()<1e-6, "The bias vectors v can't satisfy Mv=m."
        bias_vector = torch.tensor(bias_vector, dtype=torch.float32)
        # basis matrix: MV=0
        self.unique = True if rM==col else False
        if self.unique:
            self.unique = True
            basis_matrix = np.zeros((col,1))
            print(f"The equation Mx=m has a unique solution x={str(list(bias_vector.reshape(-1)))}!")
        else:
            self.unique = False
            M_rref, pivot_idx = Matrix(M).rref()
            M_rref = np.array(M_rref.tolist()).astype(np.float32)
            free_idx = [idx for idx in range(col) if idx not in pivot_idx]
            basis = []
            for fi in free_idx:
                vector = np.zeros((col))
                vector[fi] = 1
                for r in range(len(pivot_idx)): # the pivot of r-th row is pivot[r]
                    vector[pivot_idx[r]] = -M_rref[r,fi]/(M_rref[r,pivot_idx[r]])
                assert abs(np.matmul(M, vector)).sum()<1e-6, "The basis matrix can't satisfy MV=0."
                basis.append(vector)
            basis_matrix = torch.tensor(np.array(basis).T, dtype=torch.float32)
        return basis_matrix, bias_vector

    def update(self, coef:np.array):
        self.coef = coef
        # if we don't adopt a latent layer, i.e. z=x, we set mu=None, W=eye(p)
        if not np.isnan(coef).any():
            mu = np.expand_dims(coef, axis=0) if len(coef.shape)==1 else coef
            assert len(mu.shape)==2, "Shape Error."
            # Wi^T=E*(Fi*mui+fi)+e, s.t. E1*(Fi*mui+fi)+e1!=0
            self.mus, batch = [mu[:,self.mus_idxs[i][0]:self.mus_idxs[i][1]] for i in range(len(self.mus_idxs))], len(mu)
            # labmdai=Fi*mui+fi (batch, latent_dim, input_dim)
            self.labs = torch.zeros(batch, self.latent_dim, self.Fs[0].shape[0])   
            for i in range(self.latent_dim):
                # li=Fi*mui+fi
                mui, Fi, fi = torch.tensor(self.mus[i], dtype=torch.float32).reshape(batch,-1,1), self.Fs[i], self.fs[i]
                self.labs[:,i:i+1,:] = (torch.matmul(Fi,mui)+fi).transpose(1,2)
            # Wi^T=E*li+e (batch, latent_dim, input_dim)
            self.weight = torch.matmul(self.labs, self.E.T) + self.e.T
        else:
            self.weight = torch.eye(self.input_dim)
        return self.weight

    def connect_ratio_limite(self, ):
        # connect limitation: Wi1=E1*li+e1!=0 (W1 are zi's weights for the connected input)
        self.idx1 = True
        for i in range(self.latent_dim):
            E1, e1 = self.Ee1[i]
            Wi1 = (torch.matmul(E1,self.labs[:,i:i+1,:].transpose(1,2))+e1).squeeze(-1)
            self.idx1 = self.idx1*torch.prod(Wi1, dim=1)!=0
        # ratio estimation: W_ij/W_ik ~ rho_j/rho_k
        self.idx2, weight = torch.full((self.idx1.sum(),), True, dtype=bool), self.weight[self.idx1]
        for i in range(self.latent_dim):
            connect, ratio, wi= self.latent_connect[i], self.latent_ratio[i], weight[:,i]
            base = wi[:,connect[0]-1]
            for j in range(1,len(ratio)):
                compare = wi[:,connect[j]-1]
                self.idx2 = self.idx2*(base*compare*ratio[j]>=0)        # sign(base)=sign(compare)*sign(ratio)
                # self.idx2 = self.idx2*(abs(base*ratio[j]-compare)<=1)   # base*ratio~compare
        return self.idx1, self.idx2

    def forward(self, x):
        # x: [batch, input_dim]->[batch, 1, input_dim]; weight: [latent_dim, input_dim]
        y = torch.prod(torch.pow(x.unsqueeze(1), self.weight), axis=-1)  
        return y

class PolynomialRegression():
    '''Polynomial Regresion'''
    def __init__(self, degree:int=10) -> None:
        self.degree = degree
        self.poly = PolynomialFeatures(degree=degree) 
        self.reg = LinearRegression(fit_intercept=False) 

    def fit(self, z:torch.tensor, y:torch.tensor):
        z_poly = self.poly.fit_transform(z)  
        self.reg.fit(z_poly, y)
        self.express = copy.deepcopy(self.reg.coef_[0])
        y_hat = self.reg.predict(z_poly)
        r2 = float(cal_r2(y, y_hat).numpy())
        return list(self.express), r2
    
class SymbolicRegression():
    """Symbolic Regression"""
    def __init__(self, iter:int=40, binary:list[str]=["+", "-", "*", "/"], unary:list[str]=["cos", "exp", "sin", "inv(x) = 1/x"], 
                    logger_dir:str='outputs/project/logger', D:torch.tensor=0, d:torch.tensor=0, 
                    units:list[str]=['kg','m','s','K','A','cd','mol'], batch:int=1024) -> None:
        self.D = D          # input unit matrix
        self.d = d          # output unit matrix
        self.units = units  # names of the basic units involved
        self.batch = batch  # number of samples in the dataset
        constraints1={"/": (-1, -1), "exp": 5, "sin": 5, "cos":5, "inv":5}
        constraints2={"exp": {"exp":0, "sin":0, 'cos':0}, "sin": {"exp":0, "sin":0, 'cos':0}, 'cos':{"exp":0, "sin":0, 'cos':0}}
        cstr1 = {key:constraints1[key] for key in constraints1.keys() if key in binary+unary}
        cstr2 = {key:{subkey:constraints2[key][subkey] for subkey in constraints2[key].keys() 
                    if subkey in binary+unary} for key in constraints2.keys() if key in binary+unary}
        self.model = PySRRegressor(niterations=iter, binary_operators=binary, unary_operators=unary, 
            extra_sympy_mappings={"inv": lambda x: 1/x}, temp_equation_file=True, tempdir=logger_dir, 
            dimensional_constraint_penalty=1000.0, batching=False, batch_size=max(self.batch, 64),
            procs=0, multithreading=False, deterministic=True, random_state=42,
            constraints=cstr1, nested_constraints=cstr2)
    def get_units(self, W:torch.tensor=1):
        '''get the units of z1~zs, y'''
        def cal_unit(v:torch.tensor, units):
            unit = ''
            for i in range(len(units)):
                if v[i] == 1:
                    unit += '*'+units[i]
                elif v[i] != 0:
                    unit += '*'+units[i]+'^{:.1f}'.format(v[i]) if int(v[i])!=v[i] else '*'+units[i]+'^{}'.format(int(v[i]))
            unit = unit[1:] if unit!='' else ""
            return unit
        assert len(W.shape)==2, "Shape Error."
        if self.d.any():
            self.U = torch.matmul(self.D, W.transpose(-2,-1))                                   # latent unit matrix
            self.z_units = [cal_unit(self.U[:,j], self.units) for j in range(self.U.shape[-1])] # latent units
            self.y_units = [cal_unit(self.d[:,j], self.units) for j in range(self.d.shape[-1])] # output unit
        else:
            self.U, self.z_units, self.y_units = 0, None, None

    def fit(self, z:torch.tensor, y:torch.tensor, W:torch.tensor):
        self.get_units(W=W)
        idxs = np.random.choice(z.shape[0], min(self.batch,z.shape[0]), replace=False) if self.batch!=-1 else torch.full((z.shape[0],), True, dtype=bool)
        self.model.fit(X=z[idxs], y=y[idxs], variable_names=[f'z_{i+1}' for i in range(z.shape[-1])], X_units=self.z_units, y_units=self.y_units)
        self.express = str(self.model.sympy())
        # self.express = self.express.replace('*','').replace(' ','')
        self.express = self.express.replace(' ','')
        for i in range(10):
            self.express = self.express.replace(f'_{i}',sub[i])
        y_hat = self.model.predict(X=z)
        r2 = float(cal_r2(y.view(-1), y_hat.reshape(-1)).numpy())
        return self.express, r2

class FINDFrame():
    def __init__(self, opt, log):
        self.opt = opt
        self.log = log
        # X\in R^n, Z\in R^m, Zi=\prod Xj^Wij, DW=0.
        self.D = torch.tensor(self.opt.Dataset.D, dtype=torch.float32) if self.opt.Dataset.D!=0 else torch.zeros((7,len(self.opt.Dataset.input_list)), dtype=torch.float32)       
        self.d = torch.tensor(self.opt.Dataset.d, dtype=torch.float32).reshape((-1,1)) if self.opt.Dataset.d!=0 else torch.zeros((self.D.shape[0],1), dtype=torch.float32)     
        self.units = self.opt.Dataset.units if self.opt.Dataset.units!=[] else ['kg','m','s','K','A','cd','mol'][:self.D.shape[0]]
        assert len(self.units)==self.D.shape[0], "The dimensions of D and units should be consistent."
        self.degree = 1 if self.d.any() else self.opt.Structure.express.degree
        self.input_dim = len(self.opt.Dataset.input_list)
        self.latent_dim = opt.Structure.latent.dim if opt.Structure.latent.adopt else self.input_dim
        self.unit_limit = True if self.D.any() else False
        # sampler, latent_module, pr_module are shared among multiple single tasks
        self.coef_range = self.opt.Structure.c2f.range
        self.sampler = DimlessSampler(norm=self.opt.Structure.express.norm, **self.opt.Dataset)
        self.latent_module = LatentLayer(D=self.D, d=self.d, latent_dim=self.latent_dim, \
            latent_connect=self.opt.Structure.latent.connect, latent_ratio=self.opt.Structure.latent.ratio, \
            coef_previous=self.opt.Structure.latent.previous)
        self.pr_module = PolynomialRegression(degree=self.degree) if opt.Structure.express.mode=='regression' \
            else INR(degree=self.degree)
        self.sr_module = SymbolicRegression(iter=opt.Structure.sr.iter, binary=opt.Structure.sr.binary, unary=opt.Structure.sr.unary,
                            logger_dir=self.log.logger_dir, D=self.D, d=self.d, units=self.units, batch=self.opt.Structure.sr.batch)
        self.terms = self.get_terms(self.degree, self.latent_dim)
        # dimensionality reduction: W[:,j]=[W1j,W2j,...,Wnj]=c*v, find c rather than W[:,j]
        self.coef_previous = self.latent_module.coef_previous
        self.coef_dim = self.latent_module.coef_dim
        self.signs = {}
        self.dataset_limitation()
        self.get_metrics()
        self.time_start = time.time()
    
    def dataset_limitation(self, ):
        '''judge the coefs range according to the dataset range'''
        self.limite = {'negative':[], 'zero':[]}
        # Zi=\prod Xj^Wij; y=f(Z)
        for i in range(self.input_dim):
            xi = self.sampler.input[:,i]
            # (1) if there is a value less than 0 for xi, limit wi to only integers;
            self.limite['negative'].append(True) if xi.min() < 0 else self.limite['negative'].append(False)
            # (2) if xi has a value of 0, limit wi>=0;
            self.limite['zero'].append(True) if abs(xi).min()==0 or (xi.min()<0 and xi.max()>0) else self.limite['zero'].append(False)

    def remove_exist(self, coefs):
        '''judge whether to run if we had already searched before'''
        def coef_sign_code(coef):
            '''encode the coef according to its signs. eg. [1,2,-3,-4,8,0]->[2,2,0,0,2,1]->2+2*3^1+2*3^4+1*3^5'''
            sign_code = np.sign(coef) + 1   
            sign_code = int(sum([sign_code[i]*3**i for i in range(len(sign_code))]))
            return sign_code
        # encode the existing coefficients according to their signs
        if self.signs == {}:
            for coef in self.metrics['mu']:
                sign_code = coef_sign_code(coef=coef)
                if sign_code not in self.signs.keys():
                    self.signs[sign_code] = [coef]  
                else:
                    self.signs[sign_code].append(coef)
        # remove existing coefficients
        pbar, final_coefs = tqdm(coefs, desc='Removing existing coefficients', leave=True, file=sys.stdout), []
        for coef in pbar:
            coef = list(coef) if type(coef).__name__!='str' else list(np.concatenate([eval(c) for c in eval(coef)],-1))
            # compare only in coefficient sets with consistent symbols to reduce search time
            sign_code = coef_sign_code(coef)
            if sign_code not in self.signs.keys():
                final_coefs.append(coef)
                self.signs[sign_code] = [coef]
            elif coef not in self.signs[sign_code]:
                final_coefs.append(coef)
                self.signs[sign_code].append(coef)
        return np.array(final_coefs) 

    def coef_limit(self, coefs):
        '''restrict coefficients based on dataset distribution to reduce search space. coefs: (batch,self.latent_dim*self.coef_dim)'''
        coefs, num_init, time_init = ((coefs*10).astype(np.int32))/10., coefs.shape[0], time.time()
        # if there's only one solution for unit equation Dx=d
        if self.latent_module.unique:
            self.opt.Structure.c2f.step = []    # stop searching for other coefficients
            return [[0.]]
        # w_ij/w_ik = rho_j/rho_k: connection relationship and ratio estimation
        if self.opt.Structure.latent.connect!=[]:
            self.latent_module.update(coef=coefs)
            idx1, idx2 = self.latent_module.connect_ratio_limite()
            coefs = coefs[idx1][idx2]
        # limit n(W)<=k1, n(W_{:,j})<=k2
        k1 = int(self.opt.Structure.c2f.k1)
        weight = self.latent_module.update(coef=coefs).reshape(coefs.shape[0], -1)
        weight[weight!=0] = 1
        idx1 = weight.sum(-1)<=k1
        coefs = coefs[idx1]
        k2 = min(max(1, int(self.opt.Structure.c2f.k2)), self.input_dim)
        weight = self.latent_module.update(coef=coefs)
        weight[weight!=0] = 1
        idx2 = ((weight.sum(-1))<=k2).sum(-1)==self.latent_dim
        coefs = coefs[idx2]
        # limit the weight based on the dataset
        weight, idx = self.latent_module.update(coef=coefs), torch.ones(coefs.shape[0], dtype=torch.int32)
        for i in range(self.input_dim):
            if self.limite['negative'][i%self.input_dim]:
                idx = idx*((weight[:,:,i]==weight[:,:,i].to(torch.int32)).sum(-1)==self.latent_dim)
            if self.limite['zero'][i%self.input_dim]:
                idx = idx*((weight[:,:,i]>=0).sum(-1)==self.latent_dim)
        coefs = coefs[np.nonzero(idx==1)].reshape([-1,coefs.shape[-1]])
        # [[c11,c12,c13],[c21,c22,c23]]~[[c21,c22,c23],[c11,c12,c13]], [[c1,c2],[c1,c2]]~[[c1,c2]], only test one of them and remove [0]^p
        if self.opt.Structure.latent.connect!=[]:
            refine_coefs = coefs
        else:
            refine_coefs = []
            pbar = tqdm(coefs, desc='Removing redundant coefficients', leave=True, file=sys.stdout)
            for coef in pbar:
                coef = set([str(list(c)) for c in coef.reshape(self.latent_dim, -1)])                       # [[c1,c2],[c1,c2]]~[[c1,c2],[0,0]]~[[c1,c2]]            
                coef = sorted(coef) + [str([0]*self.latent_module.E.shape[-1])]*(self.latent_dim-len(coef)) # [[c1,c2]]~[[c1,c2],[0,0]]           
                refine_coefs.append(str(coef))                                                              # [[c1,c2],[c3,c4]]~[[c3,c4],[c1,c2]]
                refine_coefs.append(str(sorted(coef)))
            refine_coefs = set(refine_coefs)
        final_coefs = self.remove_exist(refine_coefs)
        print(f'It took {time.time()-time_init}s to reduce the search space from {num_init} to {len(final_coefs)} coefficients.')
        return final_coefs

    def get_metrics(self, precision:float=0.1):
        '''read the metrics save in project/metrics/metrics.csv'''
        self.metrics = {'mu':[], 'W':[], 'n(W)':[], 'r2_1':[], 'norm(PR)':[], 'PR':[], 'r2_2':[], 'SR':[]}
        metrics_path = os.path.join(self.log.metric_dir, 'metrics.csv')
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path).sort_values(by="r2_1", ascending=False)
            for i in range(len(df)):
                weight = eval(df.loc[i,'W'])
                if sum([int(w*10)%int(precision*10) for w in weight])==0:
                    for key in self.metrics.keys():
                        if not self.opt.Structure.latent.adopt and key=='mu':
                            data = None
                        elif type(df.loc[i,key]).__name__=='str' and key not in ['r2_2', 'SR']:
                            data = eval(df.loc[i,key])
                        else:
                            data = df.loc[i,key]
                        self.metrics[key].append(data)

    def get_terms(self, n, p):
        '''get the polynomial terms'''
        terms = ['c']
        idxs = [i for i in range(1, p+1)]   # the idxs of z_1~z_p
        for i in range(1, n+1):
            term_combinations = combinations_with_replacement(idxs, i)
            for comb in term_combinations:
                count, term = {}, ''
                for c in comb:
                    count[c] = 1 if c not in count.keys() else count[c]+1
                for k in count.keys():
                    term = term+'z'+sub[k]+sup[count[k]] if p>1 else term+'z'+sup[count[k]]
                terms.append(term)
        return terms
    
    def simple_express(self, express, dim):
        '''simplify the expression'''
        terms = self.terms if len(express)==len(self.terms) else self.get_terms(self.degree, dim)
        express = np.array(express)
        thres = min(0.1, 0.1*max(abs(express[1:]))) if len(express)>0 else 0.1
        express[1:][abs(express[1:])<thres] = 0
        idxs1, idxs2 = express!=0, np.arange(len(terms), dtype=np.int32)[express!=0]
        return express[idxs1], [terms[i] for i in idxs2]
    
    def performance_display(self, top:int=20, sparse:int=64, precision:float=0.1):
        '''display the performance'''
        def dis_coef(coef, sign:bool=False, acc:int=2):
            if coef==0:
                return '+0.0'
            acc, pn = int(10**acc), int(coef/abs(coef))
            c = f'{int(acc*coef+pn*0.5)/acc}' if (abs(coef)>=1./acc and abs(coef)<10) else '{:.2e}'.format(coef)
            sc = '+'+c if (sign and coef>0) else c
            return sc
        self.get_metrics(precision=precision)
        mus, Ws, nWs, r2_1, pr, r2_2, sr = self.metrics['mu'], self.metrics['W'], self.metrics['n(W)'], \
            self.metrics['r2_1'], self.metrics['PR'], self.metrics['r2_2'], self.metrics['SR']
        sr_flag = sum([r22!='*' for r22 in self.metrics['r2_2']])>=1
        top_dict = {'id':[], chr(956):[], f'z=f{sub[1]}(x)':[], 'n(W)':[], 'n(pr)':[], f'R{sup[2]}{sub[1]}':[], f'PR: y=f{sub[2]}(z)':[]}
        if sr_flag:
            top_dict[f'R{sup[2]}{sub[2]}'], top_dict[f'SR: y=f{sub[2]}(z)'] = [], []
        info_len = {k:len(k) for k in top_dict.keys()}
        for idx in range(len(mus)):
            if len(top_dict['id'])>=top:
                break
            if nWs[idx]>sparse:
                continue
            mu = '['+'_'.join([dis_coef(c, sign=True, acc=1) for c in mus[idx]])+']' if mus[idx]!=None else 'none'
            W = [dis_coef(c, sign=True, acc=1) for c in Ws[idx]]
            z_term = [sub_sup('x',j+1,W[j+i*self.input_dim]) for i in range(self.latent_dim) for j in range(self.input_dim)]
            z = '_'.join([''.join(z_term[i*self.input_dim:(i+1)*self.input_dim]) for i in range(self.latent_dim)])
            express, terms = self.simple_express(pr[idx], int(len(mus[idx])/self.coef_dim)) if mus[idx]!=None else self.simple_express(pr[idx], 1)
            r21, nW, nf = dis_coef(r2_1[idx], acc=4), str(nWs[idx]), str(len(express))
            express_pr = 'y=' + dis_coef(express[0]) if len(express)>0 else 'csv acc error'
            for i in range(1, len(express)):
                express_pr = express_pr + dis_coef(express[i], sign=True, acc=2) + terms[i]
            express_pr = express_pr[:100]+'...' if len(express_pr)>100 else express_pr
            id_dict = {'id':str(idx), chr(956):mu, f'z=f{sub[1]}(x)':z, 'n(W)':nW, 'n(pr)':nf, f'R{sup[2]}{sub[1]}':r21, f'PR: y=f{sub[2]}(z)':express_pr}
            if sr_flag:
                r22, express_sr = dis_coef(eval(str(r2_2[idx])), acc=4) if r2_2[idx]!='*' else '*', str(sr[idx])
                id_dict[f'R{sup[2]}{sub[2]}'], id_dict[f'SR: y=f{sub[2]}(z)'] = r22, express_sr
            for k in top_dict.keys():
                top_dict[k].append(id_dict[k])
                info_len[k] = max(info_len[k], len(id_dict[k]))
        info = str([k.ljust(info_len[k],' ') for k in top_dict.keys()])[1:-1].replace('\'','').replace(',', ' |')+ ' |'
        title = f'Performance {min(top,len(mus))}/{len(mus)} (n(W)<={sparse}, precision={precision})'
        show_info = '-'*len(info) + '\n' + ' '*int((len(info)-len(title))/2) + title + '\n' + '_'*len(info) + '\n' + info + '\n'
        for idx in range(len(top_dict['id'])):
            show_info += str([top_dict[k][idx].ljust(info_len[k],' ') for k in top_dict.keys()])[1:-1].replace('\'','').replace(',', ' |').replace('_',',')+ ' |\n'
        show_info += '-'*len(info)
        f_info = open(os.path.join(self.log.metric_dir, 'performance.txt'), 'w+', encoding='utf-8')
        f_info.write(show_info)
        print(show_info)
    
    def get_top_idx(self, top, thres):
        if self.latent_module.unique:
            print(self.latent_module.unique_info)
            quit()
        for thres_idx in range(min(len(self.metrics['r2_1'])-1,3), min(len(self.metrics['r2_1']),top)):
            if self.metrics['r2_1'][thres_idx] < thres:
                break
        return thres_idx+1

    def grid_initial(self, ):
        '''stage 1: regression for [-m,...,-1,0,1,...,m]^p.  t~(2m+1)^p''' 
        print('\033[1;32m' + '='*60 + ' '*4 + 'GRID INITIALIZATION START' + ' '*4 + '='*60 + '\033[0m')
        # if we don't adopt a latent layer, i.e. z=x, we set mu=None, W=eye(p)
        if not self.opt.Structure.latent.adopt:
            coefs = np.full((1, 1), np.nan)
        # if we have set fixed coefs on the yaml file
        elif self.opt.Structure.c2f.fix!=[] and self.opt.Structure.c2f.fix!=[[]]:
            fix_coefs = ((np.array(self.opt.Structure.c2f.fix)*10).astype(np.int32))/10.
            fix_coefs = [coef.tolist() for coef in fix_coefs]
            coefs = self.remove_exist(fix_coefs)
            print('Start searching near the given point...')
        else:
            init_step = self.opt.Structure.c2f.init_step
            shape = [[c, c, 1] for c in self.coef_previous]
            shape = shape + [self.coef_range + [int((self.coef_range[-1]-self.coef_range[0])/init_step)+1]]*(self.coef_dim-len(self.coef_previous))
            coefs = np.array(create_flattened_coords(shape=shape), dtype=np.float32) if len(shape)!=0 else []
            coefs = self.coef_limit(coefs=coefs)
        self.train(coefs=coefs)
        self.get_metrics()
        self.performance_display(**self.opt.Display)
        time_cost = time.time() - self.time_start
        self.log.log_metrics({'stage time':time_cost}, 0)
        print(f'Time Cost: {time_cost}')
        print('\033[1;32m' + '='*60 + ' '*4 + ' GRID INITIALIZATION END ' + ' '*4 + '='*60 + '\033[0m')
    
    def grid_refine(self, top:int=20, thres:float=0.5, step:float=0.5, id:int=0):
        '''stage 2: search near the top coefficients. eg. [1.0-step, 1.0, 1.0+step] for 1.0.     t~top*3^p-top''' 
        print('\033[1;32m' + '='*60 + ' '*4 + f' GRID REFINEMENT {id} START ' + ' '*4 + '='*60 + '\033[0m')
        print(f'Setting: top({top}) thres({thres}) step({step})\n')
        thres_idx = self.get_top_idx(top, thres)
        top_coefs = self.metrics['mu'][:thres_idx]
        # search near the top coefficients, step=0 indicates further optimization exclusively on the top coefficients
        coefs = []
        for top_coef in top_coefs:
            # if we don't adopt a latent layer, i.e. z=x, we set mu=0, W=eye(p)
            if top_coef==[0]:
                continue
            shape = [[c, c, 1] for c in self.coef_previous]
            for c in top_coef[len(self.coef_previous):]:
                c_min = c-step if c-step>=self.coef_range[0] else c
                c_max = c+step if c+step<=self.coef_range[1] else c
                num = 3 if (c-step>=self.coef_range[0]) and (c+step<=self.coef_range[1]) else 2
                shape.append([c_min, c_max, num]) if step!=0 else shape.append([c, c, 1])
            coefs.append(np.array(create_flattened_coords(shape=shape), dtype=np.float32))
        coefs = self.coef_limit(coefs=np.concatenate(coefs,axis=0)) if coefs!=[] else []
        self.train(coefs=coefs)
        self.get_metrics()
        self.performance_display(**self.opt.Display)
        time_cost = time.time() - self.time_start
        self.log.log_metrics({'stage time':time_cost}, id)
        print(f'Time Cost: {time_cost}')
        print('\033[1;32m' + '='*60 + ' '*4 + f'  GRID REFINEMENT {id} END  ' + ' '*4 + '='*60 + '\033[0m')
    
    def pr2sr(self, top, thres):
        '''stage 3: replace polynomial expressions with symbolic expressions after we find the best latent variables'''
        print('\033[1;32m' + '='*60 + ' '*2 + f'POLYNOMIAL TO SYMBOLIC START ' + ' '*2 + '='*60 + '\033[0m')
        thres_idx = self.get_top_idx(top, thres)
        top_coefs = self.metrics['mu'][:thres_idx]
        pbar = tqdm(range(len(top_coefs)), desc='Training', leave=True, file=sys.stdout)
        y = (self.sampler.label*self.sampler.output_norm)
        x = self.sampler.input
        if 'r2_2' not in self.metrics.keys():
            self.metrics['r2_2'], self.metrics['SR'] = ['*']*len(self.metrics['mu']), ['*']*len(self.metrics['mu'])
        for i in pbar:
            if self.metrics['r2_2'][i]=='*':
                top_coef = np.array(top_coefs[i]) if top_coefs[i]!=None else np.full((1, 1), np.nan)
                W = self.latent_module.update(coef=top_coef).reshape(self.latent_dim, self.input_dim)
                z = self.latent_module(x) 
                express, r2 = self.sr_module.fit(z=z, y=y, W=W)
                self.metrics['r2_2'][i], self.metrics['SR'][i] = r2, express
            pbar.update(1)
        df = pd.DataFrame(data=self.metrics)
        df.to_csv(os.path.join(self.log.metric_dir, 'metrics.csv'), encoding='utf-8')
        self.performance_display(**self.opt.Display)
        time_cost = time.time() - self.time_start
        self.log.log_metrics({'stage time':time_cost}, len(self.opt.Structure.c2f.refine_step)+1)
        print(f'Time Cost: {time_cost}')
        print('\033[1;32m' + '='*60 + ' '*2 + f' POLYNOMIAL TO SYMBOLIC END  ' + ' '*2 + '='*60 + '\033[0m')
        return self.metrics

    def train(self, coefs):
        '''search coefs with c2f pr'''
        pbar = tqdm(coefs, desc='Training', leave=True, file=sys.stdout)
        for coef in pbar:
            # update latent weights
            self.latent_module.update(coef=coef)
            self.sampler.input2latent(latent_module=self.latent_module, latent_norm=self.opt.Structure.latent.norm)
            # get metrics
            metrics = {'mu':list(self.latent_module.coef), 'W':[float(w) for w in self.latent_module.weight.view(-1)], 
                       'n(W)': int(np.array(self.latent_module.weight!=0).sum())}
            if torch.isnan(self.sampler.latent).any() or torch.isinf(self.sampler.latent).any():
                metrics['r2_1'], metrics['norm(PR)'], metrics['PR']  = 0, [], []
            else:
                PR_norm, r2 = self.pr_module.fit(self.sampler.latent, self.sampler.label)
                PR = self.sampler.invnorm(express_norm=PR_norm, pr_module=self.pr_module)
                metrics['r2_1'], metrics['norm(PR)'], metrics['PR']  = r2, PR_norm, PR
            metrics['r2_2'], metrics['SR'] = '*', '*'
            for key in metrics.keys():
                self.metrics[key].append(metrics[key])
            pbar.update(1)
        df = pd.DataFrame(data=self.metrics).sort_values(by="r2_1", ascending=False)
        df.to_csv(os.path.join(self.log.metric_dir, 'metrics.csv'), encoding='utf-8')
        return self.metrics

    def start(self, mode_dict:dict={'mode':'train','eval':0.1,'refine':0.0}):
        '''FIND: latent identify->c2f polynomial regression->symbolic regression'''
        if mode_dict['mode'] == 'train':
            # stage 1: initialization, step 1.0
            self.grid_initial()
            # stage 2: c2f search near the top coefficients, step 0.5, 0.2, 0.1
            top, step = self.opt.Structure.c2f.top, self.opt.Structure.c2f.refine_step
            if self.opt.Structure.latent.adopt:
                for i in range(len(step)):
                    self.grid_refine(top=top, thres=min(0.6+i*0.1, 0.98), step=step[i], id=i+1)
            # stage 3: replace polynomial expressions with symbolic expressions
            if self.opt.Structure.sr.adopt:
                self.pr2sr(top=top, thres=0.90)
            self.log.close()
        elif mode_dict['mode'] == 'eval':
            self.performance_display(top=mode_dict['top'], sparse=mode_dict['sparse'], precision=mode_dict['eval'])
        elif mode_dict['mode'] == 'refine':
            self.grid_refine(top=10, thres=0.90, step=mode_dict['refine'], id=1)
            if self.opt.Structure.sr.adopt:
                self.pr2sr(top=10, thres=0.90)
            self.performance_display(top=mode_dict['top'], sparse=mode_dict['sparse'], precision=mode_dict['eval'])
        else:
            raise NotImplemented