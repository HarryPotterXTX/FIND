import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, asin, exp, cos
from sklearn.linear_model import LinearRegression, Ridge

plt.rcParams["font.family"] = "Arial"
np.set_printoptions(suppress=True)

class SeqReg(object):

    def __init__(self):
        pass

    def normalize(self, X, y):
        '''
        Normalization the data
        '''
        norm_coef_X = np.mean(np.abs(np.mean(X, axis=0)))
        norm_coef_y = np.mean(np.abs(np.mean(y, axis=0)))
        norm_coef = min(norm_coef_X, norm_coef_y)
        # print('Before X', pd.DataFrame(np.concatenate([X, y], axis=1)).describe())
        X = X / norm_coef
        y = y / norm_coef
        # print('After X', pd.DataFrame(np.concatenate([X, y], axis=1)).describe())
        return X, y

    def fit_fixed_threshold(self, X, y, alpha=1.0, threshold=0.005, is_normalize=True):
        if is_normalize:
            X, y = self.normalize(X, y)
        
        # initialize a linear regression model
        # model = LinearRegression(fit_intercept=False)
        model = Ridge(fit_intercept=False, alpha=1)
        model.fit(X, y)
        # r2 = model.score(X, y)
        for idx in range(3):
            coef = model.coef_
            flag = np.repeat((np.abs(coef) > threshold).astype(int).reshape(1,-1), 
                             X.shape[0], axis=0)
            X1 = copy.copy(X)
            X1 = np.multiply(X1, flag)
            model.fit(X1, y)
            r2 = model.score(X1, y)
            print(f'training {idx} r2: {r2}')
        coef = np.squeeze(model.coef_)
        return coef, X1

    def fit_dynamic_thresh(self, X, y, non_zero_term=4, alpha=1.0, threshold=0.005, 
                is_normalize=True, fit_intercept=False, model_name='Ridge', max_iter=200):
        '''
        decrease the threshold when there are only limited non-zero terms
        and increase the threshold when thre are more non-zeros terms
        '''
        if is_normalize:
            X, y = self.normalize(X, y)
        
        # initialize a linear regression model
        if model_name == 'Ridge':
            model = Ridge(fit_intercept=fit_intercept, alpha=alpha)
        elif model_name == 'LR':
            model = LinearRegression(fit_intercept=fit_intercept)
        else:
            raise Exception('Wrong model_name.')
        model.fit(X, y)
        count = 0

        while count <= max_iter:
            coef = model.coef_
            flag = np.repeat((np.abs(coef) > threshold).astype(int).reshape(1,-1), 
                             X.shape[0], axis=0)
            cur_non_zero_term = np.sum(flag[0,:])
            X1 = copy.copy(X)
            X1 = np.multiply(X1, flag)
            model.fit(X1, y)
            r2 = model.score(X1, y)
            # print(f'training r2: {r2}, threshold: {threshold}, cur_non_zero_term: {cur_non_zero_term}')
            if cur_non_zero_term == non_zero_term:
                break
            elif cur_non_zero_term < non_zero_term:
                threshold *= 0.95
            else:
                threshold *= 1.05
            count += 1

        coef = np.squeeze(model.coef_)
        if fit_intercept:
            coef_list = coef.tolist()
            coef_list.append(float(model.intercept_))
            coef = np.array(coef_list)

        return coef, X1, r2

def PolyDiffPoint(u, x, deg=3, diff=1, index=None):
    '''
    Poly diff
    The original code of this part: https://github.com/snagcliffs/PDE-FIND
    '''
    n = len(x)
    # if index == None: index = int((n-1)/2)
    if index == None: index = (n-1)//2

    # Fit to a polynomial
    poly = np.polynomial.chebyshev.Chebyshev.fit(x, u, deg)
    
    # Take derivatives
    derivatives = []
    for d in range(1, diff + 1):
        derivatives.append(poly.deriv(m=d)(x[index]))
    
    return derivatives

class RLCDataset(object):
    '''
    Generate data for spring-mass-damping systems
    '''
    def __init__(self, R, L, C, I0, V0, et=20, Nt=800):
        super(RLCDataset, self).__init__()
        self.R = R
        self.L = L
        self.C = C
        self.I0 = I0
        self.V0 = V0
        self.et = et
        self.Nt = Nt

    def solution(self):
        t = np.linspace(0, self.et, self.Nt, endpoint=False)
        R, L, C, I0, V0 = self.R, self.L, self.C, self.I0, self.V0
        # Underdamped
        if R**2<4*L/C:
            a, Wd = R/(2*L), sqrt(1/(L*C)-(R/(2*L))**2)
            z = (R*I0-L*a*I0+V0)/(L*Wd)
            A = sqrt(I0**2+z**2)
            phi = asin(z/A)
            i = A*np.exp(-a*t)*np.cos(Wd*t+phi)
        # Critical Damping
        elif R**2==4*L/C:
            A = I0
            B = (L*a*A-V0-R*I0)/L
            i = (A+B*t)*np.exp(-a*t)
        # Overdamped
        else:
            s1, s2 = (-R+sqrt(R**2-4*L/C))/(2*L), (-R-sqrt(R**2-4*L/C))/(2*L)
            B = (R*I0+L*I0*s1+V0)/(L*s1-L*s2)
            A = I0-B
            i = A*np.exp(s1*t)+B*np.exp(s2*t)
        info = {'t': t, 'i': i}
        df = pd.DataFrame(info)
        return df

class FitEqu(object):
    '''
    For a given data, fit the governing equation.
    '''
    def __init__(self):
        super(FitEqu, self).__init__()
        
    def prepare_data(self, R, L, C, I0, V0, et, Nt):
        '''
        generate the dataset
        '''
        dataset = RLCDataset(R, L, C, I0, V0, et=et, Nt=Nt)
        data = dataset.solution()  # {'t': t, 'i': i}
        return data
    
    def cal_derivatives(self, data, dt, Nt, deg=3, num_points=100, boundary_t=5):
        '''
        prepare library for regression
        '''
        x_clean = data['i'].to_numpy()
        t = np.arange(2*boundary_t, Nt-2*boundary_t)
        # points = np.random.choice(t, num_points, replace=False)
        points = t
        num_points = points.shape[0]

        x = np.zeros((num_points, 1))
        xt = np.zeros((num_points, 1))
        xtt = np.zeros((num_points, 1))

        Nt_sample = 2 * boundary_t - 1
        for p in range(num_points):
            t = points[p]
            x[p] = x_clean[t]
            x_part = x_clean[t-int((Nt_sample-1)/2): t+int((Nt_sample+1)/2)]
            xt[p], xtt[p] = PolyDiffPoint(x_part, np.arange(Nt_sample)*dt, deg, 2)

        return x, xt, xtt

    @staticmethod
    def build_library(x, xt, xtt):
        '''
        build the library for sparse regression: xt=f(c,x,xtt,x^2,x*xt,x*xtt,xt^2,xt*xtt,xtt^2)
        '''
        X_library = [
            np.ones_like(x),
            x, 
            xtt, 
            x**2, 
            np.multiply(x.reshape(-1, 1), xt.reshape(-1, 1)),
            np.multiply(x.reshape(-1, 1), xtt.reshape(-1, 1)),
            xt**2,
            np.multiply(xt.reshape(-1, 1), xtt.reshape(-1, 1)),
            np.multiply(xtt.reshape(-1, 1), xtt.reshape(-1, 1)),
        ]
        X_library = np.squeeze(np.stack(X_library, axis=-1))
        names = ['c', 'x', 'xtt', 'x^2', 'x*xt', 'x*xtt', 'xt^2', 'xt*xtt', 'xtt^2']
        y_library = xt
        return X_library, y_library, names
    
    @staticmethod
    def fit(X_library, y_library, threshold=0.002):
        '''
        squential threshold with dynamic threshold
        '''
        model = SeqReg()
        coef, _, r2 = model.fit_dynamic_thresh(X_library, y_library, 
                        is_normalize=False, non_zero_term=2, threshold=threshold, fit_intercept=False, model_name='LR')
        print('Fitting r2', r2)
        return coef


def prepare_dataset(is_show=False):
    '''
    prepare a sets of dataset with different parameters
    '''
    data = []
    fit_equ = FitEqu()
    # R, L, C, I0, V0
    params = [
        [0.10, 1.0, 1.0, 0.10, 0.3],
        [0.50, 0.25, 0.3, 0.05, 0.6],
        [0.70, 0.1, 0.1, 0.02, 0.9],
        [0.07, 0.2, 0.2, 0.07, 1.0],
        [0.20, 0.1, 1.0, 0.05, 0.4],
        [0.12, 0.3, 0.7, 0.08, 0.6],
        [0.60, 0.6, 0.6, 0.09, 0.2],
        [0.20, 0.4, 0.3, 0.10, 0.3]
    ]
    et, Nt = 20, 800
    if is_show: fig = plt.figure(); 
    for R, L, C, I0, V0 in params:
        dt = et / float(Nt)
        df_each = fit_equ.prepare_data(R, L, C, I0, V0, et, Nt)
        if is_show: plt.plot(df_each['t'], df_each['i'])
        x, xt, xtt = fit_equ.cal_derivatives(df_each, dt, Nt)
        X_library, y_library, names = fit_equ.build_library(x, xt, xtt)
        coef = fit_equ.fit(X_library, y_library)
        data.append([R, L, C, I0, V0] + list(coef))

    if is_show: 
        plt.xlabel('Time: t (s)', fontsize=20)
        plt.ylabel('current: i (A)', fontsize=20)
        plt.tick_params(labelsize=14)
        plt.tight_layout()
        plt.show()

    df = pd.DataFrame(data, columns=['R', 'L', 'C', 'I0', 'V0'] + [f'a{i}' for i in range(len(coef))])
    return df

if __name__=="__main__":
    # fix it=f(c,i,itt,i^2,i*it,i*itt,it^2,it*itt,itt^2)
    csv_path = 'dataset/pde_rlc.csv'
    df = prepare_dataset(is_show=True)
    print(df)
    df.to_csv(csv_path)