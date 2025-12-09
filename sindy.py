# https://xiaoyuxie.top/PyDimension-Book/examples/discover_spring_clean.html
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
import copy, argparse

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

class FitEqu(object):
    '''
    For a given data, fit the governing equation.
    '''
    def __init__(self):
        super(FitEqu, self).__init__()
    
    def cal_derivatives(self, series, dt, Nt, deg=3, num_points=100, boundary_t=5):
        '''
        prepare library for regression
        '''
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
            x[p] = series[t]
            x_part = series[t-int((Nt_sample-1)/2): t+int((Nt_sample+1)/2)]
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
    
def main():
    series_path = args.d
    save_path = args.s
    data = []
    df = pd.read_csv(series_path)
    settings = df.columns.tolist()
    settings.remove('Unnamed: 0')
    settings.remove('t')
    time = df['t']
    dt, Nt = time[1]-time[0], len(time)
    fit_equ = FitEqu()
    for setting in settings:
        sets = [kv.split('=') for kv in setting.split(',')]
        keys, values = [s[0] for s in sets], [float(s[1]) for s in sets]
        series = df[setting].to_numpy()
        x, xt, xtt = fit_equ.cal_derivatives(series, dt, Nt)
        X_library, y_library, names = fit_equ.build_library(x, xt, xtt)
        coef = fit_equ.fit(X_library, y_library)
        data.append(values + list(coef))
    df = pd.DataFrame(data, columns=keys + [f'a{i}' for i in range(len(coef))])
    print(df)
    df.to_csv(save_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='SINDy')
    parser.add_argument('-d', type=str, default='dataset/pde_spring_series.csv', help='time series path')
    parser.add_argument('-s', type=str, default='dataset/pde_spring.csv', help='time series path')
    parser.add_argument('-refine', type=float, default=0.0, help='refine step')
    parser.add_argument('-top', type=int, default=20, help='number of displayed coefficients')
    parser.add_argument('-sparse', type=int, default=20, help='sparsity of displayed coefficients')
    args = parser.parse_args()
    main()