import copy
import numpy as np
import pandas as pd

path = 'dataset/toy0.csv'
input_list = ['x1', 'x2', 'x3', 'x4', 'x5']
# input_list = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']
output_list = ['y']

df = pd.read_csv(path)
X, Y = df[input_list].to_numpy(), df[output_list].to_numpy()
p, batch = len(input_list), len(X)

sort_X = [sorted(set(X[:,j])) for j in range(p)]
idx_X = [{sort_X[i][j]:j for j in range(len(sort_X[i]))} for i in range(p)]

Y_dict = {}
for i in range(batch):
    x, y = str(list(X[i])), float(Y[i][0])
    Y_dict[x] = y
# print(Y_dict)

W = []
for i in range(batch):
    w = []  # (py/pxj)*xj~=(y0-yj)/(x0-xj)*xj
    xi, yi = list(X[i]), float(Y[i][0])
    for j in range(p):
        xij0 = xi[j]
        id = idx_X[j][xij0]
        if id==0:
            break
        else:
            id1 = id - 1
        xij1 = sort_X[j][id1]
        delta_xij = xij0 - xij1

        new_xi = copy.deepcopy(xi)
        new_xi[j] = xij1
        if str(new_xi) not in Y_dict.keys():
            break
        yij1 = Y_dict[str(new_xi)]
        delta_yij = yi - yij1
        
        w.append(delta_yij/delta_xij*xij0)
    if len(w)==p:
        W.append(w)

print(len(W))
W = np.array(W)
COR = np.zeros((p,p))
R = np.zeros((p,p))
for i in range(p):
    for j in range(p):
        wi, wj = W[:,i].flatten(), W[:,j].flatten()
        # r = (wi/wj).mean()
        # r = (wi*wj-wi.mean()*wj.mean()).sum()/(wi**2-wi.mean()**2).sum()
        r = (wi*wj-wi.mean()*wj.mean()).sum()/(wj**2-wj.mean()**2).sum()
        COR[i,j] = np.corrcoef(wi, wj)[0, 1]
        R[i,j] = r
print(COR)
print(R)