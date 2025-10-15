import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import combinations
import copy, sys, os, warnings
from .FIND import sub

warnings.filterwarnings("ignore")
plt.rcParams['mathtext.default'] = 'regular'
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def UnaryOperation(data, name, unit, operator):
    results = {
        'cos': [np.cos(data), f'cos({name})', unit],
        'sin': [np.sin(data), f'sin({name})', unit],
        'tan': [np.tan(data), f'tan({name})', unit],
        'exp': [np.exp(data), f'exp({name})', unit],
        'tanh': [np.tanh(data), f'tanh({name})', unit],
        'inv': [1./data, f'inv({name})', -unit],
        'sqrt': [np.sqrt(data), f'sqrt({name})', 0.5*unit],
        'square': [np.square(data), f'square({name})', 2.0*unit],
        'abs': [np.abs(data), f'abs({name})', unit],
        'exp-': [np.exp(-data), f'exp(-{name})', unit]
    }
    if operator in ['cos','sin','tan','exp','tanh', 'exp-'] and unit.any():
        return []
    return results[operator]

def BinaryOperation(data1, data2, name1, name2, unit1, unit2, operator):
    results = {
        '+': [data1+data2, f'{name1}+{name2}', unit1],
        '-': [data1-data2, f'{name1}-{name2}', unit1],
        '*': [data1*data2, f'{name1}*{name2}', unit1+unit2],
        '/': [data1/data2, f'{name1}/{name2}', unit1-unit2],
    }
    if operator in ['+','-'] and (unit1-unit2).any():
        return []
    return results[operator]

def InputCombination(csv_path:str, input_list:list, output_list:list, unit_matrix:list, origin:bool=False, mixed:bool=False, only_binary:bool=False):
    '''Input Combination:  
        Unary Operation: x1~xp -> sin(x1),cos(x1),...,abs(xp)
        Binary Operation: sin(x1),cos(x1),...,abs(xp) -> sin(x1)-sin(x3),sqrt(x2)+sqrt(x4)
        mixed: True -> opt1(xi)+opt2(xj); False -> opt1(xi)+opt1(xj)
        only_binary: True -> opt1(xi)+opt2(xj); False -> opt(xi)
    '''
    # load original data
    csv_data = pd.read_csv(csv_path)
    X, Y = csv_data[input_list].to_numpy(), csv_data[output_list].to_numpy().reshape(-1)
    D = np.array(unit_matrix) if unit_matrix!=0 else np.zeros([1,len(input_list)])
    assert X.shape[-1]==D.shape[-1], "The dimension of the input variable is inconsistent with that of the dimensional matrix."
    if origin:
        return X, Y, input_list, D
    # unary operation: cos\exp\sin for dimensionless number
    # unary = ['cos','sin','tan','exp','tanh','inv','sqrt','square','abs','exp-']
    unary = ['sin','exp','inv','sqrt','square','abs','exp-']
    UnaryData, UnaryName, UnaryUnit, UnaryOperator = copy.deepcopy(X), copy.deepcopy(input_list), copy.deepcopy(D), ['x']*len(input_list)
    pbar = tqdm(range(len(input_list)*len(unary)), desc='Unary Combination', leave=True, file=sys.stdout)
    for i in range(len(input_list)):
        xi, name, unit = X[:,i:i+1], input_list[i], D[:,i:i+1]
        for operator in unary:
            result = UnaryOperation(data=xi, name=name, unit=unit, operator=operator)
            if len(result)==3 and np.isnan(result[0]).sum()==0 and np.isinf(result[0]).sum()==0:
                UnaryData = np.concatenate([UnaryData, result[0]], -1)
                UnaryName.append(result[1])
                UnaryUnit = np.concatenate([UnaryUnit, result[2]], -1)
                UnaryOperator.append(operator)
            pbar.update(1)
    pbar.close()
    # binary operation: x1+x2 or x1-x2. No need for x1*x2 or x1/x2.
    binary = ['+', '-']
    if only_binary:
        BinaryData, BinaryName, BinaryUnit = copy.deepcopy(X), copy.deepcopy(input_list), copy.deepcopy(D)
    else:
        BinaryData, BinaryName, BinaryUnit = copy.deepcopy(UnaryData), copy.deepcopy(UnaryName), copy.deepcopy(UnaryUnit)
    pbar = tqdm(range(int(0.5*len(UnaryName)*(len(UnaryName)-1)*len(binary))), desc='Binary Combination', leave=True, file=sys.stdout)
    for i in range(len(UnaryName)):
        for j in range(i+1,len(UnaryName)):
            for operator in binary:
                if mixed==True or UnaryOperator[i]==UnaryOperator[j]:
                    x1, name1, unit1 = UnaryData[:,i:i+1], UnaryName[i], UnaryUnit[:,i:i+1]
                    x2, name2, unit2 = UnaryData[:,j:j+1], UnaryName[j], UnaryUnit[:,j:j+1]
                    result = BinaryOperation(data1=x1, data2=x2, name1=name1, name2=name2, unit1=unit1, unit2=unit2, operator=operator)
                    if len(result)==3:
                        BinaryData = np.concatenate([BinaryData, result[0]], -1)
                        BinaryName.append(result[1])
                        BinaryUnit = np.concatenate([BinaryUnit, result[2]], -1)
                pbar.update(1)  
    pbar.close()
    print(f'\nGet {len(BinaryName)} Input Combinations: {BinaryName}.')
    return BinaryData, Y, BinaryName, BinaryUnit

def plot(connect_info, correlation_info, ratio_info, input_dim):    
    p, s, pos = input_dim, len(connect_info), [(1, 4)]
    G = nx.DiGraph()

    # input, latent and output variables
    for i in range(1, p+1):
        G.add_node(i, desc='x'+sub[i])
        pos.append((1, p+1-i))
    for i in range(1, s+1):
        G.add_node(p+i, desc='z'+sub[i])
        pos.append((3, p/2.+s/2.-i+1))
    G.add_node(p+s+1, desc='y')
    pos.append((5, p/2.+0.5))

    edge_weights = []
    normal_edges = []
    terminal_edges = []

    # connection informations
    for i in range(len(connect_info)):
        for j in range(len(connect_info[i])):
            x, z = connect_info[i][j], p+i+1
            correlation, ratio = correlation_info[i][j], ratio_info[i][j]
            G.add_edge(x, z, name='{:.2f} ({:.2f})'.format(ratio, correlation), weight=correlation)
            edge_weights.append(correlation)
            normal_edges.append((x, z))
    for z in range(p+1, p+s+1):
        G.add_edge(z, p+s+1, name='', weight=0)
        terminal_edges.append((z, p+s+1))

    plt.figure(figsize=(8, 6))

    nx.draw_networkx_nodes(G, pos, node_size=600, node_color='lightblue', alpha=1)

    cmap = plt.cm.bwr
    norm = plt.Normalize(vmin=-1, vmax=1)

    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, edge_color=edge_weights, edge_vmin=-1, edge_vmax=1, edge_cmap=cmap, width=2) 
    nx.draw_networkx_edges(G, pos, edgelist=terminal_edges, edge_color="black", width=2, arrowsize=5)

    node_labels = nx.get_node_attributes(G, 'desc')
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, font_color='black')

    edge_labels = nx.get_edge_attributes(G, 'name')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_color='black')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, label="Edge Weight (Possibility of Connection)", ax=plt.gca())

    plt.title('Estimation of Latent Structure', fontsize=14)
    plt.axis('off')
    plt.show()


def identify(COR, RATIO, thres:list=[0.1,0.95]):
    p = COR.shape[0]
    # find irrelevant variables
    irrelevant = []
    for i in range(p):
        idx = [j for j in range(p)]
        idx.pop(i)
        if np.isnan(COR[i]).sum()==p or abs(COR[i][idx]).max()<thres[0]:
            irrelevant.append(i+1)
    # find connected nodes
    connect_dict = {i:[] for i in range(1,p+1)}
    correlation_dict = {i:[] for i in range(1,p+1)}
    ratio_dict = {i:[] for i in range(1,p+1)}
    for i in range(p):
        for j in range(p):
            if abs(COR[i][j])>thres[1]:
                connect_dict[i+1].append(j+1)
                correlation_dict[i+1].append(COR[i][j])
                ratio_dict[i+1].append(RATIO[i][j])
    connect_info, correlation_info, ratio_info = [], [], []
    for key in connect_dict.keys():
        if connect_dict[key] not in connect_info and key not in irrelevant:
            connect_info.append(connect_dict[key])
            correlation_info.append(correlation_dict[key])
            ratio_info.append(ratio_dict[key])
    # if connect_info=[[1,2],[3,4],[2,5],[1,2,5]], here [1,2,5] can be removed because {1,2,5}={1,2}+{2,5}
    sets, remove = [set(lst) for lst in connect_info], []
    for comb in combinations(connect_info, 2):
        if set(comb[0]+comb[1]) in sets:
            id = connect_info.index(list(set(comb[0]+comb[1])))
            remove.append(id)
    for id in set(remove):
        connect_info.pop(id), correlation_info.pop(id), ratio_info.pop(id)
    # print the information
    print('\033[1;32m' + '='*60 + ' '*4 + ' Identification of the Latent Structure ' + ' '*4 + '='*60 + '\033[0m')
    data = {'Connection': [], 'Wᵢⱼ/Wᵢₖ': ratio_info, 'Possibility': correlation_info}
    for i in range(len(connect_info)):
        show = '['
        for j in range(len(connect_info[i])):
            show += f'x{sub[connect_info[i][j]]},' if j !=len(connect_info[i])-1 else f'x{sub[connect_info[i][j]]}'
        show += f'] → z{sub[i+1]}'
        data['Connection'].append(show)
    df = pd.DataFrame(data)
    print(df.to_markdown(tablefmt="plain"))
    return connect_info, correlation_info, ratio_info

def estimate(X, Y, etype:str='diff'):
    batch, p = X.shape
    sort_X = [sorted(set(X[:,j])) for j in range(p)]
    idx_X = [{sort_X[i][j]:j for j in range(len(sort_X[i]))} for i in range(p)]

    Y_dict, RHO = {str(list(X[i])):float(Y[i]) for i in range(batch)}, []
    pbar = tqdm(range(batch), desc='Estimating the partial derivatives...', leave=True, file=sys.stdout)
    for i in pbar:
        x, y, rho = list(X[i]), float(Y[i]), []
        for j in range(p):
            xj = x[j]
            id = idx_X[j][xj]
            new_x = copy.deepcopy(x)
            # estimation of partial derivatives using differential
            if etype=='diff':
                if id==0:
                    break
                xj1 = sort_X[j][id-1]
                new_x[j] = xj1
                yj1 = Y_dict[str(new_x)]
                rhoj = (y-yj1)/(xj-xj1)*xj  # (py/pxj)*xj~=(y-yj1)/(xj-xj1)*xj
                rho.append(rhoj)
            # estimation of partial derivatives using polynomial regression
            else:
                xj1s, yj1s = [sort_X[j][min(len(sort_X[j])-1,max(0,id+k))] for k in range(-5,6)], []
                for xj1 in xj1s:
                    new_x[j] = xj1
                    yj1s.append(Y_dict[str(new_x)])
                poly = np.polynomial.chebyshev.Chebyshev.fit(np.array(xj1s), np.array(yj1s), deg=3)
                rhoj = poly.deriv(m=1)(xj)*xj
                rho.append(rhoj)
        if len(rho)==p:
            RHO.append(rho)
        pbar.update(1)

    # correlation coefficient and estimated ratios of weights
    RHO, COR, RATIO = np.array(RHO), np.zeros((p,p)), np.zeros((p,p))
    for i in range(p):
        for j in range(p):
            rhoi, rhoj = RHO[:,i].flatten(), RHO[:,j].flatten()
            COR[i,j] = np.corrcoef(rhoi, rhoj)[0, 1] if min(abs(rhoi).max(),abs(rhoj).max())>1e-10 and min(abs(rhoi).mean(),abs(rhoj).mean())>1e-3 else np.nan
            # RATIO[i,j] = (rhoi/rhoj).mean()
            RATIO[i,j] = (rhoi*rhoj-rhoi.mean()*rhoj.mean()).sum()/(rhoj**2-rhoj.mean()**2).sum()
    return COR, RATIO