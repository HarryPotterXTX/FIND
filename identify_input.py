import os
import sys
import shap
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from utils.Logger import reproduc

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

def main():
    print('\nStart Training Model...')
    model = RandomForestRegressor(random_state=42)
    model.fit(X, Y)

    print('\nStart Explaining the Data...')
    explainer = shap.TreeExplainer(model)
    test = X[np.random.choice(X.shape[0], min(200,X.shape[0]), replace=False)]
    shap_values = explainer.shap_values(test)

    shap.summary_plot(shap_values, test, feature_names=input_list, plot_type="bar", show=True)
    shap.summary_plot(shap_values, test, feature_names=input_list, show=True)

if __name__=="__main__":
    reproduc()
    csv_path = 'dataset/solar.csv'

    input_list = ['Diameter', 'Density', 'Gravity', 'Escape Velocity', 'Rotation Period', 'Length of Day',
        'Distance from Sun', 'Perihelion', 'Aphelion', 'Orbital Period', 'Orbital Velocity']
    output_list = ['Mass']
    D = [[ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 1., -3.,  1.,  1.,  0.,  0.,  1.,  1.,  1.,  0.,  1.],
        [ 0.,  0., -2., -1.,  1.,  1.,  0.,  0.,  0.,  1., -1.]]

    X, Y, input_list, units = InputCombination(csv_path, input_list, output_list, D, origin=True, mixed=True, only_binary=True)

    main()