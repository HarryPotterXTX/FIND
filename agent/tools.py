import numpy as np
import pandas as pd
from langchain_tavily import TavilySearch
from langchain_core.tools import InjectedToolCallId, tool
from sklearn.ensemble import RandomForestRegressor
from omegaconf import OmegaConf

from typing import List, Dict, Any

import os, shap, sys, shutil
sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
from utils.Logger import reproduc
from utils.Identify import InputCombination

@tool
def identify_input(data_path:str, output:str) -> str:
    """
    Use SHAP to analyze the data and identify the top variables that contribute the most to the output.

    Args:
        data_path (str): Dataset path.
        output (str): Name of the variable to be predicted.
       
    Returns:
        sorted_input_list (List[str]): List of input variables sorted by their contribution to the output.
        sorted_shap_list (List[str]): List of SHAP values normalized by dividing by the maximum SHAP value.
        next: Recommendation for next step processing.
    """
    data = pd.read_csv(data_path)
    input_list = data.select_dtypes(include=['int', 'float']).columns.tolist()
    if 'Unnamed: 0' in input_list:
        input_list.remove('Unnamed: 0')
    input_list.remove(output)
    output_list = [output]
    D = 0
    X, Y, input_list, units = InputCombination(data_path, input_list, output_list, D, origin=True, mixed=True, only_binary=True)

    model = RandomForestRegressor(random_state=42)
    model.fit(X, Y)

    explainer = shap.TreeExplainer(model)
    test = X[np.random.choice(X.shape[0], min(200,X.shape[0]), replace=False)]
    shap_values = explainer.shap_values(test)

    sorted_pairs = sorted(zip(abs(shap_values).mean(0).tolist(), input_list), key=lambda x: x[0], reverse=True)
    sorted_input_list = [item[1] for item in sorted_pairs]
    sorted_shap_list = [item[0] for item in sorted_pairs]
    sorted_shap_list = [s/max(sorted_shap_list) for s in sorted_shap_list]

    shap.summary_plot(shap_values, test, feature_names=input_list, plot_type="bar", show=True)
    # shap.summary_plot(shap_values, test, feature_names=input_list, show=True)

    prompt = 'Ask the user what to do next. Suggested question: Shall I proceed \
        to set the input list as the top 5 variables with the highest contribution, \
        or would you prefer to manually select which variables to use as inputs?'.replace('  ','')
    return sorted_input_list, sorted_shap_list, prompt

@tool
def understand():
    """
    Understand the meaning of each configuration parameter according to the instructions in the opt/config.yaml file.
    Afterwards, if the user has requirements, modify the corresponding parameters accordingly.

    Args:
        
    Returns:
        prompt (str): Prompt for LLM to understand how to modify the parameters according to user requirements.
        next (str): Recommendation for next step processing.
    """
    prompt = 'Read and remember the following configuration instructions. If the user \
        has any needs, modify the corresponding variables accordingly.\n'.replace('  ','') \
        + str(OmegaConf.load('opt/config.yaml'))
    next = 'Ask the user what to do next. Suggested question: You can provide \
        the location of the dataset and the variable to be predicted, and then \
        we will proceed to the next step.'.replace('  ','')
    opt = OmegaConf.load('opt/agent_template.yaml')
    OmegaConf.save(opt, 'opt/agent_instance.yaml', resolve=True)
    return prompt, next

@tool
def modify_parameter(data_path:str, output:str, keys:List[str], value:Any):
    """
    Modify the corresponding parameters in the opt/agent_instance.yaml.

    Args:
        data_path (str): dataset csv path.
        output (str): the variable to be predicted.
        keys (List[str]): Key list of the variable.
        value (Any): The new value to which the variable is to be modified. 

        eg. keys=['Dataset','data_path], value='data.csv' -> opt.Dataset.data_path='data.csv'
        
    Returns:
        state (bool): Return True if the modification is successful, otherwise return False.
    """
    def set_nested_attr(obj, keys, value):
        if len(keys) == 1:
            setattr(obj, keys[0], value)
        else:
            current_obj = getattr(obj, keys[0])
            set_nested_attr(current_obj, keys[1:], value)

    config_path = 'opt/agent_instance.yaml'
    opt = OmegaConf.load(config_path)
    # Get dimensional matrix
    unit_path = os.path.join('dataset_units', os.path.basename(data_path).split('.')[0]+'.yaml')
    if os.path.exists(unit_path):
        unit_opt = OmegaConf.load(unit_path)
        opt.Dataset.units = unit_opt.units
        opt.Dataset.d = unit_opt.dimensional[output]
        if keys==['Dataset', 'input_list']:
            opt.Dataset.D = np.array([unit_opt.dimensional[var] for var in value]).T.tolist()
    # Fix output dir
    opt.Dataset.data_path = data_path
    opt.Dataset.output_list = [output]
    opt.Log.project_name = 'agent'
    opt.Log.time = False
    set_nested_attr(opt, keys=keys, value=value)
    OmegaConf.save(opt, 'opt/agent_instance.yaml', resolve=True)
    return True

def get_performance_find():
    path = 'outputs/agent/metrics/performance.txt'
    with open(path, 'r+') as f:
        datas = f.readlines()
    for i in range(len(datas)):
        data = datas[i].replace(' ', '')
        d = data.split('|')
    data = datas[4].replace(' ', '').split('|')
    coef, zs, cf, r2, pr = data[1], data[2], data[4], data[5], data[6]
    coef = [float(c) for c in coef[1:-1].split(',')]
    zs = zs.split(',')
    if len(zs)==1:
        z = 'z='+zs[0]
    else:
        z = ''
        for i in range(len(zs)):
            z += 'z'+sub[i+1]+'='+zs[i]+','
        z = z[:-1]
    formula = z + ',' + pr
    return coef, formula, int(cf), float(r2)


sub = {i:chr(ord('\u2080')+i) for i in range(10)}
@tool
def find():
    """
    Use FIND (Formulas IN Data) to discover the formulas after modifing opt/agent_instance.yaml.
    This program will automatically call the command python main.py -p opt/agent_instance.yaml.

    Args:
        
    Returns:
        coeff (List[float]): The top coefficient learned by the model.
        formula (str): The optimal formula learned by the model. eg. z=x₂⁻¹˙⁰x₄⁺²˙⁰, y=0.22+0.93z.
        cf (int): Polynomial complexity. eg. y=1+2x, cf=2.
        r2 (float): The coefficient of determination of the discovered formula. 
        next (str): Recommendation for next step processing.
    """
    try:
        if os.path.exists('outputs/agent'):
            shutil.rmtree('outputs/agent')
        os.system('python main.py -p opt/agent_instance.yaml')
    except Exception as e:
        pass
    # formula
    coef, formula, cf, r2 = get_performance_find()
    # next step
    if cf>5:
        next = f'The obtained optimal coefficient is {coef}, and the formula is {formula}. \
            The formula complexity is {cf}, which is quite complex. Do you want to fix the \
            coefficients (Structure.c2f.fix={[coef]}) and simplify the formula using symbolic \
            regression (Structure.sr.adopt=True)?'.replace('  ','')
    else:
        next = f'The obtained optimal coefficient is {coef}, and the formula is {formula}. \
            Would you like me to help you theoretically derive this formula to verify its \
            rationality and reliability?'.replace('  ','')
    if r2<0.95:
        next = f'The R2 metric of the formula is too low. Please try to analyze the reasons \
            for the failure in data mining.'.replace('  ','')
    return coef, formula, cf, r2, next

def get_performance_sr():
    path = 'outputs/agent_sr/metrics/performance.txt'
    with open(path, 'r+') as f:
        datas = f.readlines()
    for i in range(len(datas)):
        data = datas[i].replace(' ', '')
        d = data.split('|')
    data = datas[4].replace(' ', '').split('|')
    zs, r2, sr = data[2], data[7], data[8]
    zs = zs.split(',')
    if len(zs)==1:
        z = 'z='+zs[0]
    else:
        z = ''
        for i in range(len(zs)):
            z += 'z'+sub[i+1]+'='+zs[i]+','
        z = z[:-1]
    formula = z + ',' + sr
    return formula, float(r2)

@tool
def sr(coef:List[float]):
    """
    Use symbolic regression to simplify the formula.

    Args:
        coef (List[float]): The top coefficient learned by the model.
        
    Returns:
        formula (str): The obtained symbolic expression. eg. z=x₂⁻¹˙⁰x₄⁺²˙⁰, y=0.5*sin(z)+z.
        r2 (float): The coefficient of determination of the discovered formula. 
        next (str): Recommendation for next step processing.
    """
    config_path = 'opt/agent_instance.yaml'
    opt = OmegaConf.load(config_path)
    # Get dimensional matrix
    opt.Structure.c2f.refine_step = []
    opt.Structure.c2f.fix = coef
    opt.Structure.sr = True
    opt.Log.project_name = 'agent_sr'
    OmegaConf.save(opt, 'opt/agent_instance.yaml', resolve=True)

    try:
        if os.path.exists('outputs/agent_sr'):
            shutil.rmtree('outputs/agent_sr')
        os.system('python main.py -p opt/agent_instance.yaml')
    except Exception as e:
        pass

    # formula
    formula, r2 = get_performance_sr()
    # next step
    next = f'The obtained symbolic expression is {formula}. \
        Compare polynomial expressions and symbolic expressions, \
        recommend the best one between the two to the user, and \
        then verify the formula.'.replace('  ','')
    return formula, r2, next

@tool
def verify(formula:str, r2:float):
    """
    Using the built-in knowledge of LLM or online search, perform theoretical derivation and verification of the obtained formula.

    Args:
        formula (str): The optimal formula.
        r2 (float): The coefficient of determination of the discovered formula. 
        
    Returns:
        prompt (str): Prompt for LLM to verify the formula.
    """
    config_path = 'opt/agent_instance.yaml'
    opt = OmegaConf.load(config_path)
    input_list = opt.Dataset.input_list
    output = opt.Dataset.output_list[0]
    if r2>0.95:
        prompt = f'The formula we discovered is {formula}, \
            where y is the {output}, x corresponds to {input_list}. \
            Based on your background knowledge, derive and verify \
            the rationality of this formula.'.replace('  ','')
    else:
        prompt = f'The R2 metric of the formula is too low. Please try \
            to analyze the reasons for the failure in data mining.'.replace('  ','')
    return prompt

@tool
def failure_analysis():
    """
    Analyze the reasons for the failure.

    Args:
        
    Returns:
        prompt (str): Prompt for LLM to analyze the reasons for the failure.
    """
    config_path = 'opt/agent_instance.yaml'
    opt = OmegaConf.load(config_path)
    input_list = opt.Dataset.input_list
    latent_dim = opt.Structure.latent.dim

    prompt = f'Common reasons for poor performance include: too few input variables, \
        insufficient number of set latent variables, excessive noise in the dataset itself, \
        and inherent lack of correlation between inputs and outputs. First, you should \
        analyze whether there is correlation between the inputs and outputs. If a correlation \
        exists, you can then recommend whether to adjust parameters such as the number of input \
        variables or latent variables. The current input variables are {input_list}, and the number of \
        latent variables is {latent_dim}. Alternatively, you can recommend data cleaning procedures.'.replace('  ','')
    return prompt

@tool
def conclusion():
    """
    Summarize the experimental results and generate an experimental report.

    Args:
        
    Returns:
        prompt (str): Prompt for LLM to generate experimental report.
    """
    prompt = f'Summarize the experimental results, including: dataset location, \
        selected inputs and outputs, discovered formulas, corresponding R2 indicators, \
        and formula validation reasoning steps. Generate a paragraph in Word format'.replace('  ','')
    return prompt

tools = [TavilySearch(max_results=2), understand, modify_parameter, identify_input, find, sr, verify, failure_analysis, conclusion]