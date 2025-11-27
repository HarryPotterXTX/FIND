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
def identify_input(data_path:str, output:str, exclude:list[str]=[]) -> str:
    """
    Use SHAP to analyze the data and identify the top variables that contribute the most to the output.

    Args:
        data_path (str): Dataset path.
        output (str): Name of the variable to be predicted.
        exclude (list[str]): Variables excluded from input identification.
       
    Returns:
        sorted_input_list (List[str]): List of input variables sorted by their contribution to the output.
        sorted_shap_list (List[str]): List of SHAP values normalized by dividing by the maximum SHAP value.
        next: Recommendation for next step processing.
    """
    data = pd.read_csv(data_path)
    input_list = data.select_dtypes(include=['int', 'float']).columns.tolist()
    if 'Unnamed: 0' in input_list:
        input_list.remove('Unnamed: 0')
    for ex in exclude:
        input_list.remove(ex)
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

    prompt = """Ask the user what to do next. Suggested question: Shall I proceed 
to set the input list as the top 6 variables with the highest contribution, 
or would you prefer to manually select which variables to use as inputs?"""
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
    prompt = """Read and remember the following configuration instructions. If the user 
has any needs, modify the corresponding variables accordingly.\n""" + str(OmegaConf.load('opt/config.yaml'))
    next = """Ask the user what to do next. Suggested question: You can provide 
the location of the dataset and the variable to be predicted, and then 
we will proceed to the next step."""
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
        next = f"""The obtained optimal coefficient is {coef}, and the formula is {formula}. 
The formula complexity is {cf}, which is quite complex. Do you want to fix the 
coefficients (Structure.c2f.fix={[coef]}) and simplify the formula using symbolic 
regression (Structure.sr.adopt=True)?"""
    else:
        next = f"""The obtained optimal coefficient is {coef}, and the formula is {formula}. 
Would you like me to help you theoretically derive this formula to verify its 
rationality and reliability?"""
    if r2<0.95:
        next = f"""The R2 metric of the formula is too low. Please try to analyze the reasons 
for the failure in data mining."""
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
    next = f"""The obtained symbolic expression is {formula}. 
Compare polynomial expressions and symbolic expressions, 
recommend the best one between the two to the user, and 
then verify the formula."""
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
        prompt = f"""The formula we discovered is {formula}, where y is the {output}, 
x corresponds to {input_list}. Based on your background knowledge, derive and verify 
the rationality of this formula."""
    else:
        prompt = f"""The R2 metric of the formula is too low. Please try to analyze 
the reasons for the failure in data mining."""
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

"""
Report Generator Tool for Scientific Experiments
"""
from datetime import datetime
import pandas as pd
from typing import Dict, List, Any, Optional

class ScientificReportGenerator:
    """
    A tool for generating scientific experiment reports
    """
    
    def __init__(self, experiment_name: str, researcher: str = "AI Research Agent"):
        """
        Initialize the report generator
        
        Args:
            experiment_name: Name of the experiment
            researcher: Name of the researcher or agent
        """
        self.experiment_name = experiment_name
        self.researcher = researcher
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.sections = {
            "experiment_objective": "",
            "experiment_preparation": {"data_source": "", "input_variables": "", "target_variable": ""},
            "experiment_results": {"discovered_formulas": []},
            "experiment_analysis": {"derivation_verification": "", "failure_analysis": ""}
        }
    
    def set_experiment_objective(self, objective: str):
        """Set the experiment objective"""
        self.sections["experiment_objective"] = objective
    
    def set_experiment_preparation(self, data_source: str, input_variables: List[str], target_variable: str):
        """Set experiment preparation details"""
        self.sections["experiment_preparation"] = {
            "data_source": data_source,
            "input_variables": ", ".join(input_variables),
            "target_variable": target_variable
        }
    
    def add_discovered_formula(self, formula: str, accuracy: Optional[float] = None, complexity: Optional[int] = None):
        """Add a discovered formula to results"""
        formula_data = {"formula": formula}
        if accuracy is not None:
            formula_data["accuracy"] = accuracy
        if complexity is not None:
            formula_data["complexity"] = complexity
        self.sections["experiment_results"]["discovered_formulas"].append(formula_data)
    
    def set_experiment_analysis(self, derivation_verification: str, failure_analysis: str = ""):
        """Set experiment analysis and verification"""
        self.sections["experiment_analysis"] = {
            "derivation_verification": derivation_verification,
            "failure_analysis": failure_analysis
        }
    
    def generate_txt_report(self, filename: Optional[str] = 'outputs/agent/logger/report.txt') -> str:
        """
        Generate a text format report
        
        Args:
            filename: Output filename (optional)
            
        Returns:
            Path to the generated file
        """
        if filename is None:
            filename = f"{self.experiment_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 60 + "\n")
            f.write(f"SCIENTIFIC EXPERIMENT REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Researcher: {self.researcher}\n")
            f.write(f"Date: {self.timestamp}\n")
            f.write("\n")
            
            # Experiment Objective
            f.write("1. EXPERIMENT OBJECTIVE\n")
            f.write("-" * 40 + "\n")
            f.write(f"{self.sections['experiment_objective']}\n")
            f.write("\n")
            
            # Experiment Preparation
            f.write("2. EXPERIMENT PREPARATION\n")
            f.write("-" * 40 + "\n")
            prep = self.sections['experiment_preparation']
            f.write(f"Data Source: {prep['data_source']}\n")
            f.write(f"Input Variables: {prep['input_variables']}\n")
            f.write(f"Target Variable: {prep['target_variable']}\n")
            f.write("\n")
            
            # Experiment Results
            f.write("3. EXPERIMENT RESULTS\n")
            f.write("-" * 40 + "\n")
            formulas = self.sections['experiment_results']['discovered_formulas']
            if formulas:
                for i, formula_data in enumerate(formulas, 1):
                    f.write(f"Formula {i}: {formula_data['formula']}\n")
                    if 'accuracy' in formula_data:
                        f.write(f"   Accuracy: {formula_data['accuracy']:.4f}\n")
                    if 'complexity' in formula_data:
                        f.write(f"   Complexity: {formula_data['complexity']}\n")
                    f.write("\n")
            else:
                f.write("No formulas discovered in this experiment.\n")
            f.write("\n")
            
            # Experiment Analysis
            f.write("4. EXPERIMENT ANALYSIS\n")
            f.write("-" * 40 + "\n")
            analysis = self.sections['experiment_analysis']
            f.write("Theoretical Derivation and Verification:\n")
            f.write(f"{analysis['derivation_verification']}\n")
            f.write("\n")
            
            if analysis['failure_analysis']:
                f.write("Failure Analysis:\n")
                f.write(f"{analysis['failure_analysis']}\n")
                f.write("\n")
            
            f.write("=" * 60 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 60 + "\n")
        
        print(f"TXT report generated: {filename}")
        return filename

@tool
def generate_scientific_report(
    experiment_name: str,
    experiment_objective: str,
    data_source: str,
    input_variables: List[str],
    target_variable: str,
    discovered_formulas: List[Dict[str, Any]],
    derivation_verification: str,
    failure_analysis: str = "",
    researcher: str = "Agent FIND"
) -> str:
    """
    Generate a scientific experiment report in specified format
    
    Args:
        experiment_name: Name of the experiment
        experiment_objective: Objective of the experiment
        data_source: Source of the experimental data
        input_variables: List of input variables used
        target_variable: Target variable to predict/discover
        discovered_formulas: List of discovered formulas with metadata
        derivation_verification: Theoretical derivation and verification
        failure_analysis: Analysis of failures if any
        researcher: Name of researcher/agent
        
    Returns:
        Path to the generated report file

    Example1: Planetary data experiment to discover physical formulas
    generate_scientific_report(
        experiment_name="Pysical Laws Discovery from Planetary Data",
        experiment_objective="Discover physical formulas from solar system dataset",
        data_source="dataset/nasa.csv (NASA Planetary Fact Sheet)",
        input_variables=['Mass', 'Diameter', 'Density', 'Escape Velocity', 'Rotation Period', 'Length of Day'],
        target_variable="Gravity",
        discovered_formulas=[
            {
                "formula": '''z=x₂⁻¹˙⁰x₄⁺²˙⁰, y=0.22+0.93z, where y is the gravitational 
acceleration, x₂ is the planetary diameter, and x₄ is the planetary escape velocity.''',
                "accuracy": 0.9959,
            }
        ],
        derivation_verification='''Based on Newtonian gravitational theory, the relationship between surface gravity (g), planetary diameter (D),
and escape velocity (v_e) is fundamentally derived as g = v_e² / D. This is obtained by combining the 
escape velocity formula v_e = √(2GM/R) and the surface gravity formula g = GM/R², and substituting the 
radius R with diameter D (where D = 2R). The given empirical equation y = 0.22 + 0.93z, where z = v_e² / D, 
represents a best-fit approximation of this theoretical relationship using observational data. The slight 
deviation of the slope from 1 and the presence of a small intercept are attributed to practical factors 
such as planetary oblateness, rotational effects, and measurement uncertainties.''',
        failure_analysis="",
        researcher="Agent FIND"
    )
    """
    
    # Initialize report generator
    report = ScientificReportGenerator(experiment_name, researcher)
    
    # Set experiment details
    report.set_experiment_objective(experiment_objective)
    report.set_experiment_preparation(data_source, input_variables, target_variable)
    
    # Add discovered formulas
    for formula_data in discovered_formulas:
        formula = formula_data.get('formula', '')
        accuracy = formula_data.get('accuracy')
        complexity = formula_data.get('complexity')
        report.add_discovered_formula(formula, accuracy, complexity)
    
    # Set analysis
    report.set_experiment_analysis(derivation_verification, failure_analysis)
    
    # Generate report in specified format
    return report.generate_txt_report()


tools = [TavilySearch(max_results=2), understand, modify_parameter, identify_input, find, sr, verify, failure_analysis, generate_scientific_report]