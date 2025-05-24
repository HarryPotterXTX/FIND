import warnings
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import os, sys, json
warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
from utils.Logger import reproduc

def create_dataset(dataset_dir, dataset_type, dataset_name, part):
    # load feynman dataset
    dataset_path = os.path.join(dataset_dir, dataset_type, dataset_name)
    print(f'Load data from {dataset_path}')
    dataset = np.loadtxt(dataset_path)
    dataset = dataset[np.random.choice(len(dataset), 5000, replace=False)] if part else dataset
    
    # extract variable names and the real equation
    equ_type = 'BonusEquations' if 'bonus' in dataset_type else 'FeynmanEquations'
    info_csv = equ_type + 'Dimensionless' if 'without' in dataset_type else equ_type
    info_csv = os.path.join(dataset_dir, info_csv+'.csv')
    df = pd.read_csv(info_csv)
    id = int(np.argwhere(df['Filename']==dataset_name)[0,0])
    info = df.loc[df['Filename']==dataset_name]
    num = int(info['# variables'])
    formula = info['Formula'][id]
    input_list = [info[f'v{i+1}_name'][id] for i in range(num)]
    output_list = [info['Output'][id]]
    print(f'real equation: {formula}')
    
    # generate input and output dimension matrices
    unit_path = os.path.join(dataset_dir, 'units.csv')
    df = pd.read_csv(unit_path)
    units = ['m', 's', 'kg', 'T', 'V']
    DX = np.zeros((5, num))
    for i in range(5):
        for j in range(num):
            DX[i,j] = df.loc[df['Variable']==input_list[j]][units[i]]
    DX = [[float(DX[i,j]) for j in range(DX.shape[-1])] for i in range(DX.shape[0])]
    DY = np.zeros((5))
    for i in range(5):
        DY[i] = df.loc[df['Variable']==output_list[0]][units[i]]
    DY = [float(DY[i]) for i in range(len(DY))]
    
    # save data to .csv file
    csv_path = 'dataset/feynman.csv'
    name = input_list + output_list
    d = {name[i]: dataset[:,i] for i in range(dataset.shape[-1])}
    df = pd.DataFrame(data=d)
    print(df)
    df.to_csv(csv_path)
    
    # write info to .yaml file
    opt = OmegaConf.load('opt/feynman.yaml')
    opt.Dataset.data_path = csv_path
    opt.Dataset.input_list = input_list
    opt.Dataset.output_list = output_list
    opt.Dataset.D = DX
    opt.Dataset.d = DY
    opt.Log.project_name = 'feynman' + dataset_name
    OmegaConf.save(opt, 'opt/feynman.yaml', resolve=True)
    print('-'*25 + 'Input Identification Information' + '*'*25)
    print(input_list)
    print(output_list)
    print(DX)
    
    # save real info 
    save_info = {'id': dataset_name, 'equation': formula, 'D': DX, 'd': DY, 'path': dataset_path}
    with open('dataset/feynman.json', 'w+') as f_json:
        json.dump(save_info, f_json, indent=2)
    f_json.close()

if __name__=="__main__":
    reproduc()
    part = True
    dataset_name = 'I.6.2a'
    dataset_name = 'I.6.2'
    # dataset_name = 'I.9.18'
    # dataset_name = 'I.12.4'
    dataset_dir = '../../Dataset/Feynman'
    dataset_type = 'Feynman_with_units'
    create_dataset(dataset_dir, dataset_type, dataset_name, part)