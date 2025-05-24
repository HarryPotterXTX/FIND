import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import os, sys, torch
sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))

from utils.Logger import reproduc
from utils.Samplers import create_flattened_coords

def latent(x, idxs, powers):
    z = 1
    for i in range(len(idxs)):
        id, power = idxs[i]-1, powers[i]
        z = z*x[:,id:id+1]**power if len(x.shape)>=2 else z*x[id:id+1]**power
    return z

def toy(id):
    size = 4096
    # latent_dim=1
    if id==1:
        x_shape = [[3,3.2,5], [-4,-3.8,5], [1,1.2,5], [1,1.2,5], [-1,-0.8,5]]
        x = create_flattened_coords(x_shape)
        # x = x[np.random.choice(len(x), min(size,len(x)), replace=False)]
        z1 = latent(x=x, idxs=[1,2], powers=[-1.7, -1.0])
        z2 = latent(x=x, idxs=[3,4], powers=[-1.2, 1.4])
        z3 = latent(x=x, idxs=[5], powers=[1.0])
        y = 3 + 0.4*z1 + 1.3*z2 -0.7*z3 + 0.6*z1*z2 + 1.2*z3**2 + z1*z2*z3
    elif id==2:
        x_shape = [[3,5,5], [-4,-2,5], [1,3,5], [1,3,5], [-1,1,5]]
        x = create_flattened_coords(x_shape)
        # x = x[np.random.choice(len(x), min(size,len(x)), replace=False)]
        z1 = latent(x=x, idxs=[1,2], powers=[-1.7, -1.0])
        z2 = latent(x=x, idxs=[3,4], powers=[-1.2, 1.4])
        z3 = latent(x=x, idxs=[5], powers=[1.0])
        y = 3 + 0.4*z1 + 1.3*z2 -0.7*z3 + 0.6*z1*z2 + 1.2*z3**2 + z1*z2*z3
    elif id==3:
        x_shape = [[3,3.2,5], [-4,-3.8,5], [1,1.2,5], [1,1.2,5], [-2,-1.8,5], [5,5.2,5], [2,2.2,5]]
        x = create_flattened_coords(x_shape)
        z1 = latent(x=x, idxs=[1,3,7], powers=[-1.7, 0.2, -1.0])
        z2 = latent(x=x, idxs=[2,4], powers=[1.0, -1.3])
        z3 = latent(x=x, idxs=[6,7], powers=[-0.6, 0.7])
        y = np.sin(2*z1+np.pi/3)-z1*z2 + np.exp(z1*z3) + np.sin(z3**2) + z1*z2*z3 + z2**2
    else:
        raise NotImplemented
    if np.nan in y:
        raise ValueError
    return x, y

def create_dataset(dataset_id):
    # generate simulation dataset
    x, y = toy(id=dataset_id)
    noise = np.random.normal(0, y.std(), y.shape)*0.0
    y = y + noise
    
    input_list = [f'x{i+1}' for i in range(x.shape[-1])]
    output_list = ['y']

    # save data to .csv file
    csv_path = 'dataset/toy0.csv'
    data = np.array(torch.concat((x, y),-1))
    name = input_list + output_list
    d = {name[i]: data[:,i] for i in range(data.shape[-1])}
    df = pd.DataFrame(data=d)
    print(df)
    df.to_csv(csv_path)

if __name__=="__main__":
    reproduc()
    dataset_id = 3
    create_dataset(dataset_id=dataset_id)