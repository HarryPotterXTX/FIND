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
    if id==0:
        x_shape = [[1,64,16], [10,128,16], [-16,16,16]]
        x = create_flattened_coords(x_shape)
        x = x[np.random.choice(len(x), min(size,len(x)), replace=False)]
        z = latent(x=x, idxs=[1,2,3], powers=[-1.2, -0.5, 1.0])
        y = np.sin(z)
    elif id==1:
        x_shape = [[1,64,16], [10,128,16], [-16,16,16]]
        x = create_flattened_coords(x_shape)
        x = x[np.random.choice(len(x), min(size,len(x)), replace=False)]
        z = latent(x=x, idxs=[1,2,3], powers=[-1.2, -0.5, 1.0])
        y = 2 + 0.7*z + 1.5*z**2 + 3.6*z**3
    elif id==2:
        x_shape = [[1,3,16], [1,3,16], [1,9,64], [3,5,16]]
        x = create_flattened_coords(x_shape)
        x = x[np.random.choice(len(x), min(size,len(x)), replace=False)]
        z = latent(x=x, idxs=[1,2,3,4], powers=[1.7, -0.7, 0.2, -0.4])
        y = 2 + 1.6*z - 1.8*z**2 + 3.6*z**3
    elif id==3:
        x_shape = [[3,7,8], [1,4,8], [1,9,8], [1,2,8], [1,2,8]]
        x = create_flattened_coords(x_shape)
        x = x[np.random.choice(len(x), min(size,len(x)), replace=False)]
        z = latent(x=x, idxs=[1,2,3,4,5], powers=[-1.7, -0.7, 0.2, -1.3, 1.5])
        y = 6 + 3.6*z - 1.8*z**2 - 1.3*z**3
    elif id==4:
        x_shape = [[1,64,16], [10,128,16], [-16,16,16]]
        x = create_flattened_coords(x_shape)
        x = x[np.random.choice(len(x), min(size,len(x)), replace=False)]
        z = latent(x=x, idxs=[1,2,3], powers=[-1.2, -0.5, 1.0])
        y = np.tanh(np.sin(z+np.pi/3)) + z**2
    # latent_dim=2
    elif id==5:
        x_shape = [[3,7,8], [-4,-1,8], [1,9,8], [1,2,8], [-1,2,8]]
        x = create_flattened_coords(x_shape)
        x = x[np.random.choice(len(x), min(size,len(x)), replace=False)]
        z1 = latent(x=x, idxs=[1,2,3], powers=[-1.7, -1.0, 0.2])
        z2 = latent(x=x, idxs=[4,5], powers=[-1.3, 1.0])
        y = 3 + 0.4*z1 + 1.3*z2 + 0.6*z1*z2 + 1.2*z2**2
    elif id==6:
        x_shape = [[3,7,8], [-4,-1,8], [1,9,8], [1,2,8], [-1,2,8]]
        x = create_flattened_coords(x_shape)
        x = x[np.random.choice(len(x), min(size,len(x)), replace=False)]
        z1 = latent(x=x, idxs=[1,2,3], powers=[-1.7, -1.0, 0.2])
        z2 = latent(x=x, idxs=[4,5], powers=[-1.3, 1.0])
        y = np.sin(2*z1+np.pi/3) - z1*z2 + z2**2
    # latent_dim=3, step: 2.0,1.0,0.5,0.2,0.1
    elif id==7:
        x_shape = [[3,7,8], [-4,-1,8], [1,9,8], [1,9,8], [-1,2,8]]
        x = create_flattened_coords(x_shape)
        x = x[np.random.choice(len(x), min(size,len(x)), replace=False)]
        z1 = latent(x=x, idxs=[1,2], powers=[-1.7, -1.0])
        z2 = latent(x=x, idxs=[3,4], powers=[-1.3, 1.4])
        z3 = latent(x=x, idxs=[5], powers=[1.0])
        y = 3 + 0.4*z1 + 1.3*z2 -0.7*z3 + 0.6*z1*z2 + 1.2*z3**2 + z1*z2*z3
    else:
        raise NotImplemented
    if np.nan in y:
        raise ValueError
    return x, y

def create_dataset(dataset_id):
    # generate simulation dataset
    x, y = toy(id=dataset_id)
    noise = np.random.normal(0, y.std(), y.shape)*0.01
    y = y + noise
    
    input_list = [f'x{i+1}' for i in range(x.shape[-1])]
    output_list = ['y']

    # save data to .csv file
    csv_path = 'dataset/toy.csv'
    data = np.array(torch.concat((x, y),-1))
    name = input_list + output_list
    d = {name[i]: data[:,i] for i in range(data.shape[-1])}
    df = pd.DataFrame(data=d)
    print(df)
    df.to_csv(csv_path)
    
    # write info to .yaml file
    opt = OmegaConf.load('opt/toy.yaml')
    opt.Dataset.data_path = csv_path
    opt.Dataset.input_list = input_list
    opt.Dataset.output_list = output_list
    opt.Dataset.D = 0
    opt.Log.project_name = 'toy' + str(dataset_id)
    OmegaConf.save(opt, 'opt/toy.yaml', resolve=True)

if __name__=="__main__":
    reproduc()
    dataset_id = 0
    create_dataset(dataset_id=dataset_id)