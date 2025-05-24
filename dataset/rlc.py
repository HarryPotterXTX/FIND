import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os ,sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))

from utils.Samplers import create_flattened_coords

def generate_rlc_data(R, L, C, U, w, phi0, t_max, num_points):
    t = np.linspace(0, t_max, num_points) 
    U_complex = U * np.exp(1j * phi0)
    Z_L = 1j * w * L       
    Z_C = 1 / (1j * w * C) 
    Z_total = R + Z_L + Z_C
    I_complex = U_complex / Z_total
    I_magnitude = np.abs(I_complex)
    I_phase = np.angle(I_complex)
    I_time = I_magnitude * np.sin(w * t + I_phase)
    return t, I_time

def extract_amplitude_phase(t, I_time, w):
    # I_time = I*sin(w*t+phi1) -> phi1 = arcsin(I_time[0]/I) - w*t[0]
    I = abs(I_time).max()
    # phi = np.arcsin(max(-1,min(1,I_time[0]/I)))-w*t[0]
    phi = np.arcsin(I_time[0]/I)-w*t[0] if I_time[1]>I_time[0] else np.pi-(np.arcsin(I_time[0]/I)-w*t[0])
    return I, phi

def plot_rlc_response(t, U, I):
    plt.figure(figsize=(10, 5))
    plt.plot(t, U, label='Input Voltage U(t)', linestyle='-', color='blue')
    plt.plot(t, I, label='Output Current I(t)', linestyle='--', color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.title('RLC Series Circuit Response')
    plt.show()

def create_dataset():
    # generate simulation dataset
    t_max, num_points = 0.5, 1000 
    shape = [[0.5,1.5,5], [0.08,0.15,10], [0.05,0.15,5], [9,11,5], [9,11,5], [0,0.5,5]]
    X = np.array(create_flattened_coords(shape))
    Y = np.zeros((X.shape[0],2))
    for i in range(X.shape[0]):
        R, L, C, U, w, phi0 = list(X[i])
        # get the corresponding current
        t, I_time = generate_rlc_data(R, L, C, U, w, phi0, t_max, num_points)
        # get the amplitude and phase of the current
        I, phi1 = extract_amplitude_phase(t, I_time, w)
        Y[i] = U/I, np.arctan(np.tan(phi0-phi1))
        # Y[i] = np.sqrt(R**2+(w*L-1./(w*C))**2), np.arctan(w*L/R-1/(w*C*R))
        # # plot the result
        # plot_rlc_response(t, U, I)

    # ave data to .csv file
    csv_path = 'dataset/rlc.csv'
    data = np.concatenate([X,Y],-1)

    input_list = ['R', 'L', 'C', 'U', 'w', 'phi0']
    output_list = ['U/I', 'phi0-phi1']
    name = input_list + output_list
    d = {name[i]: data[:,i] for i in range(data.shape[-1])}
    df = pd.DataFrame(data=d)
    print(df)
    df.to_csv(csv_path)

if __name__=="__main__":
    create_dataset()