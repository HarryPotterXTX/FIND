import pandas as pd
from utils.Identify import estimate, identify, plot

def main():
    df = pd.read_csv(path)
    X, Y = df[input_list].to_numpy(), df[output_list].to_numpy()
    COR, RATIO = estimate(X, Y, etype)
    print(COR)
    connect, correlation, ratio = identify(COR, RATIO, thres=[0.1, 0.9])
    plot(connect, correlation, ratio, COR.shape[0])

if __name__=="__main__":
    # identify the latent structure of the toy dataset
    path = 'dataset/toy0.csv'
    input_list = ['x1', 'x2', 'x3', 'x4', 'x5']
    input_list = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']
    output_list = ['y']

    # identify the latent structure of the rlc dataset
    path = 'dataset/rlc.csv'
    input_list = ['R', 'L', 'C', 'U', 'w', 'phi0']
    output_list = ['U/I', 'phi0-phi1'][1]

    # derivative estimation method: differential or polynomial fitting
    etype = ['diff', 'pr'][0]
    main()