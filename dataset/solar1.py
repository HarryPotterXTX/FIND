import os, sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))

data =  [['Mercury', 5.852857e10, 5.727818e10, 0.3244425e24, 7605382],
    ['Venus', 10.81012e10, 10.80988e10, 4.861260e24, 19407924,],
    ['Earth', 14.95104e10, 14.94896e10, 5.975000e24, 31557600],
    ['Mars', 22.82995e10, 22.73016e10, 0.6387275e24, 59359846],
    ['Jupiter', 77.82562e10, 77.73441e10, 1902.141e24, 374336251],
    ['Saturn', 142.7208e10, 142.4993e10, 569.4175e24, 929623781],
    ['Uranus', 287.0700e10, 286.7501e10, 87.11550e24, 2651311764],
    ['Neptune', 449.5683e10, 449.5517e10, 103.1285e24, 5200313789]]

def create_dataset():
    csv_path = 'dataset/solar1.csv'
    name = ['Planet', 'a', 'b', 'm', 'T']
    d = {name[j]: [data[i][j] for i in range(len(data))] for j in range(len(name))}
    df = pd.DataFrame(data=d)
    print(df)
    df.to_csv(csv_path)

if __name__=="__main__":
    create_dataset()