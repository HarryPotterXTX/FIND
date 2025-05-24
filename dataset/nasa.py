import os, sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))

def create_dataset():
    # SOLAR
    solar = {
        'Name': ['MERCURY', 'VENUS', 'EARTH', 'MOON', 'MARS', 'JUPITER', 'SATURN', 'URANUS', 'NEPTUNE', 'PLUTO'],
        'Stellar Mass': [1.989*1e30]*10,
        'Stellar Radius': [696300*1e3]*10,
        'Planet Mass': [c*1e24 for c in [0.330, 4.87, 5.97, 0.073, 0.642, 1898, 568, 86.8, 102, 0.0130]],
        'Planet Radius': [c*1e3/2. for c in [4879, 12104, 12756, 3475, 6792, 142984, 120536, 51118, 49528, 2376]],
        'Orbital Radius': [c*1e9 for c in [57.9, 108.2, 149.6, 0.384, 228.0, 778.5, 1432.0, 2867.0, 4515.0, 5906.4]],
        'Orbital Period': [c*24*3600 for c in [88.0, 224.7, 365.2, 27.3, 687.0, 4331, 10747, 30589, 59800, 90560]]
    }
    # TRAPPIST-1
    trappist1 = {
        'Name': ['TRAPPIST-1 b', 'TRAPPIST-1 c', 'TRAPPIST-1 d', 'TRAPPIST-1 e', 'TRAPPIST-1 f', 'TRAPPIST-1 g', 'TRAPPIST-1 h'],
        'Stellar Mass': [1.989*1e30*0.898]*7,
        'Stellar Radius': [696300*1e3*0.1192]*7,
        'Planet Mass': [c*5.97*1e24 for c in [1.374, 1.308, 0.388, 0.692, 1.039, 1.321, 0.326]],
        'Planet Radius': [c*12756*c*1e3/2. for c in [1.116, 1.097, 0.788, 0.920, 1.045, 1.129, 0.755]],
        # 'Orbital Radius': [c*696300*1e3*0.1192 for c in [20.843, 28.549, 40.216, 52.855, 69.543, 84.591, 111.817]],
        'Orbital Radius': [c*149.6*1e9 for c in [0.01154, 0.01580, 0.02227, 0.02925, 0.03849, 0.04683, 0.06189]],
        'Orbital Period': [c*24*3600 for c in [1.510826, 2.421937, 4.049219, 6.101013, 9.207540, 12.352446, 18.772866]]
    }
    # KOI-351
    koi351 = {
        'Name': ['KOI-351 b', 'KOI-351 c', 'Kepler-90 i', 'KOI-351 d', 'KOI-351 e', 'KOI-351 f', 'KOI-351 g', 'KOI-351 h'],
        'Stellar Mass': [1.989*1e30*1.1080]*8,
        'Stellar Radius': [696300*1e3*1.24773]*8,
        'Planet Mass': [None]*8,
        'Planet Radius': [c*12756*c*1e3/2. for c in [1.31, 1.19, 1.32, 2.87, 2.66, 2.88, 8.1, 11.3]],
        # 'Orbital Radius': [c*696300*1e3*1.24773 for c in [13.2, 16.0, 33.8, 56.1, 74.7, 86.4, 127.3, 180.7]],
        'Orbital Radius': [c*149.6*1e9 for c in [0.074, 0.089, 0.1201380843, 0.32, 0.42, 0.48, 0.71, 1.01]],
        'Orbital Period': [c*24*3600 for c in [7.008151, 8.719375, 14.44912, 59.73667, 91.93913, 124.9144, 210.60697, 331.60059]]
    }
    # GJ 667
    gj667= {
        'Name': ['GJ 667 C b', 'GJ 667 C c', 'GJ 667 C f', 'GJ 667 C e', 'GJ 667 C g'],
        'Stellar Mass': [1.989*1e30*0.33]*5,
        'Stellar Radius': [None]*5,
        'Planet Mass': [c*5.97*1e24 for c in [5.6, 3.8, 2.7, 2.7, 4.6]],
        'Planet Radius': [None]*5,
        'Orbital Radius': [c*149.6*1e9 for c in [0.050431, 0.125, 0.156, 0.213, 0.549]],
        'Orbital Period': [c*24*3600 for c in [7.2030, 28.140, 39.026, 62.24, 256.2]]
    }
    # all the data
    nasa = {}
    for k in solar.keys():
        nasa[k] = solar[k] + trappist1[k] + koi351[k] + gj667[k]

    csv_path = 'dataset/nasa.csv'
    df = pd.DataFrame(data=nasa)
    print(df)
    df.to_csv(csv_path)

if __name__=="__main__":
    create_dataset()