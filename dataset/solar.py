# MERCURY 	 VENUS 	 EARTH 	 MOON 	 MARS 	 JUPITER 	 SATURN 	 URANUS 	 NEPTUNE 	 PLUTO 
# Mass (1024kg)	0.330	4.87	5.97	0.073	0.642	1898	568	86.8	102	0.0130
# Diameter (km)	4879	12,104	12,756	3475	6792	142,984	120,536	51,118	49,528	2376
# Density (kg/m3)	5429	5243	5514	3340	3934	1326	687	1270	1638	1850
# Gravity (m/s2)	3.7	8.9	9.8	1.6	3.7	23.1	9.0	8.7	11.0	0.7
# Escape Velocity (km/s)	4.3	10.4	11.2	2.4	5.0	59.5	35.5	21.3	23.5	1.3
# Rotation Period (hours)	1407.6	-5832.5	23.9	655.7	24.6	9.9	10.7	-17.2	16.1	-153.3
# Length of Day (hours)	4222.6	2802.0	24.0	708.7	24.7	9.9	10.7	17.2	16.1	153.3
# Distance from Sun (106 km)	57.9	108.2	149.6	0.384*	228.0	778.5	1432.0	2867.0	4515.0	5906.4
# Perihelion (106 km)	46.0	107.5	147.1	0.363*	206.7	740.6	1357.6	2732.7	4471.1	4436.8
# Aphelion (106 km)	69.8	108.9	152.1	0.406*	249.3	816.4	1506.5	3001.4	4558.9	7375.9
# Orbital Period (days)	88.0	224.7	365.2	27.3*	687.0	4331	10,747	30,589	59,800	90,560
# Orbital Velocity (km/s)	47.4	35.0	29.8	1.0*	24.1	13.1	9.7	6.8	5.4	4.7
# Orbital Inclination (degrees)	7.0	3.4	0.0	5.1	1.8	1.3	2.5	0.8	1.8	17.2
# Orbital Eccentricity	0.206	0.007	0.017	0.055	0.094	0.049	0.052	0.047	0.010	0.244
# Obliquity to Orbit (degrees)	0.034	177.4	23.4	6.7	25.2	3.1	26.7	97.8	28.3	119.5
# Mean Temperature (C)	167	464	15	-20	-65	-110	-140	-195	-200	-225
# Surface Pressure (bars)	0	92	1	0	0.01	Unknown*	Unknown*	Unknown*	Unknown*	0.00001
# Number of Moons	0	0	1	0	2	95	146	28	16	5
# Ring System?	No	No	No	No	No	Yes	Yes	Yes	Yes	No
# Global Magnetic Field?	Yes	No	Yes	No	No	Yes	Yes	Yes	Yes	Unknown

import os, sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))

def create_dataset():
    csv_path = 'dataset/solar.csv'
    d = {'Name': ['MERCURY', 'VENUS', 'EARTH', 'MOON', 'MARS', 'JUPITER', 'SATURN', 'URANUS', 'NEPTUNE', 'PLUTO'],
        'Mass': [c*1e24 for c in [0.330, 4.87, 5.97, 0.073, 0.642, 1898, 568, 86.8, 102, 0.0130]],
        'Diameter': [c*1e3 for c in [4879, 12104, 12756, 3475, 6792, 142984, 120536, 51118, 49528, 2376]],
        'Density': [5429, 5243, 5514, 3340, 3934, 1326, 687, 1270, 1638, 1850],
        'Gravity': [3.7, 8.9, 9.8, 1.6, 3.7, 23.1, 9.0, 8.7, 11.0, 0.7],
        'Escape Velocity': [c*1e3 for c in [4.3, 10.4, 11.2, 2.4, 5.0, 59.5, 35.5, 21.3, 23.5, 1.3]],
        'Rotation Period': [c*3600 for c in [1407.6, -5832.5, 23.9, 655.7, 24.6, 9.9, 10.7, -17.2, 16.1, -153.3]],
        'Length of Day': [c*3600 for c in [4222.6, 2802.0, 24.0, 708.7, 24.7, 9.9, 10.7, 17.2, 16.1, 153.3]],
        'Distance from Sun': [c*1e9 for c in [57.9, 108.2, 149.6, 0.384, 228.0, 778.5, 1432.0, 2867.0, 4515.0, 5906.4]],
        'Perihelion': [c*1e9 for c in [46.0, 107.5, 147.1, 0.363, 206.7, 740.6, 1357.6, 2732.7, 4471.1, 4436.8]],
        'Aphelion': [c*1e9 for c in [69.8, 108.9, 152.1, 0.406, 249.3, 816.4, 1506.5, 3001.4, 4558.9, 7375.9]],
        'Orbital Period': [c*24*3600 for c in [88.0, 224.7, 365.2, 27.3, 687.0, 4331, 10747, 30589, 59800, 90560]],
        'Orbital Velocity':	[c*1e3 for c in [47.4, 35.0, 29.8, 1.0, 24.1, 13.1, 9.7, 6.8, 5.4, 4.7]],
        # 'Orbital Inclination': [7.0, 3.4, 0.0, 5.1, 1.8,1.3, 2.5, 0.8, 1.8, 17.2],
        # 'Orbital Eccentricity':	[0.206, 0.007, 0.017, 0.055, 0.094, 0.049, 0.052, 0.047, 0.010, 0.244],
        # 'Obliquity to Orbit': [0.034, 177.4, 23.4, 6.7, 25.2, 3.1, 26.7, 97.8, 28.3, 119.5],
        # 'Mean Temperature':	[167, 464, 15, -20, -65, -110, -140, -195, -200, -225]
    }
    # # remove the data of MOON and PLUTO
    # for key in d.keys():
    #     d[key].pop(3)
    #     d[key].pop(-1)
    df = pd.DataFrame(data=d)
    print(df)
    df.to_csv(csv_path)

if __name__=="__main__":
    create_dataset()