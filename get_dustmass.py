import numpy as np
import caesar
import yt
import sys
import pickle

if len(sys.argv) < 5:
    raise Exception('Missing arguments. Arugments are: print/save index/num snap first_gal last_gal')

snap = sys.argv[3]
filepath = '/home/leisheri/caesar_files/'
caesar_file = filepath+f"caesar_"+str(snap)+".hdf5"
obj = caesar.load(caesar_file)
gal_file = '/orange/narayanan/leisheri/simba/m25n512/snap'+str(snap)+'/snap'+str(snap)+'_gas_gals.txt'

data = {}

if len(sys.argv) == 5:
    gal = sys.argv[4]

    if sys.argv[2] == 'index':
        with open(gal_file, "r") as file:
            lines = file.readlines()
            gal_idx = lines[int(gal)].strip()
    
    elif sys.argv[2] == 'num':
        gal_idx = int(gal)
        
    try:
        if sys.argv[1] == 'print':
            print('Galaxy num '+str(gal_idx)+' dust mass:',np.log10(obj.galaxies[int(gal_idx)].masses['dust'].to("Msun")))
        elif sys.arvg[1] == 'save':
            data[str(gal_idx)] = np.log10(obj.galaxies[int(gal_idx)].masses['dust'].to("Msun"))
            
    except Exception as e:
        print(e)


elif len(sys.argv) == 6:
    first_gal = sys.argv[4]
    last_gal = sys.argv[5]

    gal_indices = []
    if sys.argv[2] == 'index':
        gal_file = '/orange/narayanan/leisheri/simba/m25n512/snap'+str(snap)+'/snap'+str(snap)+'_gas_gals.txt'
        for i in range(int(first_gal),int(last_gal)+1):
            with open(gal_file, "r") as file:
                lines = file.readlines()
                gal_indices.append(lines[i - 1].strip())
    
    elif sys.argv[2] == 'num':
        gal_indices = list(np.arange(int(first_gal),int(last_gal)+1))

    for i in gal_indices:
        try:
            if sys.argv[1] == 'print':
                print('Galaxy num '+str(i)+' dust mass:',np.log10(obj.galaxies[int(i)].masses['dust'].to("Msun")))
            elif sys.argv[1] == 'save':
                data[str(i)] = np.log10(obj.galaxies[int(i)].masses['dust'].to("Msun"))
                
        except Exception as e:
            print(e)

if sys.argv[1] == 'save':
    with open(f'/orange/narayanan/leisheri/prospector/snap_'+str(snap)+'_actual_dmass.pkl', 'wb') as file:
        pickle.dump(data, file)