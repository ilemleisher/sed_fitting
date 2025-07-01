import numpy as np
import caesar
import sys
import pickle

if len(sys.argv) < 6:
    raise Exception('Missing arguments. Arugments are: print/save para/nonpara index/num snap first_gal last_gal*')

data = {}
    
snap = sys.argv[4]
filepath = '/orange/narayanan/leisheri/prospector/'
gal_file = '/orange/narayanan/leisheri/simba/m25n512/snap'+str(snap)+'/snap'+str(snap)+'_gas_gals.txt'

    
if len(sys.argv) == 6:
    gal = sys.argv[5]
    
    if sys.argv[3] == 'index':
        with open(gal_file, "r") as file:
            lines = file.readlines()
            gal_idx = lines[int(gal)].strip()
    elif sys.argv[3] == 'num':
        gal_idx = gal
    
    if sys.argv[2] == "nonpara":
        file_name = filepath+f"snap"+str(snap)+"_gal"+str(gal_idx)+"_props_nonpara.pkl"
        
    elif sys.argv[2] == "para":
        file_name = filepath+f"snap"+str(snap)+"_gal"+str(gal_idx)+"_props_para.pkl"
        
    try:
        with open(file_name, 'rb') as file:
            data_props = pickle.load(file)
        dust_mass = data_props['dust_mass_quan']
        
        if sys.argv[1] == 'print':
            print("Galaxy num "+str(gal_idx)+" inferred dust mass quantiles:\n", dust_mass)
        elif sys.argv[1] == 'save':
            data[str(gal_idx)] = dust_mass
            
    except Exception as e:
        print(e)
    
elif len(sys.argv) == 7:
    first_gal = sys.argv[5]
    last_gal = sys.argv[6]

    gal_indices = []
    if sys.argv[3] == 'index':
        for i in range(int(first_gal),int(last_gal)+1):
            with open(gal_file, "r") as file:
                lines = file.readlines()
                gal_indices.append(lines[i - 1].strip())
                
    elif sys.argv[3] == 'num':
        gal_indices = list(np.arange(int(first_gal),int(last_gal)+1))
        
    for i in gal_indices:
        if sys.argv[2] == "nonpara":
            file_name = filepath+f"snap"+str(snap)+"_gal"+str(i)+"_props_nonpara.pkl"

        elif sys.argv[2] == "para":
            file_name = filepath+f"snap"+str(snap)+"_gal"+str(i)+"_props_para.pkl"
        
        try:
            with open(file_name, 'rb') as file:
                data_props = pickle.load(file)
            dust_mass = data_props['dust_mass_quan']
            
            if sys.argv[1] == 'print':
                print("Galaxy num "+str(i)+" inferred dust mass quantiles:\n", dust_mass)
            elif sys.argv[1] == 'save':
                data[str(i)] = dust_mass

        except Exception as e:
            print(e)

if sys.argv[1] == 'save':
    with open(f'/orange/narayanan/leisheri/prospector/snap_'+str(snap)+'_inferred_dmass.pkl', 'wb') as file:
        pickle.dump(data, file)