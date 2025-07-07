import numpy as np
import caesar
import sys
import pickle
import glob, re, os

if len(sys.argv) < 3:
    raise Exception('Missing arguments. Expected: [snap] [para/nonpara] [priors (optional)]')

snap = sys.argv[1]

filepath = '/orange/narayanan/leisheri/prospector/'

caesar_path = '/home/leisheri/caesar_files/'
caesar_file = caesar_path+f"caesar_"+str(snap)+".hdf5"
obj = caesar.load(caesar_file)

data = {}

if len(sys.argv) == 3:
    priors = ""
elif len(sys.argv) == 4:
    priors = "_"+str(sys.argv[3])

filename = "snap"+str(snap)+"_gal*_props_"+str(sys.argv[2])+priors+".pkl"

matching_files = glob.glob(f"{filepath}/{filename}")

gals = []

for file_path in matching_files:
    filename = os.path.basename(file_path)
    match = re.search(rf"snap{snap}_gal(\d+)_props_"+str(sys.argv[2])+priors+"\.pkl", filename)
    if match:
        gals.append(match.group(1))

for gal in gals:
    with open(filepath+"snap"+str(snap)+"_gal"+str(gal)+"_props_"+str(sys.argv[2])+priors+".pkl", 'rb') as f:
        data_props = pickle.load(f)
        
    dust_mass = data_props['dust_mass_quan']
    
    data[str(gal)]=[]
    data[str(gal)].append(dust_mass)
    data[str(gal)].append(np.log10(obj.galaxies[int(gal)].masses['dust'].to("Msun")))

with open(f'/orange/narayanan/leisheri/prospector/snap_'+str(snap)+'_compiled_dmass_'+str(sys.argv[2])+priors+'.pkl', 'wb') as file:
    pickle.dump(data,file)