import numpy as np
import caesar, h5py, sys, pickle
import glob, re, os
import astropy.units as u

if len(sys.argv) < 4:
    raise Exception('Missing arguments. Expected: [snap] [para/nonpara] [subfind/caesar] [priors (optional)]')

snap = sys.argv[1]
source = sys.argv[3]

if source == 'caesar':
    filepath = '/orange/narayanan/leisheri/prospector/3results/'
    caesar_path = '/orange/narayanan/leisheri/simba/m25n512/caesar_cats/'
    caesar_file = caesar_path+f"caesar_"+str(snap)+".hdf5"
    obj = caesar.load(caesar_file)
    
elif source == 'subfind':
    filepath = '/orange/narayanan/leisheri/prospector/smuggle/3results/'

data = {}

if len(sys.argv) == 4:
    priors = ""
elif len(sys.argv) == 5:
    priors = "_"+str(sys.argv[4])

filename = "snap"+str(snap)+"_gal*_props_"+str(sys.argv[2])+priors+".pkl"

matching_files = glob.glob(f"{filepath}/{filename}")

gals = []

for file_path in matching_files:
    filename = os.path.basename(file_path)
    match = re.search(rf"snap{snap}_gal(\d+)_props_"+str(sys.argv[2])+priors+"\.pkl", filename)
    if match:
        gals.append(match.group(1))

for gal in gals:
    
    data[str(gal)]=[]
        
    with open(filepath+"snap"+str(snap)+"_gal"+str(gal)+"_props_"+str(sys.argv[2])+priors+".pkl", 'rb') as f:
        data_props = pickle.load(f)
        
    dust_mass = data_props['dust_mass_quan']
    
    data[str(gal)].append(dust_mass)
    
    if source == 'caesar':
        data[str(gal)].append(np.log10(obj.galaxies[int(gal)].masses['dust'].to("Msun")))
        
    elif source == 'subfind':
        subfind_path = '/blue/narayanan/jkelleyderzon/arepo_runs/UVlum_zooms/m50/z4/gal'+str(gal)+'_ml12_clump/'
        subfind_file = subfind_path+f'output/groups_'+str(snap)+'/fof_subhalo_tab_'+str(snap)+'.0.hdf5'

        with h5py.File(subfind_file, 'r') as file:
            galaxy = file['Subhalo']
            data[str(gal)].append(np.log10(galaxy['SubhaloMassType'][0][3]* 1.e10/0.7))


if source == 'caesar':
    with open(filepath+'snap_'+str(snap)+'_compiled_dmass_'+str(sys.argv[2])+priors+'.pkl', 'wb') as file:
        pickle.dump(data,file)

elif source == 'subfind':
    with open(filepath+'snap_'+str(snap)+'_compiled_dmass_'+str(sys.argv[2])+priors+'.pkl', 'wb') as file:
        pickle.dump(data,file)
