## Imports
from prospect.models import priors, transforms, sedmodel
from prospect.sources import CSPSpecBasis, FastStepBasis
from sedpy.observate import load_filters
from astropy import constants
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from hyperion.model import ModelOutput
import numpy as np
from prospect.io import read_results as pread
from tqdm.auto import tqdm
from corner import quantile
import math, pickle, sys, sedpy
import functions as fn

snap_num = sys.argv[1]
gal_num = sys.argv[2]
prior_setup = sys.argv[3]
galaxies = sys.argv[4]

sed_path,h5_path,output_path = fn.get_paths(galaxies)
    
z = fn.redshift(snap_num,galaxies)

if galaxies == 'subfind':
    obs = fn.build_obs(z,sed_path+'gal'+str(gal_num)+'_ml12/snap'+str(snap_num)+'/snap'+str(snap_num)+'.galaxy0.rtout.sed')

elif galaxies == 'caesar':
    obs = fn.build_obs(z,sed_path+'snap'+str(snap_num)+'.galaxy'+str(gal_num)+'.rtout.sed')

if prior_setup != "priors0":
    results, observations_dict, model = pread.results_from(h5_path+'snap_'+str(snap_num)+'_galaxy_'+str(gal_num)+'_nonpara_'+str(prior_setup)+'_fit.h5')
else:
    results, observations_dict, model = pread.results_from(h5_path+'snap_'+str(snap_num)+'_galaxy_'+str(gal_num)+'_nonpara_fit.h5')
    
sps = pread.get_sps(results)
model_params = model.theta_labels()

seds_spec = []
seds_mag = []
surviving_mass_frac = []
dmass = []

weights = results.get('weights',None) #likelihood values
idx = np.argsort(weights)[-5000:] #select 5000 most likely
for i in tqdm(idx):
    thetas = results['chain'][i]
    spec, mags, mass_frac = model.predict(thetas,sps=sps,obs=observations_dict)
    seds_spec.append(spec)
    seds_mag.append(mags)
    surviving_mass_frac.append(mass_frac)
    dmass.append(sps.ssp.dust_mass)

corrected_stellar_mass_posterior = 10**results['chain'][idx,model_params.index('logmass')] * np.array(surviving_mass_frac)
smass_quan = quantile(np.log10(corrected_stellar_mass_posterior), [.16,.5,.84], weights=results['weights'][idx])
dmass_quan = quantile(np.log10(dmass), [.16, .5, .84], weights=results['weights'][idx])
data_props = {'dust_mass_quan': dmass_quan, 'spec': seds_spec, 'mag': seds_mag, 'stellar_mass_quan':smass_quan}

if prior_setup != "priors0":
    with open(output_path+'snap'+str(snap_num)+'_gal'+str(gal_num)+'_props_nonpara_'+str(prior_setup)+'.pkl', 'wb') as file:
        pickle.dump(data_props, file)
else:
    with open(output_path+'snap'+str(snap_num)+'_gal'+str(gal_num)+'_props_nonpara.pkl', 'wb') as file:
        pickle.dump(data_props, file)
