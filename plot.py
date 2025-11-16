import matplotlib.pyplot as plt
import numpy as np
from prospect.models import priors, transforms
from prospect.models import sedmodel
from prospect.sources import CSPSpecBasis
from prospect.io import read_results as pread
import math, pickle, sys, sedpy
from corner import quantile
from tqdm.auto import tqdm
from sklearn.metrics import r2_score
from astropy.cosmology import FlatLambdaCDM
from hyperion.model import ModelOutput
from astropy import units as u
from astropy import constants
from sedpy.observate import load_filters
import functions as fn

matplotlib.rcParams.update({
    "savefig.facecolor": "w",
    "figure.facecolor" : 'w',
    "figure.figsize" : (10,8),
    "text.color": "k",
    "legend.fontsize" : 20,
    "font.size" : 30,
    "axes.edgecolor": "k",
    "axes.labelcolor": "k",
    "axes.linewidth": 3,
    "xtick.color": "k",
    "ytick.color": "k",
    "xtick.labelsize" : 25,
    "ytick.labelsize" : 25,
    "ytick.major.size" : 12,
    "xtick.major.size" : 12,
    "ytick.major.width" : 2,
    "xtick.major.width" : 2,
    "font.family": 'STIXGeneral',
    "mathtext.fontset" : "cm"
})

def zfrac_to_masses_log(logmass=None, z_fraction=None, agebins=None, **extras):
    sfr_fraction = np.zeros(len(z_fraction) + 1)
    sfr_fraction[0] = 1.0 - z_fraction[0]
    for i in range(1, len(z_fraction)):
        sfr_fraction[i] = np.prod(z_fraction[:i]) * (1.0 - z_fraction[i])
    sfr_fraction[-1] = 1 - np.sum(sfr_fraction[:-1])
    time_per_bin = np.diff(10**agebins, axis=-1)[:, 0]
    mass_fraction = sfr_fraction * np.array(time_per_bin)
    mass_fraction /= mass_fraction.sum()

    masses = 10**logmass * mass_fraction
    return masses

from prospect.sources import FastStepBasis
def build_sps(**kwargs):
    """
    This is our stellar population model which generates the spectra for stars of a given age and mass.
    Because we are using a non parametric SFH model, we do have to use a different SPS model than before
    """
    from prospect.sources import FastStepBasis
    sps = FastStepBasis(zcontinuous=1)
    return sps

galaxies = sys.argv[3]
snap = sys.argv[2]
gal = sys.argv[1]

z = fn.redshift(snap, galaxies)
sed_path,h5_path,output_path=fn.get_paths(galaxies)

if galaxies == 'arepo':
    obs = fn.build_obs(z,sed_path+'gal'+str(gal)+'_sobol_044/snap'+str(snap)+'/snap'+str(snap)+'.galaxy0.rtout.sed')

elif galaxies == 'caesar':
    obs = fn.build_obs(z,sed_path+'snap'+str(snap)+'_galaxy'+str(gal)+'.rtout.sed')

sps = build_sps()

with open(output_path+'snap'+str(snap)+'_gal'+str(gal)+'_props_nonpara.pkl','rb') as file:
    data_props = pickle.load(file)
results, observations_dict, model = pread.results_from(h5_path+'snap_'+str(snap)+'_galaxy_'+str(gal)+'_nonpara_fit.h5')

seds_spec = data_props['spec']
sed_dist = []
for i in tqdm(range(len(sps.wavelengths*(1+z)))):
    sed_dist.append(quantile([item[i] for item in seds_spec], [0.16, 0.5, 0.84]))

plt.figure(figsize=(12, 11))
plt.title('Gal '+str(gal)+' at z '+str(np.round(z,decimals=2)))
plt.loglog(sps.wavelengths*(1+z), [item[1] for item in sed_dist], color='cornflowerblue', lw=2, label='Prospector SED')
plt.fill_between(sps.wavelengths*(1+z), [item[0] for item in sed_dist], [item[2] for item in sed_dist], color='cornflowerblue', alpha=0.3)

plt.plot(observations_dict['pd_wav'], observations_dict['pd_sed'], color='black', ls=':', lw=3, label='Powderday SED')
plt.scatter([x.wave_mean for x in observations_dict['filters']], observations_dict['maggies'],
            color='black', marker='o', s=100, alpha=0.6,zorder=10, label="'Observed' photometry")
plt.xlim([1e2, 1e8])
plt.ylim(bottom=1e-15)
plt.ylabel('Flux [maggies]')
plt.xlabel('Wavelength [$\AA$]')
plt.legend()
plt.savefig("prospector_fit_"+str(galaxies)+"_snap_"+str(snap)+"_gal_"+str(gal)+".png")
