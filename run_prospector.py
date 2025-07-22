import numpy as np
from prospect.io import write_results as writer
from prospect.fitting import fit_model
import sys, os, math, sedpy
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from sedpy.observate import load_filters
from astropy import constants
from hyperion.model import ModelOutput
from prospect.models import priors, sedmodel
from prospect.sources import FastStepBasis
import functions as fn

snap_num = sys.argv[1]
gal_num = sys.argv[2]
prior_setup = sys.argv[3]
galaxies = sys.argv[4]

sed_path,h5_path,output_path = fn.get_paths(galaxies)

z = fn.redshift(snap_num,galaxies)

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

def build_sps(**kwargs):
    """
    This is our stellar population model which generates the spectra for stars of a given age and mass. 
    Because we are using a non parametric SFH model, we do have to use a different SPS model than before 
    """
    sps = FastStepBasis(zcontinuous=1)
    return sps

def build_model(**kwargs):
    print('building model')
    model_params = []
    cosmo = FlatLambdaCDM(H0=68, Om0=0.3, Tcmb0=2.725)

    if(z<10**-4):
        dl = (10*u.Mpc)
    else:
        dl = cosmo.luminosity_distance(z).to(u.Mpc)
        
    #Unchanging parameters
    model_params.append({'name': "lumdist", "N": 1, "isfree": False,"init": dl.value,"units": "Mpc"})
    model_params.append({'name':'zred','N':1,'isfree':False,'init':z})   
    model_params.append({'name': 'imf_type', 'N': 1,'isfree': False,'init': 2})
    model_params.append({'name': 'duste_umin', 'N': 1,'isfree': True,'init': 1.0,'prior': priors.TopHat(mini=0.1, maxi=25.0)})
    model_params.append({'name': 'duste_qpah', 'N': 1,'isfree': True,'init': 3.0,'prior': priors.TopHat(mini=0.0, maxi=10.0)})  
    model_params.append({'name': 'duste_gamma', 'N': 1,'isfree': True,'init': 0.01,'prior': priors.TopHat(mini=0.0, maxi=1.0)})
    model_params.append({'name': 'add_agb_dust_model', 'N': 1,'isfree': False,'init': 0})
    model_params.append({'name': 'logmass', 'N': 1,'isfree': True,'init': 10.0,'prior': priors.Uniform(mini=9., maxi=12.)})
    model_params.append({'name': 'logzsol', 'N': 1,'isfree': True,'init': -0.5,'prior': priors.Uniform(mini=-1., maxi=0.2)})
    model_params.append({'name': "sfh", "N": 1, "isfree": False, "init": 3})
    model_params.append({'name': "mass", 'N': 3, 'isfree': False, 'init': 1., 'depends_on':zfrac_to_masses_log})
    model_params.append({'name': "agebins", 'N': 1, 'isfree': False,'init': []})
    model_params.append({'name': "z_fraction", "N": 2, 'isfree': True, 'init': [0, 0],'prior': priors.Beta(alpha=1.0, beta=1.0, mini=0.0, maxi=1.0)})    
    
    #Changing parameters
    if prior_setup == "priors0":
        model_params.append({'name': 'dust_type', 'N': 1,'isfree': False,'init': 2,'prior': None})
        model_params.append({'name': 'dust2', 'N': 1,'isfree': True, 'init': 0.1,'prior': priors.ClippedNormal(mini=0.0, maxi=2.0, mean=0.0, sigma=0.3)})
        model_params.append({'name': 'add_dust_emission', 'N': 1,'isfree': False,'init': 1,'prior': None})
    
    elif prior_setup == "priors1":
        model_params.append({'name': 'dust_type', 'N': 1,'isfree': False,'init': 0,'prior': None})
        model_params.append({'name': 'dust2', 'N': 1,'isfree': True, 'init': 0.1,'prior': priors.TopHat(mini=0.0, maxi=5.0)})
        model_params.append({'name': 'dust_index', 'N': 1,'isfree': True, 'init': -0.7,'prior': priors.TopHat(mini=-3.0, maxi=0.0)})
        model_params.append({'name': 'add_dust_emission', 'N': 1,'isfree': False,'init': 1,'prior': None})
        
    elif prior_setup == "priors2":
        model_params.append({'name': 'dust1', 'N': 1,'isfree': True, 'init': 0.1,'prior': priors.TopHat(mini=0.0, maxi=5.0)})
        model_params.append({'name': 'dust1_index', 'N': 1,'isfree': True, 'init': -1.0,'prior': priors.TopHat(mini=-3.0, maxi=0.0)})
        model_params.append({'name': 'dust2', 'N': 1,'isfree': True, 'init': 0.1,'prior': priors.TopHat(mini=0.0, maxi=5.0)})
        model_params.append({'name': 'mwr', 'N': 1,'isfree': True, 'init': 3.1,'prior': priors.TopHat(mini=0.0, maxi=10.0)})
        model_params.append({'name': 'uvb', 'N': 1,'isfree': True, 'init': 1.0,'prior': priors.TopHat(mini=0.0, maxi=5.0)})

    #here we set the number and location of the timebins, and edit the other SFH parameters to match in size
    n = [p['name'] for p in model_params]
    cosmo = FlatLambdaCDM(H0=68, Om0=0.3, Tcmb0=2.725)
    tuniv = np.round(cosmo.age(z).to('Gyr').value,decimals=1)                                                                                                                                                                                                          
    nbins=10
    tbinmax = (tuniv * 0.8) * 1e9 #earliest time bin goes from age = 0 to age = 2.8 Gyr
    lim1, lim2 = 7.47, 8.0 #most recent time bins at 30 Myr and 100 Myr ago                                                                                                                                                                                                 
    agelims = [0,lim1] + np.linspace(lim2,np.log10(tbinmax),nbins-2).tolist() + [np.log10(tuniv*1e9)]
    agebins = np.array([agelims[:-1], agelims[1:]])

    zinit = np.array([(i-1)/float(i) for i in range(nbins, 1, -1)])
    # Set up the prior in `z` variables that corresponds to a dirichlet in sfr
    # fraction. 
    alpha = np.arange(nbins-1, 0, -1)
    zprior = priors.Beta(alpha=alpha, beta=np.ones_like(alpha), mini=0.0, maxi=1.0)

    model_params[n.index('mass')]['N'] = nbins
    model_params[n.index('agebins')]['N'] = nbins
    model_params[n.index('agebins')]['init'] = agebins.T
    model_params[n.index('z_fraction')]['N'] = nbins-1
    model_params[n.index('z_fraction')]['init'] = zinit
    model_params[n.index('z_fraction')]['prior'] = zprior

    model = sedmodel.SedModel(model_params)


    return model

def build_all(pd_dir,**kwargs):

    return (fn.build_obs(z,pd_dir,**kwargs), build_model(**kwargs),
            build_sps(**kwargs))


#parameters that will be passed to dynesty, the posterior sampler. typically can just ignore these / use these defaults
run_params = {'verbose':False,
              'debug':False,
              'output_pickles': True,
              'nested_bound': 'multi', # bounding method                                                                                      
              'nested_sample': 'auto', # sampling method                                                                                      
              'nested_nlive_init': 400,
              'nested_nlive_batch': 200,
              'nested_bootstrap': 0,
              'nested_dlogz_init': 0.05,
              'nested_weight_kwargs': {"pfrac": 1.0},
              }

gal = gal_num
snap = snap_num

if __name__ == '__main__':
    
    if galaxies == 'subfind':
        pd_dir = sed_path+'gal'+str(gal)+'_ml12/snap'+str(snap)+'/snap'+str(snap)+'.galaxy0.rtout.sed'
        
    elif galaxies == 'caesar':
        pd_dir = sed_path+'snap'+str(snap)+'.galaxy'+str(gal)+'.rtout.sed'
        
    obs, model, sps = build_all(pd_dir,**run_params)
    run_params["sps_libraries"] = sps.ssp.libraries
    run_params["param_file"] = __file__
    if prior_setup != "priors0":
        hfile = h5_path+"/snap_"+str(snap)+"_galaxy_"+str(gal)+"_nonpara_"+str(prior_setup)+"_fit.h5"
    else:
        hfile = h5_path+"/snap_"+str(snap)+"_galaxy_"+str(gal)+"_nonpara_fit.h5"
    print('Running fits')
    output = fit_model(obs, model, sps, [None,None],**run_params)
    print('Done. Writing now')
    writer.write_hdf5(hfile, run_params, model, obs,
              output["sampling"][0], output["optimization"][0],
              tsample=output["sampling"][1],
              toptimize=output["optimization"][1])
