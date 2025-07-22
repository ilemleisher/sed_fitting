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
import math, pickle, sys, sedpy, os
from prospect.io import write_results as writer
from prospect.fitting import fit_model

def get_paths(galaxies):
    if galaxies == 'subfind':
        sed_path = '/blue/narayanan/jkelleyderzon/arepo_runs/UVlum_zooms/pd_clump_tests/pd_clump_CFoff/m50/z4/'
        h5_path = '/orange/narayanan/leisheri/prospector/smuggle/2_h5/'
        output_path = '/orange/narayanan/leisheri/prospector/smuggle/3results/'
    
    elif galaxies == 'caesar':
        sed_path = '/orange/narayanan/leisheri/pd_sed_files/'
        h5_path = '/orange/narayanan/leisheri/prospector/2_h5/'
        output_path = '/orange/narayanan/leisheri/prospector/3results/'
        
    return sed_path,h5_path,output_path

def redshift(snap,galaxies):
    snap_dict = {'134':2.754,'074':6.014,'080':5.530,'087':5.024,'095':4.515,'104':4.015,'115':3.489,'127':3.003,'142':2.496,'160':2.0,'183':1.497,'212':1.007,'252':0.501,'305':0.0}
    smgl_snap_dict = {'001':13.989,'002':13.011,'003':11.980,'004':11.025,'005':9.990,'006':9.000,'007':8.099}

    if galaxies == 'subfind':
        z = smgl_snap_dict[snap]
    
    elif galaxies =='caesar':
        z = snap_dict[snap]
    
    return z

def build_obs(z, pd_dir, **kwargs):
        
    cosmo = FlatLambdaCDM(H0=68, Om0=0.3, Tcmb0=2.725)
    m = ModelOutput(pd_dir)
    wav, lum = m.get_sed(inclination=0,aperture=-1)
    wav  = np.asarray(wav)*u.micron         
    
    if(z<10**-4):
        dl = (10*u.Mpc)
    else:
        dl = cosmo.luminosity_distance(z).to(u.Mpc)
    
    wav = wav.to(u.AA)
    lum = np.asarray(lum)*u.erg/u.s
    flux = lum/(4.*3.14*dl**2.)*(1+z)  #this is where you would include the 1+z term for the flux
    nu = constants.c.cgs/(wav.to(u.cm))
    nu = nu.to(u.Hz)
    flux /= nu
    flux = flux.to(u.Jy)
    maggies = flux / 3631.  
    
    flam = lum/(4.*math.pi*(dl)**2.)/wav/(1+z)
    flam = flam.to(u.erg/u.s/(u.cm**2)/u.AA)
    
    wav = wav*(1+z)
    
    # these filter names / transmission data come from sedpy
    # it's super easy to add new filters to the database but for now we'll just rely on what sedpy already has
    jwst_nircam = ['jwst_f070w', 'jwst_f090w', 'jwst_f115w', 'jwst_f150w', 'jwst_f200w', 
                   'jwst_f277w', 'jwst_f356w', 'jwst_f444w']
    herschel_pacs = ['herschel_pacs_70', 'herschel_pacs_100', 'herschel_pacs_160']
    herschel_spire = ['herschel_spire_250', 'herschel_spire_350', 'herschel_spire_500']
    filternames = (jwst_nircam + herschel_pacs + herschel_spire)
    
    filters_unsorted = load_filters(filternames)
    waves_unsorted = [x.wave_mean for x in filters_unsorted]
    filters = [x for _,x in sorted(zip(waves_unsorted,filters_unsorted))]
        
    gal_phot = sedpy.observate.getSED(np.flip(wav.value),np.flip(flam.value),filterlist = filters,linear_flux = True)
    gal_phot = np.array(gal_phot)
    gal_phot_err = gal_phot * 0.03
    
    obs = {}
    #put some useful things in our dictionary. Prospector exepcts to see, at the least, the filters, photmetry
    #and errors, and if available, the spectrum information. I also include the full powderday SED for easy 
    #access later
    obs['filters'] = filters
    obs['maggies'] = gal_phot
    obs['maggies_unc'] = gal_phot_err
    obs['phot_mask'] = np.isfinite(gal_phot)
    obs['wavelength'] = None
    obs['spectrum'] = None
    obs['pd_sed'] = maggies
    obs['pd_wav'] = wav

    return obs
