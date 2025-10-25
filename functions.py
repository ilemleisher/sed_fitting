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
        #sed_path = '/blue/narayanan/jkelleyderzon/arepo_runs/UVlum_zooms/pd_clump_tests/pd_clump_CFoff/m50/z4/'
        sed_path = '/orange/narayanan/leisheri/prospector/smuggle/test/'
        h5_path = '/orange/narayanan/leisheri/prospector/smuggle/test/h5/'
        output_path = '/orange/narayanan/leisheri/prospector/smuggle/test/'
    
    elif galaxies == 'caesar':
        sed_path = '/orange/narayanan/leisheri/pd_sed_files/'
        h5_path = '/orange/narayanan/leisheri/prospector/obs/jwst_nircam+herschel_pacs+herschel_spire+galex/h5/'
        output_path = '/orange/narayanan/leisheri/prospector/obs/jwst_nircam+herschel_pacs+herschel_spire+galex/results/'
        
    elif galaxies == 'arepo':
        sed_path = '/blue/narayanan/desika.narayanan/pd_runs/smuggle_sfh_ML/cosmic_sands/m100/z0/sobol_044_production/'
        h5_path = '/orange/narayanan/leisheri/prospector/obs/jwst_nircam+herschel_pacs+herschel_spire+galex/zoomin/h5/'
        output_path = '/orange/narayanan/leisheri/prospector/obs/jwst_nircam+herschel_pacs+herschel_spire+galex/zoomin/results/'
        
    return sed_path,h5_path,output_path

def redshift(snap,galaxies):
    snap_dict = {'047':9.033,'055':7.963,'134':2.754,'074':6.014,'080':5.530,'087':5.024,'095':4.515,'104':4.015,'115':3.489,'127':3.003,'142':2.496,'160':2.0,'183':1.497,'212':1.007,'252':0.501,'305':0.0}
    smgl_snap_dict = {'001':13.989,'002':13.011,'003':11.980,'004':11.025,'005':9.990,'006':9.000,'007':8.099}
    arepo_snap_dict = {'0': 15.0,'1': 14.750511891636476,'2': 14.499070055796654,'3': 14.250876925423212,'4': 13.999250037498127,'5': 13.749262536873157,'6': 13.499057561258518,'7': 13.249073810202336,'8': 12.999720005599888,'9': 12.749484394335212,'10': 12.500742540839747,'11': 12.250298131707964,'12': 12.000520020800831,'13': 11.750223128904755,'14': 11.5,'15': 11.250398137939484,'16': 11.000480019200769,'17': 10.749500646222534,'18': 10.499540018399264,'19': 10.24985937675779,'20': 9.999890001099988,'21': 9.75037626316921,'22': 9.499790004199916,'23': 9.250102501025012,'24': 9.0,'25': 8.750390015600624,'26': 8.500285008550255,'27': 8.249838127832763,'28': 8.000090000900009,'29': 7.749671887304226,'30': 7.4997875053123675'31': 7.250144377526606,'32': 7.0,'33': 6.750135627373479,'34': 6.5001875046876165,'35': 6.2500543754078155,'36': 5.999860002799944,'37': 5.749915626054674,'38': 5.499837504062399,'39': 5.25,'40': 4.9998800023999515,'41': 4.750100626760968,'42': 4.499945000549994,'43': 4.249895002099958,'44': 4.0,'45': 3.7499168764546624,'46': 3.5000450004500046,'47': 3.250074376301585,'48': 3.0,'49': 2.7499531255859297,'50': 2.5000525007875116,'51': 2.2500243751828135,'52': 2.000030000300003,'53': 1.749972500274997,'54': 1.5,'55': 1.2500225002250023,'56': 1.0,'57': 0.7499956250109374,'58': 0.49999250003749984,'59': 0.25,'60': 0.0}

    if galaxies == 'subfind':
        z = smgl_snap_dict[snap]
    
    elif galaxies =='caesar':
        z = snap_dict[snap]
        
    elif galaxies =='arepo':
        z = arepo_snap_dict[snap]
    
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
   # herschel_pacs = []
    herschel_spire = ['herschel_spire_250', 'herschel_spire_350', 'herschel_spire_500']
   # herschel_spire = ['herschel_spire_500']
    galex = ['galex_NUV','galex_FUV']

    filternames = (jwst_nircam + herschel_pacs + herschel_spire + galex)
    
    filters_unsorted = load_filters(filternames)
    waves_unsorted = [x.wave_mean for x in filters_unsorted]
    filters = [x for _,x in sorted(zip(waves_unsorted,filters_unsorted))]
        
    gal_phot = sedpy.observate.getSED(np.flip(wav.value),np.flip(flam.value),filterlist = filters,linear_flux = True)
    gal_phot = np.array(gal_phot)
    gal_phot_err = gal_phot * 0.03
    
    for i in range(len(filters)):
        filter_wave = filters[i].wavelength
        filter_trans = filters[i].transmission
        trans_max = np.max(filter_trans)
        trans_norm = filter_trans/trans_max
        wave_limit = 912*(1+z)
        if(np.min(filter_wave[trans_norm > 10**-3]) < wave_limit):
            gal_phot[i] = np.nan
    
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
