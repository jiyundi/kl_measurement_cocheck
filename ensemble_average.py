import os
os.environ["OMP_NUM_THREADS"]="1"

import numpy as np
import time
import argparse

import astropy.units as u
import matplotlib.pyplot as plt
import getdist
from getdist import plots

import sys
sys.path.append('../')
sys.path.append('../../KLens/')

from emcee_sampler import EmceeSampler
from tfCube_wrapper import gen_mock_data
import kl_tools


## TFcube settings

line_species = 'Halpha'

pars = {}
pars['g1'] = 0.05  #0.05
pars['g2'] = 0.05 #0.05
pars['sini'] = 0.6 # 0.5

pars['redshift'] = 0.4

pars['aspect'] = 0.2
pars['r_hl_image'] = 0.5
pars['r_hl_spec'] = 0.5
pars['theta_int'] = 0.  #np.pi/6.

pars['slitWidth'] = 1.00
pars['ngrid'] = 64
pars['image_size'] = 64

pars['norm'] = 0.0

pars['Resolution'] = 5000. #6000.
pars['expTime'] = 60.*30.  # 60.*30.
pars['pixScale'] = .1185#.05925
pars['nm_per_pixel'] = 0.033
pars['throughput'] = 0.29
pars['psfFWHM'] = 0.5

pars['area'] = 3.14 * (1000./2.)**2

pars['vcirc'] = 200.

linelist = np.empty(5, dtype=[('species', np.str_, 16),
                              ('lambda', float),
                              ('flux', float)])
linelist['species'] = ['OIIa', 'OIIb', 'OIIIa', 'OIIIb', 'Halpha']
linelist['lambda'] = [372.7092, 372.9875, 496.0295, 500.8240, 656.461]

fiber_SDSS = np.pi * 1.5**2
refSDSSspec = 3.*1e-17 * u.erg/u.second/u.Angstrom/u.cm**2
refSDSSspec = refSDSSspec.to(u.erg/u.second/u.nm/u.cm**2)

linelist['flux'] = refSDSSspec.value / fiber_SDSS  # [unit: erg/s/cm2/nm/arcsec2]

pars['linelist'] = linelist

## Define inpute arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dir_name')  # Output file name 
parser.add_argument('--file_name')  # Output file name 
parser.add_argument('--inoise')  # Noise
parser.add_argument('--sini')  # sini
parser.add_argument('--g1')  # g1
parser.add_argument('--g2')  # g2
parser.add_argument('--vscale')  # vscale
parser.add_argument('--delta_slit')  # offset of orthogonal slits from major axis
parser.add_argument('--snr')
parser.add_argument('--sigma_TF_intr')
parser.add_argument('--Resolution')
args = parser.parse_args()

fit_pars = ['g1', 'g2', 'vcirc', 'sini', 'theta_int', 'r_hl_image', 'r_hl_spec', 'vscale', 'v_0', 'I0', 'bkg_level']

snr = args.snr
snr_dict = {'150':1.34, '100':0.99, '50':0.41, '30':0.255}  #For r_hl = 0.75
snr30_dict_noiseless = {'2000':0.395 , '5000':0.255, '10000':0.19, '20000':0.145 , '50000':0.115}  #factors for scaling flux based on Resolution
snr30_dict = {'2000': 0.305, '10000':0.196}

## Select fit parameters
## For mock data
pars['g1'] = 0.05
pars['g2'] = 0.05
pars['sini'] = 0.6
pars['slitWidth'] = 1.0#.0
pars['r_hl_spec'] = 0.75
pars['r_hl_image'] = 0.75
pars['theta_int'] = np.pi/3
pars['v_0'] = 10.0

pars['Resolution'] = 5000. #6000.


# Set input parameter
for arg in vars(args):
    if getattr(args, arg) is not None and arg not in ['dir_name', 'file_name', 'inoise', 'delta_slit', 'snr']:
        pars[arg] = float(getattr(args, arg))

flux_factor = snr30_dict[args.Resolution]  #snr_dict[snr]
#flux_factor = snr_dict[snr]
pars['linelist']['flux'] = flux_factor*refSDSSspec.value / fiber_SDSS  # [unit: erg/s/cm2/nm/arcsec2]

delta_slit = 0.0    
if getattr(args, 'delta_slit') is not None:
    delta_slit = float(args.delta_slit)

dir_name = args.dir_name
file_name = args.file_name
inoise = int(args.inoise)

save_path = '/xdisk/timeifler/pranjalrs/emcee_output/ensemble_average/SNR'+snr+'_'+dir_name+'/chain_info'+file_name+'.pkl'

data_info = gen_mock_data(pars, axis=delta_slit, inoise=inoise)

start = time.time()
sampler = EmceeSampler(data_info=data_info, fit_par_names=fit_pars)
analyzer = sampler.run_mpi(nwalkers=30, nsteps=30000, outputfile_name=save_path)
print('This took %.2f mins'%((time.time()-start)/60))
