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
parser.add_argument('--inoise')  # Noise
parser.add_argument('--sini')  # sini
parser.add_argument('--g1')  # g1
parser.add_argument('--g2')  # g2
args = parser.parse_args()


fit_pars = ['g1', 'g2', 'vcirc', 'sini', 'theta_int', 'r_hl_image', 'r_hl_spec', 'vscale', 'v_0', 'r_0', 'I0', 'bkg_level']

## Select fit parameters
## For mock data
pars['g1'] = 0.05
pars['g2'] = 0.05
pars['sini'] = 0.6
pars['slitWidth'] = 1.0#.0
pars['r_hl_spec'] = 0.5
pars['r_hl_image'] = 0.5
pars['theta_int'] = np.pi/3
pars['v_0'] = 10.0

pars['linelist']['flux'] = 1/6*refSDSSspec.value / fiber_SDSS  # [unit: erg/s/cm2/nm/arcsec2]

# Set input parameter
for arg in vars(args):
    if getattr(args, arg) is not None and arg not in ['dir_name', 'inoise']:
        pars[arg] = float(getattr(args, arg))
file_name = args.dir_name
inoise = int(args.inoise)

save_path = './emcee_output/parameter_constraints/SNR30_'+file_name+'/' #+sys.argv[1]+'_'+str(np.round(float(sys.argv[2]), 2))+'/' 

data_info = gen_mock_data(pars, inoise=inoise)

start = time.time()
sampler = EmceeSampler(data_info=data_info, fit_par_names=fit_pars)
analyzer = sampler.run_mpi(nwalkers=100, nsteps=50000, outputfile_name=save_path)
print('This took %.2f mins'%((time.time()-start)/60))
