import os
os.environ["OMP_NUM_THREADS"]="1"

import numpy as np
import time
import argparse

import astropy.units as u
import matplotlib.pyplot as plt
import getdist
from getdist import plots
import joblib

import sys
sys.path.append('../src/')
sys.path.append('../../KLens/')

from ultranest_sampler import UltranestSampler
from tfCube_wrapper import gen_deimos_mock_data
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
linelist['lambda'] = [372.6032, 372.8815, 496.0295, 500.8240, 656.2819]

fiber_SDSS = np.pi * 1.5**2
refSDSSspec = 3.*1e-17 * u.erg/u.second/u.Angstrom/u.cm**2
refSDSSspec = refSDSSspec.to(u.erg/u.second/u.nm/u.cm**2)

linelist['flux'] = refSDSSspec.value / fiber_SDSS  # [unit: erg/s/cm2/nm/arcsec2]

pars['linelist'] = linelist

## Define inpute arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dir_name', default='test_dir', type=str)  # Output file name 
parser.add_argument('--snr', default='30', type=str)
parser.add_argument('--inoise', default=1, type=int)  # Noise
parser.add_argument('--sini')  # sini
parser.add_argument('--g1')  # g1
parser.add_argument('--g2')  # g2
parser.add_argument('--vscale')  # vscale
parser.add_argument('--theta_int')  # vscale
parser.add_argument('--delta_slit', default=0., type=float)  # offset of orthogonal slits from major axis
parser.add_argument('--beta', type=float)  # offset of orthogonal slits from major axis
parser.add_argument('--axis', type=str)
parser.add_argument('--Resolution')
args = parser.parse_args()

fit_pars = {'shared_params': ['gamma_t', 'vcirc', 'sini', 'theta_int', 'r_hl_image', 'flux'], 
            'line_params':['v_0', 'r_hl_spec', 'vscale', 'I01', 'I02', 'bkg_level']}

snr = args.snr
axis = args.axis
delta_slit = args.delta_slit
snr_dict = {'150':1.34, '100':0.99, '50':0.41, '30':0.255}  #For r_hl = 0.75


## Select fit parameters
## For mock data
gamma_t = 0.05
beta = args.beta
pars['g1'] = -gamma_t * np.cos(2*beta)
pars['g2'] = gamma_t * np.sin(2*beta)
pars['sini'] = 0.6
pars['slitWidth'] = 1.0#.0
pars['r_hl_spec'] = 0.75
pars['r_hl_image'] = 0.75
pars['theta_int'] = np.pi/4
pars['v_0'] = 10.0

pars['Resolution'] = 5000. #6000.

# Set input parameter
for arg in vars(args):
    if getattr(args, arg) is not None and arg not in ['dir_name', 'inoise', 'delta_slit', 'snr', 'axis', 'beta']:
        pars[arg] = float(getattr(args, arg))
        print(f'set {arg} value to {pars[arg]}')

fid_pars = {'shared_params':{'gamma_t': gamma_t,
                'sini': pars['sini'],
                'r_hl_image': pars['r_hl_image'],
                'theta_int': pars['theta_int']},
        'line_params':{'Halpha_params':{'r_hl_spec': pars['r_hl_spec'],
                        'v_0': pars['v_0']}}
        }



#flux_factor = snr30_dict[args.Resolution]  #snr_dict[snr]
flux_factor = snr_dict[snr]
pars['linelist']['flux'] = flux_factor*refSDSSspec.value / fiber_SDSS  # [unit: erg/s/cm2/nm/arcsec2]

if getattr(args, 'delta_slit') is not None:
    delta_slit = float(args.delta_slit)

dir_name = args.dir_name
inoise = int(args.inoise)

#save_path = '/xdisk/timeifler/pranjalrs/emcee_output/TFcube_validation/SNR'+snr+'_'+dir_name+'/chain_info'+file_name+'.pkl'
save_path = '/xdisk/timeifler/pranjalrs/emcee_output/TFcube_test/SNR'+snr+'_'+dir_name

data_info = gen_deimos_mock_data(pars, 'Halpha', axis=axis, delta=delta_slit, inoise=0)
data_info['galaxy']['beta'] = beta*u.radian
a, b = 1.718, 3.869
logM = a + b*np.log10(pars['vcirc'])
data_info['galaxy']['Mstar'] = 10**(logM+0.4)

start = time.time()
inference = UltranestSampler(data_info, fit_pars)

## Sampler
sampler = inference.run(output_dir=save_path)
sampler.print_results()
sampler.plot()
sampler.plot_trace()
print('This took %.2f mins'%((time.time()-start)/60))
joblib.dump(data_info, f'{save_path}/data_info.pkl')
