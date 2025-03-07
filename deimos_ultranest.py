import os
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import yaml

import getdist

from ultranest_sampler import UltranestSampler

parser = argparse.ArgumentParser()
parser.add_argument('--TFoffset', type=float, default=0.)
parser.add_argument('--config',   type=str)
parser.add_argument('--mask',     type=str)
parser.add_argument('--slit',     type=str)
parser.add_argument('--test',     type=bool, default=False)
parser.add_argument('--line',                default='OII')
parser.add_argument('--run',      type=int)
args = parser.parse_args()

# isTest      = args.test
# line        = args.line.strip('"').split(',')
# config_file = args.config
# mask        = args.mask
# slit        = args.slit
# run         = args.run
# TF_offset   = args.TFoffset

TF_offset   = 0.0
config_file = 'config.yaml'
mask        = 'C'
slit        = 95
isTest      = False
line        = ['OII']
run         = 1

print(f'Emission lines to be used: {line}')
## Update to location of `data_info`
data_info = joblib.load('../zzz_spec_095_T2.pkl')

# Load YAML file for config
with open(f'../config/{config_file}', 'r') as stream:
	config = yaml.safe_load(stream)


config['galaxy_params']['line_species']    = line
config['galaxy_params']['obs_type']        = 'slit'
config['galaxy_params']['log10_Mstar']     = 10.0 # data_info['galaxy']['log10_Mstar'] + TF_offset
config['galaxy_params']['log10_Mstar_err'] =  0.2 # data_info['galaxy']['log10_Mstar_err']
config['likelihood'   ]['fit_image']       = False # by JD
config['galaxy_params']['line_profile_path'] = '../zzz_spec_extract_095_OII_T2.pkl'

## Path to extracted emission lines
inference = UltranestSampler(data_info, config)

## Output path
save_path = './'
#---------------------------  Run Ultranest --------------------------- #
sampler = inference.run(output_dir=save_path, test_run=isTest, run_num=run)
sampler.print_results()
sampler.plot()
sampler.plot_trace()

joblib.dump(data_info, f'{save_path}/run{run}/data_info.pkl')
with open( f'{save_path}/run{run}/config.yaml', 'w') as file:
	yaml.dump(inference.config.__repr__, file)

#---------------------------  Load samples --------------------------- #
samples = sampler.results['samples']

MC_samples = getdist.MCSamples(samples=samples, names=inference.config.params.names, sampler='nested')

estimates = sampler.results['maximum_likelihood']['point']

best_fit_dict = inference.params.gen_param_dict(inference.config.params.names, estimates)

this_line_dict = {**best_fit_dict['shared_params'], **best_fit_dict[f'{line[0]}_params']}
spec = inference.spec_model[0].get_observable(this_line_dict)

input('Finished. Press to continue...:')
# AttributeError: 'UltranestSampler' object has no attribute 'image_model'
image = inference.image_model.get_image(best_fit_dict['shared_params'])

spec_chi2 = 0.5*inference.calc_spectrum_loglike(best_fit_dict)
image_chi2 = 0.5*inference.calc_image_loglike(best_fit_dict)


#---------------------------  Plot OII --------------------------- #
fig, axs = plt.subplots(2, 3, figsize=(20, 12), facecolor='white')
cmap = plt.cm.binary

meta_spec = data_info['spec'][0]['par_meta']
lambda_min, lambda_max = meta_spec['lambda_grid'][0][0], meta_spec['lambda_grid'][0][-1]
x_min, x_max = inference.spec_model[0].slit_x[0], inference.spec_model[0].slit_x[-1]
extent = [lambda_min.value, lambda_max.value, x_min, x_max]

std = inference.var_spec[0]**0.5
data = inference.data_spec[0]

spec[std>25] = 0.
data[std>25] = 0.

im1 = axs[0, 0].imshow(spec, aspect='auto', extent=extent)
im2 = axs[0, 1].imshow(data, aspect='auto', extent=extent)
im3 = axs[0, 2].imshow((spec-data), aspect='auto', extent=extent)


divider = make_axes_locatable(axs[0, 1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im2, cax=cax)
plt.sca(axs[0, 1])

divider = make_axes_locatable(axs[0, 2])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im3, cax=cax)
plt.sca(axs[0, 2])

axs[0, 0].text(lambda_min.value+2.5, x_max-1.2, line[0], c='white', fontsize=20, weight='bold')
axs[0, 2].text(lambda_min.value+2.5, x_max-1.2, r"$\chi^2=$"+f"{spec_chi2:.2f}", c='white', fontsize=20, weight='bold')

#---------------------------  Plot Image --------------------------- #

data = inference.data_image[15:-15, 15:-15]
im1 = axs[1, 0].imshow(image[15:-15, 15:-15], vmin=data.min(), vmax=data.max(), aspect='auto')
im2 = axs[1, 1].imshow(data, vmin=data.min(), vmax=data.max(), aspect='auto')
im3 = axs[1, 2].imshow((image[15:-15, 15:-15]-data), aspect='auto')


divider = make_axes_locatable(axs[1, 1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im2, cax=cax)
plt.sca(axs[1, 1])

divider = make_axes_locatable(axs[1, 2])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im3, cax=cax)
plt.sca(axs[1, 2])

axs[1, 0].text(x_min+.2, x_max-0.5, "Image", c='white', fontsize=20, weight='bold')
axs[1, 2].text(x_min+.2, x_max-0.5, r"$\chi^2=$"+f"{image_chi2:.2f}", c='white', fontsize=20, weight='bold')


for row_id, _ in enumerate(axs):
	for col_id, subax in enumerate(axs[row_id]):
		if row_id==0 and col_id in [0, 1, 2]:
			subax.set(xlabel=r'$\lambda$ [$\AA$]')

		elif row_id==1 and col_id in [0, 1, 2]:
			subax.set(xlabel=r'$\Delta$ RA [arcsec]')

		if col_id !=0:
			subax.tick_params(axis='y', which='both', left=False, labelleft=False)

		subax.grid()

axs[0, 0].set_ylabel('Slit Position [arcsec]')
axs[1, 0].set_ylabel(r'$\Delta$ Dec [arcsec]')

axs[0, 0].set_title('Best Fit')
axs[0, 1].set_title('Observation')
axs[0, 2].set_title('Residual')

plt.subplots_adjust(hspace=0.25, wspace=0.1)
plt.savefig( f'{save_path}/run{run}/best_fit_spec.pdf', dpi=300, bbox_inches='tight')
