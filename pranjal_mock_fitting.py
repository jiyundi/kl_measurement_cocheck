import joblib
from   astropy import wcs
import astropy.units as u
import numpy as np
import galsim
from ultranest_sampler import UltranestSampler
import matplotlib.pyplot as plt

import utils as utils
from parameters import Parameters
from spec_model  import SlitModel
from image_model import ImageModel
from mock import Mock

#--------------------------------- 1. Set meta data for Image and Spectrum ---------------------------------#
## For object
REDSHIFT = 0.6
beta = 0.  # Angle w.r.t. cluster center (in radian)
LOG10_MSTAR = 10.5
LOG10_MSTAR_ERR = 0.0
RA_OBJ, DEC_OBJ = 180.0*u.deg, 32.0*u.deg


# OUTPUT Image Settings
IMAGE_SNR = 80
IMAGE_SHAPE = (51, 51) # (x, y)
SKY_VAR_IMAGE = np.ones(IMAGE_SHAPE)*1500
IMAGE_PIX_SCALE = 0.2 # arcsec/pix
IMAGE_PSF_FWHM = 0.6 # arcsec

# Create WCS
AP_WCS = wcs.WCS(naxis=2)
AP_WCS.wcs.crpix = [np.round(IMAGE_SHAPE[0]/2), np.round(IMAGE_SHAPE[1]/2)]  # Central pixel
AP_WCS.wcs.crval = [180, 32]  # RA, Dec at central pixel
AP_WCS.wcs.ctype = ['RA---TAN' , 'DEC--TAN']  # Projection, see: https://docs.astropy.org/en/stable/wcs/supported_projections.html
AP_WCS.wcs.cdelt = [1, 1]
AP_WCS.wcs.pc = np.array([[-IMAGE_PIX_SCALE/3600, 0],
                         [0, IMAGE_PIX_SCALE/3600]]) # convert arcsec/pix -> deg/pix
GALSIM_WCS = galsim.AstropyWCS(wcs=AP_WCS)


# OUTPUT Spec Settings
SPEC_SNR = 40
SPEC_SHAPE = (40, 100) # (x, lambda)
SKY_VAR_SPEC = np.ones(SPEC_SHAPE)*40
SPEC_PIX_SCALE = 0.1 # arcsec/pix


# Create wavelength grid
LAMBDA_wav = 656.2819*(1+REDSHIFT)  # For Halpha in nm
LAMBDA_SCALE = 0.03  # nm/pix
LAMBDA_1D = utils.build_1d_grid(SPEC_SHAPE[1], LAMBDA_SCALE) + LAMBDA_wav
LAMBDA_GRID = np.repeat([LAMBDA_1D], SPEC_SHAPE[0], axis=0)*u.nm
cont_model_spec = np.zeros(SPEC_SHAPE)

#
SLIT_WIDTH = 1
SLIT_LEN = 6
slit_LPA = 0.0 # in deg
slit_LPA2 = 90. # in deg
#------------------------------- 2. Setup Parameters -------------------------------#
# Common between image & spec
gammat = 0.05
cosi = 0.4
theta_int = 0.  # in radian

# Image parameters
r_hl_disk = 1.
dx_disk = 0.
dy_disk = 0.
flux = 5
r_hl_bulge = 0.0
flux_bulge = 0.
dx_bulge = 0.
dy_bulge = 0.

# Spec parameters
vcirc =  10**((LOG10_MSTAR  - 1.718) / 3.869)  # Hack for now; should call a function to get vcirc
vscale = 0.5  # in arcsec
v_0 = 0.  # in km/s
dx_vel = 0.  # fraction of r_hl_disk
I01 = 12  # arbitray units
bkg_level = 0.  # arbitrary units
#------------------------------- 3. Create datavector -------------------------------#
mock_params = {
    'shared_params-gamma_t': gammat,
    'shared_params-vcirc': vcirc,
    'shared_params-cosi': cosi,
    'shared_params-theta_int': theta_int,
    'shared_params-r_hl_disk': r_hl_disk,
    'shared_params-dx_disk': dx_disk,
    'shared_params-dy_disk': dy_disk,
    'shared_params-flux': flux,
    'shared_params-r_hl_bulge': r_hl_bulge,
    'shared_params-flux_bulge': flux_bulge,
    'shared_params-dx_bulge': dx_bulge,
    'shared_params-dy_bulge': dy_bulge,
    'shared_params-vscale': vscale,
    'Halpha_params-v_0': v_0,
    'Halpha_params-dx_vel': dx_vel,
    'Halpha_params-I01': I01,
    'Halpha_params-bkg_level': bkg_level
}

meta_gal = {'RA': RA_OBJ,
            'Dec': DEC_OBJ,
            'redshift': REDSHIFT,
            'log10_Mstar': LOG10_MSTAR,
            'log10_Mstar_err': LOG10_MSTAR_ERR,
            'beta': beta*u.radian
            }

meta_image = {'ngrid': IMAGE_SHAPE,
            'pixScale': IMAGE_PIX_SCALE,
            'psfFWHM': IMAGE_PSF_FWHM,
            'wcs': GALSIM_WCS,
            'ap_wcs': AP_WCS,
            'RA': RA_OBJ.value,
            'Dec': DEC_OBJ.value}


meta_spec = {'line_species': 'Halpha',  # need i-1 since i is 1-indexed and lines is 0-indexed
            'lambda_grid': LAMBDA_GRID,
            'pixScale': SPEC_PIX_SCALE,
            'ngrid': SPEC_SHAPE,
            'slitRA': RA_OBJ,
            'slitDec': DEC_OBJ,
            'slitWidth': SLIT_WIDTH,
            'slitLen': SLIT_LEN,
            'slitLPA': slit_LPA*u.deg,
            'slitWPA': slit_LPA*u.deg + 90*u.deg,  # Assume rectangular slit
            'rhl': r_hl_disk*0.5
            }


# Now initialize model
params = Parameters({'shared_params':{'beta':meta_gal['beta']}}, line_species=['Halpha'])
updated_dict = params.gen_param_dict(mock_params.keys(), mock_params.values())

spec_model = SlitModel(obj_param=meta_gal, meta_param=meta_spec)

this_line_dict = {**updated_dict['shared_params'], **updated_dict['Halpha_params']}
spec_data = spec_model.get_observable(this_line_dict)
spec_var = SKY_VAR_SPEC
# Set variance to match SNR
spec_var = Mock._set_snr(spec_data, spec_var, SPEC_SNR, 'spec', verbose=False)


image_model = ImageModel(meta_image=meta_image)
image_data = image_model.get_image(updated_dict['shared_params'])
image_var = image_data + SKY_VAR_IMAGE
# Set variance to match SNR
image_var = Mock._set_snr(image_data, image_var, IMAGE_SNR, 'image', verbose=False)

# For image



spec_data_info = {
    'data': spec_data,
    'var': spec_var,
    'cont_model': cont_model_spec,
    'par_meta': meta_spec}

image_data_info = {
    'data': image_data,
    'var': image_var,
    'par_meta': meta_image}

mock_data_info = {
    'spec': [spec_data_info],
    'image': image_data_info,
    'galaxy': meta_gal,
    'fid_params': updated_dict}

# For second slit position angle
meta_spec2 = meta_spec.copy()
meta_spec2['slitLPA'] = slit_LPA2*u.deg
meta_spec2['slitWPA'] = slit_LPA2*u.deg + 90*u.deg  # Assume rectangular slit

spec_model2 = SlitModel(obj_param=meta_gal, meta_param=meta_spec2)

spec_data2 = spec_model2.get_observable(this_line_dict)
spec_var2 = Mock._set_snr(spec_data2, spec_var, SPEC_SNR, 'spec', verbose=False)

print('Final spec1 SNR: ', utils.calculate_spec_snr(spec_data, spec_var))
print('Final spec2 SNR: ', utils.calculate_spec_snr(spec_data2, spec_var2))
print('Final Image SNR: ', utils.calculate_image_snr(image_data, image_var))


spec_data_info2 = {
'data': spec_data2,
'var': spec_var2,
'cont_model': cont_model_spec,
'par_meta': meta_spec2}

mock_data_info['spec'].append(spec_data_info2)

fig = plt.figure(figsize=(15, 4))
gs = fig.add_gridspec(nrows=1, ncols=3, 
                      height_ratios=[1], 
                      width_ratios=[1, 1, 1])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2], 
                      projection=mock_data_info['image']['par_meta']['ap_wcs'])
ax1.imshow(spec_data, origin='lower', cmap='viridis', aspect='auto')
ax1.set_title('Mock Spectrum: slit PA 0.0 deg')
ax2.imshow(spec_data2, origin='lower', cmap='viridis', aspect='auto')
ax2.set_title('Mock Spectrum: slit PA 90 deg')
ax3.imshow(image_data, origin='lower', cmap='viridis', aspect='equal')
ax3.set_title('Mock Image')
fig.suptitle('Mock Galaxy')
fig.savefig('1_pranjal_ipynb.png')


# Load YAML file for config
config_dic = {
    'galaxy_params': {
        'obs_type': 'slit', 
        'line_species': ['Halpha'], 
        'log10_Mstar': None, 
        'log10_Mstar_err': None, 
        'line_profile_path': None # '../zzz_spec_extract_095_OII_T2.pkl'
        }, 
    'likelihood': {
        'fit_image':  True, 
        'fit_spec':   True, 
        'fid_params': None, 
        'set_non_analytic_prior': None
        }, 
    'TFprior': {
        'use_TFprior': True, 
        'log10_vTF':   np.log(142), 
        'sigmaTF':     0, 
        'a':            None, 
        'b':            None, 
        'sigmaTF_intr': None, 
        'relation':     None
        }, 
    'params': {
        'shared_params': {
            'image_snr': {
                'prior': {'min': 24, 'max': 132}, 
                'latex_name': '{\\rm Image SNR}'
                }, 
            'spec_snr': {
                'prior': {'min': 13, 'max': 33}, 
                'latex_name': '{\\rm Spec SNR}'
                }, 
            'gamma_t': {
                'prior': {'min': -0.1, 'max': 0.1}, 
                'latex_name': 'g_+'
                }, 
            'cosi': {
                'prior': {'min': 0, 'max': 1}, 
                'latex_name': '\\cos{(i)}'
                }, 
            'theta_int': {
                'prior': {'min': 0, 'max': 2*np.pi}, 
                'latex_name': '\\theta_{\\mathrm{int}}'
                }, 
            'vscale': {
                'prior': {'min': 0.1, 'max': 3}, 
                'latex_name': 'r_{\\mathrm{vscale}}'
                }, 
            'r_hl_disk': {
                'prior': {'min': 0.15, 'max': 3}, 
                'latex_name': 'r_{\\mathrm{hl, disk}}'
                },
            'r_hl_bulge': {
                'prior': {'min': 0.15, 'max': 3}, 
                'latex_name': 'r_{\\mathrm{hl, bulge}}'
                } 
            }
        }
    }


# ## Path to extracted emission lines
inference = UltranestSampler(mock_data_info, config_dic)

## Output path
save_path = './'
run       = 9.0
isTest    = False
#---------------------------  Run Ultranest --------------------------- #
sampler = inference.run(output_dir=save_path, test_run=isTest, run_num=run)
sampler.print_results()
sampler.plot()
sampler.plot_trace()

# joblib.dump(data_info, f'{save_path}/run{run}/data_info.pkl')
# with open( f'{save_path}/run{run}/config.yaml', 'w') as file:
#     yaml.dump(inference.config.__repr__, file)

# #---------------------------  Load samples --------------------------- #
# samples = sampler.results['samples']

# MC_samples = getdist.MCSamples(samples=samples, names=inference.config.params.names, sampler='nested')

# estimates = sampler.results['maximum_likelihood']['point']

# best_fit_dict = inference.params.gen_param_dict(inference.config.params.names, estimates)

# this_line_dict = {**best_fit_dict['shared_params'], **best_fit_dict[f'{line[0]}_params']}
# spec = inference.spec_model[0].get_observable(this_line_dict)

# input('Finished. Press to continue...:')
# # AttributeError: 'UltranestSampler' object has no attribute 'image_model'
# image = inference.image_model.get_image(best_fit_dict['shared_params'])

# spec_chi2 = 0.5*inference.calc_spectrum_loglike(best_fit_dict)
# image_chi2 = 0.5*inference.calc_image_loglike(best_fit_dict)


# #---------------------------  Plot OII --------------------------- #
# fig, axs = plt.subplots(2, 3, figsize=(20, 12), facecolor='white')
# cmap = plt.cm.binary

# meta_spec = data_info['spec'][0]['par_meta']
# lambda_min, lambda_max = meta_spec['lambda_grid'][0][0], meta_spec['lambda_grid'][0][-1]
# x_min, x_max = inference.spec_model[0].slit_x[0], inference.spec_model[0].slit_x[-1]
# extent = [lambda_min.value, lambda_max.value, x_min, x_max]

# std = inference.var_spec[0]**0.5
# data = inference.data_spec[0]

# spec[std>25] = 0.
# data[std>25] = 0.

# im1 = axs[0, 0].imshow(spec, aspect='auto', extent=extent)
# im2 = axs[0, 1].imshow(data, aspect='auto', extent=extent)
# im3 = axs[0, 2].imshow((spec-data), aspect='auto', extent=extent)


# divider = make_axes_locatable(axs[0, 1])
# cax = divider.append_axes("right", size="5%", pad=0.05)
# cbar = fig.colorbar(im2, cax=cax)
# plt.sca(axs[0, 1])

# divider = make_axes_locatable(axs[0, 2])
# cax = divider.append_axes("right", size="5%", pad=0.05)
# cbar = fig.colorbar(im3, cax=cax)
# plt.sca(axs[0, 2])

# axs[0, 0].text(lambda_min.value+2.5, x_max-1.2, line[0], c='white', fontsize=20, weight='bold')
# axs[0, 2].text(lambda_min.value+2.5, x_max-1.2, r"$\chi^2=$"+f"{spec_chi2:.2f}", c='white', fontsize=20, weight='bold')

# #---------------------------  Plot Image --------------------------- #

# data = inference.data_image[15:-15, 15:-15]
# im1 = axs[1, 0].imshow(image[15:-15, 15:-15], vmin=data.min(), vmax=data.max(), aspect='auto')
# im2 = axs[1, 1].imshow(data, vmin=data.min(), vmax=data.max(), aspect='auto')
# im3 = axs[1, 2].imshow((image[15:-15, 15:-15]-data), aspect='auto')


# divider = make_axes_locatable(axs[1, 1])
# cax = divider.append_axes("right", size="5%", pad=0.05)
# cbar = fig.colorbar(im2, cax=cax)
# plt.sca(axs[1, 1])

# divider = make_axes_locatable(axs[1, 2])
# cax = divider.append_axes("right", size="5%", pad=0.05)
# cbar = fig.colorbar(im3, cax=cax)
# plt.sca(axs[1, 2])

# axs[1, 0].text(x_min+.2, x_max-0.5, "Image", c='white', fontsize=20, weight='bold')
# axs[1, 2].text(x_min+.2, x_max-0.5, r"$\chi^2=$"+f"{image_chi2:.2f}", c='white', fontsize=20, weight='bold')


# for row_id, _ in enumerate(axs):
#     for col_id, subax in enumerate(axs[row_id]):
#         if row_id==0 and col_id in [0, 1, 2]:
#             subax.set(xlabel=r'$\lambda$ [$\AA$]')

#         elif row_id==1 and col_id in [0, 1, 2]:
#             subax.set(xlabel=r'$\Delta$ RA [arcsec]')

#         if col_id !=0:
#             subax.tick_params(axis='y', which='both', left=False, labelleft=False)

#         subax.grid()

# axs[0, 0].set_ylabel('Slit Position [arcsec]')
# axs[1, 0].set_ylabel(r'$\Delta$ Dec [arcsec]')

# axs[0, 0].set_title('Best Fit')
# axs[0, 1].set_title('Observation')
# axs[0, 2].set_title('Residual')

# plt.subplots_adjust(hspace=0.25, wspace=0.1)
# plt.savefig( f'{save_path}/run{run}/best_fit_spec.pdf', dpi=300, bbox_inches='tight')
