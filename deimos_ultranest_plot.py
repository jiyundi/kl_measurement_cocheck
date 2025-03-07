## For each slit, overplots contours for all emission lines
import argparse
import joblib
import json
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import scipy.stats
import seaborn as sns
from tabulate import tabulate

import astropy.units as u
from astropy.coordinates import SkyCoord
import galsim
import getdist
from getdist import plots
import pygtc

import sys
sys.path.append('../src/')
sys.path.append('../../KLens/')

from kl_model import FitParameters, Parameters
from ultranest_sampler import UltranestSampler

import ipdb

plt.rc('axes', titlesize=22)     # fontsize of the axes title
plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
plt.rc('axes', linewidth=1.0)
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
plt.rc('legend', fontsize=20)    # legend fontsize
plt.rc('grid', alpha=0.18)
plt.rc('grid', color='lightgray')
plt.rc('font', size=13)          # controls default text sizes
plt.rc('mathtext', fontset='stix')
plt.rc('mathtext', rm='BitStream Vera Sans')
plt.rc('mathtext', it='BitStream Vera Sans:italic')
plt.rc('mathtext', bf='BitStream Vera Sans:bold')
plt.rc('font', family='STIXGeneral')
plt.rc('text', usetex=False)
plt.rc('figure', dpi=300)
plt.rc('savefig', bbox='tight')

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str)
parser.add_argument('--triangle', default=False, type=bool)
parser.add_argument('--true_shear', default=False, type=bool)
args = parser.parse_args()

true_shear = args.true_shear

def plot_residuals(inference, data, data_info, line, axs, ilegend):
    estimates = data['maximum_likelihood']['point']

    best_fit_dict = inference.params.gen_param_dict(inference.fit_params.names, estimates)
    vcirc = best_fit_dict['shared_params']['vcirc']
    sini = (1- best_fit_dict['shared_params']['cosi']**2)**0.5
    vsini = vcirc * sini
    
    this_line_dict = {**best_fit_dict['shared_params'], **best_fit_dict[f'{line}_params']}
    spec = inference.spec_model[0].get_spectrum(this_line_dict, inference.meta_spec[0]['slitLPA'])
    image = inference.image_model.get_image(best_fit_dict['shared_params'])

    spec_chi2 = 0.5*inference.calc_spectrum_loglike(best_fit_dict)
    image_chi2 = 0.5*inference.calc_image_loglike(best_fit_dict)

    # Compute variance of the filtered spec
    spec_data = data_info['spec'][0]['data']
    spec_var = data_info['spec'][0]['var']
    cont_model = data_info['spec'][0]['cont_model'][:, np.newaxis]

    im_var = data_info['image']['var']

    n = 5000

    spec_chisq_dist = []
    image_chisq_dist = []

    for i in range(n):
        noise_realization = np.random.normal(loc=np.zeros(spec_var.shape), scale=spec_var**0.5)
        spec_chisq_dist.append(np.sum(noise_realization**2/(spec_data+spec_var + cont_model))/2)


        noise_realization = np.random.normal(loc=np.zeros(im_var.shape), scale=im_var**0.5)
        image_chisq_dist.append(np.sum(noise_realization**2/(im_var)/2))
    
    spec_chisq_dist = np.array(spec_chisq_dist)
    meta_spec = data_info['spec'][0]['par_meta']
    lambda_min, lambda_max = meta_spec['lambda_grid'][0][0], meta_spec['lambda_grid'][0][-1]
    x_min, x_max = inference.spec_model[0].spatial_xx[0][0], inference.spec_model[0].spatial_xx[0][-1]
    extent = [lambda_min.value, lambda_max.value, x_min, x_max]

    std = data_info['spec'][0]['var']**0.5
    data = data_info['spec'][0]['data']
    residual_spec = spec-data

    spec[std>25] = 0.
    data[std>25] = 0.
    
    ## 1. Plot residuals
    ## Plot with origin='lower' otherwise the slit positions are flipped w.r.t. the rotation curve
    im1 = axs[0, 0].imshow(spec, aspect='auto', extent=extent, origin='lower')
    im2 = axs[0, 1].imshow(data, aspect='auto', extent=extent, origin='lower')
    im3 = axs[0, 2].imshow((spec-data), aspect='auto', extent=extent, origin='lower')
    
    ## 2. Plot chi2 dist.
    axs[0, 3].hist(spec_chisq_dist[np.isfinite(spec_chisq_dist)], bins=50)
    axs[0, 3].axvline(spec_chi2, c='r')
    axs[0, 3].set_xlabel('$\chi^2$')
    
    
    # 3. Plot residual dist.
#     axs[0, 4].hist((residual_spec/std).flatten(), bins=20, density=True, histtype='step', linewidth='2', label='residual/std')
#     axs[0, 4].hist((data_info['spec'][0]['data']/std**0.5).flatten(), bins=40, density=True, histtype='step', linewidth='2', label='obs./std')
    
#     x = np.linspace(-5, 5, 100)
#     axs[0, 4].plot(x, scipy.stats.norm.pdf(x, 0, 1), c='k', label='$\mathcal{N}(\mu=0, \sigma=1)$')
    
#     ax[0, 4].plot(np.mean(meta_spec['lambda_grid'], axis=0), np.sum(spec, axis=0), label='best fit')
#     ax[0, 4].plot(np.mean(meta_spec['lambda_grid'], axis=0), np.sum(data, axis=0), label='obs.')

    ## 4. Plot rotation curve
    spatial_x = inference.spec_model[0].spatial_xx[0]
    f = joblib.load(line_prof_path)[line]  # From extract_line_profile.py
    lambda_rest = inference.spec_model[0].line_wav
    mu = f[1][:, 0]
    mu_err = f[5][:, 0]

    rot_curve= ((mu[mu!=0]/lambda_rest[0]).decompose() - 1)*299792.45
    mu_err = ((mu_err[mu!=0]/lambda_rest[0]).decompose())*299792.45
    mu_err = np.clip(mu_err, 0, 50)
    
    vfield = inference.spec_model[0].vfield[0]
    axs[0, 3].imshow(vfield)
    vfield = np.ma.array(vfield, mask= inference.spec_model[0].slit_mask)
    axs[0, 4].plot(spatial_x, np.mean(vfield, axis=0), '--', c='dodgerblue', zorder=9, label='Best fit')
    axs[0, 4].errorbar(spatial_x[mu!=0], rot_curve, yerr=mu_err, c='dodgerblue', alpha=0.5, fmt='o', capsize=3, label='Extracted from obs.')
    
    # Check if it's a doublet
    if np.any(f[1][:, 1] != 0):
        mu = f[1][:, 1]
        mu_err = f[5][:, 1]
        rot_curve= ((mu[mu!=0]/lambda_rest[1]).decompose() - 1)*299792.45
        mu_err = ((mu_err[mu!=0]/lambda_rest[1]).decompose())*299792.45
        mu_err = np.clip(mu_err, 0, 50)

        vfield = inference.spec_model[0].vfield[1]
        vfield = np.ma.array(vfield, mask= inference.spec_model[0].slit_mask)
    
        axs[0, 4].errorbar(spatial_x[mu!=0], rot_curve, yerr=mu_err, c='orangered', alpha=0.5, fmt='o', capsize=3, label='Extracted from obs.')
        axs[0, 4].plot(spatial_x, np.mean(vfield, axis=0), ':', c='orangered', zorder=10)

    
    if vsini>150:
        axs[0, 4].set_ylim(-300, 300)
    
    else:
        axs[0, 4].set_ylim(-150, 150)

    axs[0, 4].set(xlabel='Slit Position [arcsec]', ylabel='L.O.S velocity [km/s]')

                   
    divider = make_axes_locatable(axs[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im2, cax=cax)
    plt.sca(axs[0, 1])

    divider = make_axes_locatable(axs[0, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im3, cax=cax)
    plt.sca(axs[0, 2])

    axs[0, 0].text(lambda_min.value+2.5, x_max-1.2, line, c='white', fontsize=20, weight='bold')
    axs[0, 2].text(lambda_min.value+2.5, x_max-1.2, "$\chi^2=$"+f"{spec_chi2:.2f}", c='white', fontsize=20, weight='bold')

    #---------------------------  Plot Image --------------------------- #

    x_min, x_max = inference.spec_model[0].spatial_xx[0][15], inference.spec_model[0].spatial_xx[0][-15]
    extent = [x_min, x_max, x_min, x_max]

    data = data_info['image']['data'][15:-15, 15:-15]
    residual_im = (image-data_info['image']['data'])

    im1 = axs[1, 0].imshow(image[15:-15, 15:-15], vmin=data.min(), vmax=data.max(), aspect='auto', extent=extent)
    im2 = axs[1, 1].imshow(data, vmin=data.min(), vmax=data.max(), aspect='auto', extent=extent)
    im3 = axs[1, 2].imshow((image[15:-15, 15:-15]-data), aspect='auto', extent=extent)

    axs[1, 3].hist(image_chisq_dist, bins=50);
    axs[1, 3].axvline(image_chi2, c='r')
    axs[1, 3].set_xlabel('$\chi^2$')

#     axs[1, 4].hist((residual_im/im_var**0.5).flatten(), bins=40, density=True, histtype='step', linewidth=2)
#     axs[1, 4].hist((data_info['image']['data']/im_var**0.5).flatten(), bins=40, density=True, histtype='step', linewidth=2)

#     axs[1, 4].plot(x, scipy.stats.norm.pdf(x, 0, 1), c='k')

    divider = make_axes_locatable(axs[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im2, cax=cax)
    plt.sca(axs[1, 1])

    divider = make_axes_locatable(axs[1, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im3, cax=cax)
    plt.sca(axs[1, 2])

    axs[1, 0].text(x_min+.2, x_max-0.5, "Image", c='white', fontsize=20, weight='bold')
    axs[1, 2].text(x_min+.2, x_max-0.5, "$\chi^2=$"+f"{image_chi2:.2f}", c='white', fontsize=20, weight='bold')


    for row_id, _ in enumerate(axs):
        for col_id, subax in enumerate(axs[row_id]):
            if row_id==0 and col_id in [0, 1, 2]:
                subax.set(xlabel='$\lambda$ [$\AA$]')

            elif row_id==1 and col_id in [0, 1, 2]:
                subax.set(xlabel='$\Delta$ RA [arcsec]')

            if col_id not in [0, 4]:
                subax.tick_params(axis='y', which='both', left=False, labelleft=False)

            subax.grid()

    axs[0, 0].set_ylabel('Slit Position [arcsec]')
    axs[1, 0].set_ylabel('$\Delta$ Dec [arcsec]')

    axs[0, 0].set_title('Best Fit')
    axs[0, 1].set_title('Observation')
    axs[0, 2].set_title('Residual')
    axs[0, 4].set_title('Rotation Curve')

    axs[1, 4].axis('off')
    
    if ilegend==0:
        axs[0, 4].legend()

def load_samples(f):
    s = np.loadtxt(f'{f}/chains/equal_weighted_post.txt', delimiter=' ', skiprows=1)
    with open(f'{f}/info/results.json', 'r') as file:
        content = file.read()
        data = json.loads(content)
    
    return s, data

font = {'family': 'DejaVu Sans', 'size':18}
slit_paths = sorted(glob.glob('/xdisk/timeifler/pranjalrs/ultranest/*'))
cmap = plt.cm.binary

run = args.run
save_triangle = args.triangle

a2261_RA, a2261_Dec = 260.612917, 32.133889
abella2261_coord = SkyCoord(a2261_RA*u.deg, a2261_Dec*u.deg, frame='fk5')


# Load gamma maps
path = '/xdisk/timeifler/pranjalrs/KL_data/CLASH_gamma_maps'
gamma1_map = galsim.fits.read(f'{path}/hlsp_clash_model_a2261_zitrin-nfw_v2_gamma1.fits')
gamma1_err_map = galsim.fits.read(f'{path}/hlsp_clash_model_a2261_zitrin-nfw_v2_gamma1-1sigmaerr.fits')

gamma2_map = galsim.fits.read(f'{path}/hlsp_clash_model_a2261_zitrin-nfw_v2_gamma2.fits')
gamma2_err_map = galsim.fits.read(f'{path}/hlsp_clash_model_a2261_zitrin-nfw_v2_gamma2-1sigmaerr.fits')

# Maps from LTM method
gamma1_ltm_map = galsim.fits.read(f'{path}/hlsp_clash_model_a2261_zitrin-ltm-gauss_v2_gamma1.fits')
gamma1_err_ltm_map = galsim.fits.read(f'{path}/hlsp_clash_model_a2261_zitrin-ltm-gauss_v2_gamma1-1sigmaerr.fits')

gamma2_ltm_map = galsim.fits.read(f'{path}/hlsp_clash_model_a2261_zitrin-ltm-gauss_v2_gamma2.fits')
gamma2_err_ltm_map = galsim.fits.read(f'{path}/hlsp_clash_model_a2261_zitrin-ltm-gauss_v2_gamma2-1sigmaerr.fits')

# Best emission lines to plot
with open('/home/u31/pranjalrs/Github_repos/kl_deimos/data/best_targets_final.json') as f:
    data = json.load(f)

best_lines = []
for mask in data.keys():
    for i in range(len(data[mask])):
        slit = data[mask][i][0]
        best_lines.append(f'{mask}_{slit}')

params = {'shared_params': [], 'line_params': []}

done = False
for this_slit_path in slit_paths:
    for this_line_path in glob.glob(f'{this_slit_path}/*'):
        # Need to make sure doublet parameters are present
        if 'OIIa' not in this_line_path:
#            pass
            continue

        try:
            samples, data = load_samples(f'{this_line_path}/{run}')
            
            
            for p in data['paramnames']:
                category, name = p.split('-')
                if category == 'shared_params':
                    params['shared_params'].append(name)
                elif category.endswith('_params'):
                    line_name, param_name = category.split('_')
                    params['line_params'].append(f'{name}')
            done = True
            break

        except FileNotFoundError:
            print(f'File not found in {this_line_path}')
            continue
        
    if done is True:
        break


n_shared = len(params['shared_params'])
a = 1.718
b = 3.869

all_slit_names = []
all_estimates = []
all_estimates_sigma = []
all_lines = []
all_data_info = []
distance = []

for this_slit_path in slit_paths:
    slit_name = this_slit_path.split('/')[-1]
    
    if slit_name not in best_lines:
        continue
        
    line_names = []
    line_samples = []
    line_samples_not_shared = []
    line_samples_not_shared_idx = []

    line_data = []
    line_data_info = []

    shear = []
    sigma_shear = []

    for this_line_path in glob.glob(f'{this_slit_path}/*'):
        start_index = this_line_path.find("[") + 1
        end_index = this_line_path.find("]")
        line = this_line_path[start_index+1:end_index-1]
        
        if len(line.split(','))>1:
            continue
        

        try:
            samples, data = load_samples(f'{this_line_path}/{run}')
            data_info = joblib.load(f'{this_line_path}/{run}/data_info.pkl')

        except FileNotFoundError:
            print(f'File not found in {this_line_path}')
            continue
        
        MC_samples = getdist.MCSamples(samples=samples, names=data['paramnames'], sampler='nested')
        ## Add derived parameters
        p = MC_samples.getParams()
        theta = p.__dict__['shared_params-theta_int']
        
        if 'shared_params-g1' in list(p.__dict__.keys()):
            g1, g2 = p.__dict__['shared_params-g1'], p.__dict__['shared_params-g2']
            shear.append(MC_samples.getMeans()[:2])
            sigma_shear.append(MC_samples.getCov().diagonal()[:2]**0.5)

        elif 'shared_params-gamma_t' in list(p.__dict__.keys()):
            g1 = -p.__dict__['shared_params-gamma_t']*np.cos(2*data_info['galaxy']['beta'])
            g2 = -p.__dict__['shared_params-gamma_t']*np.sin(2*data_info['galaxy']['beta'])
            
            gt_mean = MC_samples.getMeans()[0]
            gt_mean_err = MC_samples.getCov().diagonal()[0]**0.5
            
            g1_mean = -gt_mean*np.cos(2*data_info['galaxy']['beta'])
            g2_mean = -gt_mean*np.sin(2*data_info['galaxy']['beta'])
            
            g1_mean_err = np.abs(gt_mean_err*np.cos(2*data_info['galaxy']['beta']))
            g2_mean_err = np.abs(gt_mean_err*np.sin(2*data_info['galaxy']['beta']))
            
            shear.append([g1_mean, g2_mean])
            sigma_shear.append([g1_mean_err, g2_mean_err])
        
        else:
            g1, g2 = 0., 0.

        g1_gal_frame = g1*np.cos(2*theta) - g2*np.sin(2*theta)
        g2_gal_frame = g1*np.sin(2*theta) + g2*np.cos(2*theta)


        if line in ['OIIa']:
            samples_not_shared = samples[:, n_shared:-1]

        else:
            # Since doublets have more parameters
            # we recast the samples for a singlet into a new array
            # with None columns for the additional parameters
            temp_names = []
            for p in data['paramnames']:
                category, name = p.split('-')
                if category == 'shared_params':
                    continue
                if category.endswith('_params'):
                    temp_names.append(f'{name}')

            samples_not_shared = np.zeros((samples.shape[0], len(params['line_params'])-1)) # exclude bkg
            samples_not_shared[:, :] = None

            # Map which column name
            idx = [params['line_params'].index(p) for p in temp_names[:-1]]

            samples_not_shared[:,idx] = samples[:, n_shared:-1]


        line_names.append(line)
        
        if np.all(g1!=0.) and np.all(g2!=0.):
            line_samples.append(np.column_stack((samples[:, :n_shared], g1_gal_frame, g2_gal_frame)))
        
        else:
            line_samples.append(samples[:, :n_shared])


        line_samples_not_shared.append(samples_not_shared)
        line_samples_not_shared_idx.append(idx)
        line_data.append(data)
        line_data_info.append(data_info)

    # Compute distance
    if len(line_names) == 0:
        continue
    slit_RA, slit_Dec = data_info['galaxy']['RA'], data_info['galaxy']['Dec']
    slit_coord = SkyCoord(slit_RA, slit_Dec, frame='fk5')
    dist = slit_coord.separation(abella2261_coord).to(u.arcmin)
    
    all_estimates.append(shear)
    all_estimates_sigma.append(sigma_shear)
    all_lines.append(line_names)
    all_data_info.append(data_info)
    distance.append(dist)
    
    all_slit_names.append(slit_name)
    
    #------------------- Get best fit -----------------------#
    fid_pars = None

    fig, axs = plt.subplots(2*len(line_names), 5, figsize=(24, 8*len(line_names)))
    
    for i, line in enumerate(line_names):
        if true_shear is True:
            RA, Dec = data_info['galaxy']['RA'].value, data_info['galaxy']['Dec'].value
            beta = data_info['galaxy']['beta']
            coord = galsim.CelestialCoord(RA*galsim.degrees, Dec*galsim.degrees )
            objpos = gamma1_map.wcs.toImage(coord)


            g1_predict = gamma1_map[objpos.round()]
            g2_predict = -gamma2_map[objpos.round()]
            fid_pars = {'shared_params':{'g1': g1_predict, 'g2': g2_predict}, 'line_params':{}}


        line_prof_path = f'/xdisk/timeifler/pranjalrs/DEIMOS_extracted_spec/data/spec_extract_{slit_name}.pkl'
        inference = UltranestSampler(line_data_info[i], params, fid_params=fid_pars, line_profile=line_prof_path)
        plot_residuals(inference, line_data[i], line_data_info[i], line, axs[2*i:2*(i+1), :], i)

    fig.tight_layout()
    plt.savefig(f'../figures/a2261/temp/{slit_name}_residual_{run}.pdf')
    plt.close()
                        
    # Plot shared parameter
    if save_triangle is True:
        params_shared = [n for n in data['paramnames'] if 'shared' in n]
        Mstar = data_info['galaxy']['Mstar']
        logvTF = (np.log10(Mstar)-a)/b
        sigmaTF = 10**logvTF*np.log(10)*0.058
        latex_names = FitParameters(params,  line_names).latex_names
        names = [f'${latex_names[n]}$' for n in params_shared]
        
        if 'shared_params-g1' in params_shared or 'shared_params-gammat' in params_shared:
            names += ['$\gamma_1^{\mathrm{gal}}$', '$\gamma_2^{\mathrm{gal}}$']
            
            paramRanges = [None]*(n_shared+2)
            priors = [None]*(n_shared+2)
        
        else:
            paramRanges = [None]*(n_shared)
            priors = [None]*(n_shared)
        
        paramRanges[params_shared.index('shared_params-vcirc')] = (10**logvTF-100, 10**logvTF+500)
        priors[params_shared.index('shared_params-vcirc')] = (10**logvTF, sigmaTF)

        GTC = pygtc.plotGTC(chains=line_samples, 
                            chainLabels=line_names,
                            priors=priors, paramRanges=paramRanges, legendMarker='All',
                            paramNames=names, filledPlots=False, customLabelFont=font, customLegendFont=font, 
                            customTickFont=font, figureSize=25)
        GTC.gca().set_title(slit_name)
        plt.savefig(f'../figures/a2261/temp/{slit_name}_contour_shared_{run}.pdf')
        plt.close()

    # Plot emission line parameters
    # Find parameter names

    if save_triangle is True:
        latex_names = FitParameters(params,  line_names).latex_names

        idx = 0
        if 'OIIa' in line_names:
            idx = line_names.index('OIIa')

        else:
            for i, s in enumerate(line_samples_not_shared):
                j = line_samples_not_shared_idx[i]
                line_samples_not_shared[i]  = line_samples_not_shared[i][:, j]
        
        params_not_shared_names = [n for n in line_data[idx]['paramnames'] if line_names[idx] in n]
        names = [f'${latex_names[n].split(" ")[0]}$' for n in params_not_shared_names]
    
        GTC = pygtc.plotGTC(chains=line_samples_not_shared, 
                            chainLabels=line_names, legendMarker='All',
                            paramNames=names[:-1], filledPlots=False,
                            customLabelFont=font, customLegendFont=font, customTickFont=font, figureSize=15)
        GTC.gca().set_title(slit_name)
        plt.savefig(f'../figures/a2261/temp/{slit_name}_contour_line_{run}.pdf')
        plt.close()

#------------------------- Make shear profile -------------------------#
cs = sns.color_palette("hls", n_colors=len(all_lines))
line_symbols = {'OIIa': 'o', 'Hb': 's', 'OIIIb': 'v', 'OIIIc': '^'}
# line_colors = {k:cs[i] for i, k in enumerate(line_symbols.keys())}

plt.figure(figsize=(12, 4))
for i in range(len(all_lines)):
    this_slit_lines = all_lines[i]
    this_data_info = all_data_info[i]['galaxy']
    beta = this_data_info['beta']

    for j, line in enumerate(this_slit_lines):
        g1, g2 = all_estimates[i][j]
        g1_err, g2_err = all_estimates_sigma[i][j]
    
        this_estimate = - g1*np.cos(2*beta) - g2*np.sin(2*beta)
        this_estimate_sigma = ((g1_err*np.cos(2*beta))**2 + (g2_err*np.sin(2*beta))**2)**0.5
    
        if this_estimate_sigma<0.15:
            plt.errorbar(distance[i].value+j*0.05, this_estimate, yerr=this_estimate_sigma, c=cs[i], marker=line_symbols[line], alpha=0.5, capsize=3)

    ymax = np.max(np.array(this_estimate)+np.array(this_estimate_sigma))
    plt.text(distance[i].value-0.3, ymax, all_slit_names[i], fontsize=8)

for i, line in enumerate(line_symbols.keys()):
    plt.errorbar([], [], xerr=0, yerr=[], c=cs[0], marker=line_symbols[line], alpha=0.5, capsize=3, label=line)

shear_obs = np.loadtxt('../data/shear_profile_Umetsu2014.txt', skiprows=1, delimiter=',')
plt.errorbar(shear_obs[:, 0], shear_obs[:, 1], xerr=shear_obs[:, 2], yerr=shear_obs[:, 3], fmt='s--', c='k', label='Umetsu+2014')

plt.axhline(0.3, c='gray', ls=':')
plt.axhline(-0.3, c='gray', ls=':')

plt.legend(fontsize=10, frameon=False)
    
plt.xlim([0.0, 10])
plt.ylim([-0.34, 0.34])

# plt.xscale('log')
plt.xlabel('R [arcmin]')
plt.ylabel('$\gamma_t$')

plt.savefig(f'../figures/a2261/shear_profile_{run}.pdf')


#------------------------- Compare with Gamma map -------------------------#

fig, ax = plt.subplots(2, 2, figsize=(15, 10))
for i in range(len(all_lines)):
    this_slit_lines = all_lines[i]
    this_estimate = all_estimates[i]
    this_estimate_sigma = all_estimates_sigma[i]
    
    this_data_info = all_data_info[i]['galaxy']
    ## Compute expected gamma_t
    RA, Dec = this_data_info['RA'].value, this_data_info['Dec'].value
    beta = this_data_info['beta']
    coord = galsim.CelestialCoord(RA*galsim.degrees, Dec*galsim.degrees )
    objpos = gamma1_map.wcs.toImage(coord)

    gamma1_predict, gamma1_predict_err = 100., 0.
    gamma2_predict, gamma2_predict_err = 100., 0.

    gamma1_ltm_predict, gamma1_ltm_predict_err = 100., 0.
    gamma2_ltm_predict, gamma2_ltm_predict_err = 100., 0.

    try:
        gamma1_predict = gamma1_map[objpos.round()]
        gamma1_predict_err = gamma1_err_map[objpos.round()]
        gamma2_predict = -gamma2_map[objpos.round()]
        gamma2_predict_err = gamma2_err_map[objpos.round()]
    
        gammat_predict = -gamma1_predict*np.cos(2*beta) - gamma2_predict*np.sin(2*beta)
        gammat_predict_err = ((gamma1_predict_err*np.cos(2*beta))**2 + (gamma2_predict_err*np.sin(2*beta))**2)**0.5

        # LTM
        gamma1_ltm_predict = gamma1_ltm_map[objpos.round()]
        gamma1_ltm_predict_err = gamma1_err_ltm_map[objpos.round()]
        gamma2_ltm_predict = -gamma2_ltm_map[objpos.round()]
        gamma2_ltm_predict_err = gamma2_err_ltm_map[objpos.round()]
    

        print(gammat_predict, all_slit_names[i])

    except:
        print(f'Object {all_slit_names[i]} not found')
        continue

    for j, line in enumerate(this_slit_lines):
        if gammat_predict<100:
            ax[0, 0].errorbar(gamma1_predict+j*0.008, this_estimate[j][0], xerr=gamma1_predict_err, yerr=this_estimate_sigma[j][0], c=cs[i], marker=line_symbols[line], alpha=0.5, capsize=3)
            ax[0, 0].text(gamma1_predict, gamma1_predict, all_slit_names[i], fontsize=8)
    
            ax[0, 1].errorbar(gamma2_predict+j*0.008, this_estimate[j][1], xerr=gamma2_predict_err, yerr=this_estimate_sigma[j][1], c=cs[i], marker=line_symbols[line], alpha=0.5, capsize=3)
            ax[0, 1].text(gamma2_predict, gamma2_predict, all_slit_names[i], fontsize=8)
            
            
            # Prediction from LTM
            ax[1, 0].errorbar(gamma1_ltm_predict+j*0.008, this_estimate[j][0], xerr=gamma1_ltm_predict_err, yerr=this_estimate_sigma[j][0], c=cs[i], marker=line_symbols[line], alpha=0.5, capsize=3)
            ax[1, 0].text(gamma1_predict, gamma1_predict, all_slit_names[i], fontsize=8)
    
            ax[1, 1].errorbar(gamma2_ltm_predict+j*0.008, this_estimate[j][1], xerr=gamma2_ltm_predict_err, yerr=this_estimate_sigma[j][1], c=cs[i], marker=line_symbols[line], alpha=0.5, capsize=3)
            ax[1, 1].text(gamma2_predict, gamma2_predict, all_slit_names[i], fontsize=8)
    
    ax[0, 1].set_title('Lensing map from NFW method')
    ax[1, 1].set_title('Lensing map from LTM method')

for i, line in enumerate(line_symbols.keys()):
    plt.errorbar([], [], xerr=0, yerr=[], c=cs[0], marker=line_symbols[line], alpha=0.5, capsize=3, label=line)

for subax in ax.flatten():
    subax.plot([-0.3, 0.3], [-0.3, 0.3], c='gray', ls=':')

    subax.axhline(0.3, c='gray', ls=':')
    subax.axhline(-0.3, c='gray', ls=':')


    
#     subax.set_xlim([-0.32, 0.32])
#     subax.set_ylim([-0.32, 0.32])

ax[0, 0].set(xlabel='$\gamma_1^{\mathrm{WL}}$', ylabel='$\gamma_1^{\mathrm{KL}}$')
ax[1, 0].set(xlabel='$\gamma_1^{\mathrm{WL}}$', ylabel='$\gamma_1^{\mathrm{KL}}$')

ax[0, 1].set(xlabel='$\gamma_2^{\mathrm{WL}}$', ylabel='$\gamma_2^{\mathrm{KL}}$')
ax[1, 1].set(xlabel='$\gamma_2^{\mathrm{WL}}$', ylabel='$\gamma_2^{\mathrm{KL}}$')

plt.savefig(f'../figures/a2261/shear_map_{run}.pdf')


## Save estimates
dtype = np.dtype([('id', 'S20'),
                  ('gammat', 'f8'), ('gammat_err', 'f8'), 
                  ('g1', 'f8'), ('g1_err', 'f8'), 
                  ('g2', 'f8'), ('g2_err', 'f8'),
                ('R', 'f8')])


est_gammat, est_g1, est_g2 = [], [], []
est_gammat_err, est_g1_err, est_g2_err = [], [], []

est_slits, est_lines = [], []
est_distance = []
for i in range(len(all_lines)):
    
    
    this_slit_lines = all_lines[i]
    this_data_info = all_data_info[i]['galaxy']
    beta = this_data_info['beta']

    for j, line in enumerate(this_slit_lines):
        g1, g2 = all_estimates[i][j]
        g1_err, g2_err = all_estimates_sigma[i][j]
    
        this_estimate = -g1*np.cos(2*beta) - g2*np.sin(2*beta)
        this_estimate_sigma = ((g1_err*np.cos(2*beta))**2 + (g2_err*np.sin(2*beta))**2)**0.5
    
        est_slits.append(all_slit_names[i])
        est_lines.append(line)

        est_gammat.append(this_estimate)
        est_gammat_err.append(this_estimate_sigma)
        est_g1.append(g1)
        est_g1_err.append(g1_err)
        est_g2.append(g2)
        est_g2_err.append(g2_err)
        
        est_distance.append(distance[i].value)

results = np.empty(len(est_slits), dtype=dtype)

results['id'] = [est_slits[i]+'_'+ est_lines[i] for i in range(len(est_slits))]
results['gammat'] = est_gammat
results['gammat_err'] = est_gammat_err
results['g1'] = est_g1
results['g1_err'] = est_g1_err
results['g2'] = est_g2
results['g2_err'] = est_g2_err
results['R'] = est_distance

np.save(f'stats_DEIMOS/results_{run}.npy', results)
