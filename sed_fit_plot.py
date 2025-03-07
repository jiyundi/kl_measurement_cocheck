import argparse
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

import prospect.io.read_results as reader
import prospect.models

import sys
sys.path.append('../scripts/')

from sed_fit import build_all

parser = argparse.ArgumentParser()
parser.add_argument('--sfh_type')
args = parser.parse_args()

def plot_posterior(file_path, fig, axis, add_label=True):
    result, obs, _ = reader.results_from(file_path, dangerous=False)
    run_params = result['run_params']

    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=axis, height_ratios=[4, 1.5], wspace=0.1, hspace=0.1)

    ax = plt.Subplot(fig, inner[0])
    pwave = np.array([f.wave_effective for f in obs["filters"]])
    # plot the data
    ax.plot(pwave, obs["maggies"], linestyle="", marker="o", color="k")
    ax.errorbar(pwave,  obs["maggies"], obs["maggies_unc"], linestyle="", color="k", zorder=10)
    ax.set_ylabel(r"$f_\nu$ (maggies)")
    ax.set_xlim(3e3, 1e4)
    ax.set_ylim(obs["maggies"].min() * 0.1, obs["maggies"].max() * 5)
    ax.set_yscale("log")
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    fig.add_subplot(ax)

    # get the best-fit SED
    bsed = result["bestfit"]
    ax.plot(bsed["restframe_wavelengths"] * (1+0.623), bsed["spectrum"], color="firebrick", label="MAP sample")
    ax.plot(pwave, bsed["photometry"], linestyle="", marker="s", markersize=10, mec="orange", mew=3, mfc="none")
    
    dust_label = dust_dict[file_path.split('_')[-2][-1]]
    imf_label = imf_dict[file_path.split('_')[-3][-1]]

    title = f'IMF: {imf_label}, Dust model: {dust_label}'
    chi = (obs["maggies"] - bsed["photometry"])**2 / obs["maggies_unc"]**2/2
    ax.text(3100, 3e-9, f'Total $\chi^2$={np.sum(chi):.1f}')
    ax.set_title(title)

    ax = plt.Subplot(fig, inner[1], sharex=ax)
    ax.plot(pwave, chi, linestyle="", marker="o", color="k")
    ax.axhline(0, color="k", linestyle=":")
    ax.set_ylim(-1, 40)
    ax.set_xlabel(r"$\lambda\ (\AA)$")
    ax.set_ylabel(r"$\chi^2_{\rm best}$")

    if add_label is False:
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    fig.add_subplot(ax)
    
    print(file_path.split('/')[-1])
    #print('Best fit magnitudes:', -2.5*np.log10(bsed["photometry"]))
    if 'parametric' in path:
        idx = result['theta_labels'].index('mass')
        print('Mass: ', np.log10(bsed['parameter'][idx]))
    
    else:
        idx = result['theta_labels'].index('total_mass')
        print('Mass: ', np.log10(bsed['parameter'][idx]*bsed['mfrac']))

mask = 'a2261b'
slit = '007'

catalog = 'HST_plus_Subaru'
sfh_type = args.sfh_type


base_path = f'/xdisk/timeifler/pranjalrs/sed_fit_{catalog}'

# hfile = f"{base_path}/{mask}_{slit}_imf{imf_type}_dust{dust_type}_{sampler}.h5"
# hfile = '../../../xdisk/sed_fit/a2261b_007_imf1_dust2_emcee_23Sep02-13.14.h5'
all_paths = {}         
all_paths['parametric'] = [
         f'{base_path}/a2261b_007_parametric_imf1_dust1_emcee.h5',
         f'{base_path}/a2261b_007_parametric_imf2_dust1_emcee.h5',
         f'{base_path}/a2261b_007_parametric_imf1_dust2_emcee.h5',
         f'{base_path}/a2261b_007_parametric_imf2_dust2_emcee.h5',
         f'{base_path}/a2261b_007_parametric_imf1_dust4_emcee.h5',
         f'{base_path}/a2261b_007_parametric_imf2_dust4_emcee.h5',]

all_paths['dirichlet'] = [f.replace('_parametric_','_') for f in all_paths['parametric']]

# title = 
imf_dict = {'1': 'Chabrier 2003',
            '2': 'Kroupa 2001'}

dust_dict = {'0': 'Power law', 
             '1': 'Milky Way',
             '2': 'Calzetti 2000',
             '4': 'Kriek & Conroy 2013'}

paths = all_paths[sfh_type]

fig = plt.figure(figsize=(12, 8))
outer = gridspec.GridSpec(3, 2, wspace=0.2, hspace=0.5)

for i, (subax, path) in enumerate(zip(outer, paths)):
    if i in [0, 1, 2, 3, 4, 5]:
        plot_posterior(path, fig, subax, add_label=False)

    else:
        plot_posterior(path, fig, subax)

fig.suptitle(f'SFH: {sfh_type}')
plt.savefig(f'../{catalog}_{sfh_type}_SED_007b.png', bbox_inches='tight', dpi=300)


import getdist
from getdist import plots


labels = ['\log M', '\log Z/Z_\odot', 'dust', 't_{\mathrm{age}}', '\log \\tau']


MC_samples = []
for path in paths:
    result, obs, _ = reader.results_from(path, dangerous=False)
    bsed = result["bestfit"]

    shape = result['chain'].shape
    chain = result['chain'].reshape(shape[0]*shape[1], shape[2])
    new_chain = np.zeros((chain.shape[0], 3))
    if 'parametric' in path:
        idx = result['theta_labels'].index('mass')
        new_chain[:, 0] = np.log10(chain[:, idx])
    

    else:
        idx = result['theta_labels'].index('total_mass')
        new_chain[:, 0] = np.log10(chain[:, idx]*bsed['mfrac'])
    
    idx = result['theta_labels'].index('dust2')
    new_chain[:, 1] = chain[:, idx]
    
    idx = result['theta_labels'].index('logzsol')
    new_chain[:, 2] = chain[:, idx]

    labels = ['Stellar Mass', 'Dust2', 'log Z/Z_\odot']

    dust_label = dust_dict[path.split('_')[-2][-1]]
    imf_label = imf_dict[path.split('_')[-3][-1]]

    title = f'IMF: {imf_label}, Dust model: {dust_label}'
    this_sample = getdist.MCSamples(samples=new_chain, names=labels, labels=labels, label=title)
    MC_samples.append(this_sample)
    path = path.replace('imf1', 'imf2')

g = plots.get_subplot_plotter()
plt.figure(facecolor='white')
g.triangle_plot(MC_samples)
plt.savefig(f'../{catalog}_{sfh_type}_triangle_plot.png', bbox_inches='tight', dpi=300)

#for path in paths[::2]:
#    MC_samples = []
#    for i in range(2):
#        result, obs, _ = reader.results_from(path, dangerous=False)
#        shape = result['chain'].shape
#        chain = result['chain'].reshape(shape[0]*shape[1], shape[2])
#        chain[:, 0] = np.log10(chain[:, 0])
    #    chain[:, -1] = np.log10(chain[:, -1])
#        labels = result['theta_labels']    

#        dust_label = dust_dict[path.split('_')[-2][-1]]
#        imf_label = imf_dict[path.split('_')[-3][-1]]
#
#        title = f'IMF: {imf_label}, Dust model: {dust_label}'
#        this_sample = getdist.MCSamples(samples=chain, names=result['theta_labels'], labels=labels, label=title)
#        MC_samples.append(this_sample)
#        path = path.replace('imf1', 'imf2')

#    g = plots.get_subplot_plotter()
#    plt.figure(facecolor='white')
#    g.triangle_plot(MC_samples)
#    plt.savefig(f'../{catalog}_{sfh_type}_triangle_plot_dust{path.split("_")[-2][-1]}', bbox_inches='tight', dpi=300)
