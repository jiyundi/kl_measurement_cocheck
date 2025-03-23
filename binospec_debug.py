import joblib
import pickle
import yaml
import getdist
import numpy as np
import galsim
from klm.ultranest_sampler import UltranestSampler
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
# matplotlib.use('TkAgg')  # 非 GUI 模式
# matplotlib.use('module://matplotlib_inline.backend_inline')
plt.style.use('default')

# Useful functions
def rough_check_gamma_convergence(inference, inference1, inference2, inference3):
    param_grid = np.linspace(-0.5, 0.5, 400) # gamma_t
    log_like, log_like1, log_like2, log_like3   = [], [], [], []
    for i in range(len(param_grid)):
        likeli = inference.calc_joint_loglike([param_grid[i]])
        log_like.append(likeli)
        
        likeli1 = inference1.calc_joint_loglike([param_grid[i]])
        log_like1.append(likeli1)
        
        likeli2 = inference2.calc_joint_loglike([param_grid[i]])
        log_like2.append(likeli2)
        
        likeli3 = inference3.calc_joint_loglike([param_grid[i]])
        log_like3.append(likeli3)
    
    fig = plt.figure()  # (length, height)
    gs = fig.add_gridspec(nrows=2, ncols=1)
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    ax1.plot(param_grid, -np.array(log_like), lw=5, label='combined: '+f'{-np.mean(log_like):.2g}')
    ax2.plot(param_grid, -np.array(log_like), lw=5, label='combined: '+f'{-np.mean(log_like):.2g}')
    ax2.plot(param_grid, -np.array(log_like1), ls='-.', label='1: '+f'{-np.mean(log_like1):.2g}')
    ax2.plot(param_grid, -np.array(log_like2), ls='--', label='2: '+f'{-np.mean(log_like2):.2g}')
    ax2.plot(param_grid, -np.array(log_like3), ls=':',  label='3: '+f'{-np.mean(log_like3):.2g}')
    ax2.axvline(x=0, color='darkred')
    ax2.set_xlabel('g_1')
    ax2.set_ylabel('Likelihood')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid()
    plt.show()
    return

def overwrite_best_fit_params_in_dic(inference):
    # See parameters.py:128 for details
    bestfitdic = inference.params.gen_param_dict(inference.config.params.names, 
                                                 [0.6])
    return bestfitdic


for iter_num in [41]:
    slit_name = '095'
    run       = iter_num
    
    # Load
    with open(f'mock_100/pkl/mock_{slit_name}_{iter_num}.pkl', "rb") as f:
        data_info = joblib.load(f)
        
        # Recover wcs(galsim.wcs) from ap_wcs
        ap_wcs  = data_info['image']['par_meta']['ap_wcs']
        data_info['image']['par_meta']['wcs'] = galsim.AstropyWCS(wcs=ap_wcs)
    
    # If needed, check mock params in this file
    mock_params = data_info['par_fit']
    
    # Load YAML file for config
    config_dic = {
        'galaxy_params': {
            'obs_type': 'slit', 
            'line_species': ['OII']*len(data_info['spec']), # same len of spec
            'log10_Mstar': np.log10(142) * 3.869 + 1.718, 
            # None, by log10(vcirc) = (log10_Mstar - 1.718) / 3.869
            'log10_Mstar_err': 0.1, # None, 
            'line_profile_path': None,
            }, 
        'likelihood': {
            'fit_image':  True, 
            'fit_spec':   True, 
            'set_non_analytic_prior': None,
            'fid_params': 
                # Comment out below 
                # if you'd fit these params!
                {
                'shared_params': {
                    'image_snr': data_info['par_fit']['shared_params-image_snr'], 
                    'spec_snr':  data_info['par_fit']['shared_params-spec_snr'], 
                    # 'g1': 0.0, 
                    'g2': 0.2, 
                    'cosi': 0.125, 
                    'theta_int': 0.0,
                    'vscale': 0.2, 
                    'r_hl_disk':  1,
                    'r_hl_bulge': 0.8, 
                    
                    # 'gamma_t': 0.2, 
                    
                    'vcirc': 142,
                    
                    'dx_disk':  0, 
                    'dy_disk':  0, 
                    'dx_bulge': 0, 
                    'dy_bulge': 0,
                    'flux':       3.0, 
                    'flux_bulge': 2.0,
                    },
                'OII_params': {
                    'dx_vel':   0, 
                    'dx_vel_2': 0, 
                    'I01': 10, 
                    'I02': 10, 
                    'bkg_level': 1,
                    },
                },
            }, 
        'TFprior': {
            'use_TFprior': True, 
            'log10_vTF':   None, # np.log10(142), 
            'sigmaTF':     None, 
            'a':            None, 
            'b':            None, 
            'sigmaTF_intr': None, 
            'relation':     None
            }, 
        'params': {
            'shared_params': {
                'g1': {
                    'prior': {'min': -0.5, 'max': 0.5}, 
                    'latex_name': r'$g_1$'
                    },
                # 'g2': {
                #     'prior': {'min': -0.5, 'max': 0.5}, 
                #     'latex_name': r'$g_2$'
                #     },
                # 'vcirc': {
                #     'prior': {'min': 119, # -3-sigma away from 142
                #               'max': 170, # +3-sigma away from 142
                #               'norm':
                #                   {'loc': 'TFprior.log10_vTF',
                #                    'scale': 'TFprior.sigmaTF'}}, 
                #     'latex_name': r'$v_{\rm circ}$'
                #     },
                # 'gamma_t': {
                #     'prior': {'min': -0.5, 'max': 0.5}, 
                #     'latex_name': 'g_+'
                #     },
                # 'cosi': {
                #     'prior': {'min': 0, 'max': 1},
                #     'latex_name': r'$\cos{(i)}$'
                #     }, 
                # 'theta_int': {
                #     'prior': {'min': -1*np.pi, 'max': 1*np.pi}, 
                #     'latex_name': r'$\theta_{\mathrm{int}}$'
                #     }, 
                # 'vscale': {
                #     'prior': {'min': 0.01, 'max': 1}, 
                #     'latex_name': r'$r_{\mathrm{vscale}}$'
                #     }, 
                # 'r_hl_disk': {
                #     'prior': {'min': 0.15, 'max': 3}, 
                #     'latex_name': r'$r_{\mathrm{hl, disk}}$'
                #     },
                # 'r_hl_bulge': {
                #     'prior': {'min': 0.15, 'max': 3}, 
                #     'latex_name': r'$r_{\mathrm{hl, bulge}}$'
                #     },
                # 'dx_disk': {
                #     'prior': {'min': -1, 'max': 1}, 
                #     'latex_name': '\\Delta_{\\rm x_disk}'
                #     },
                # 'dy_disk': {
                #     'prior': {'min': -1, 'max': 1}, 
                #     'latex_name': '\\Delta_{\\rm y_disk}'
                #     },
                # 'dx_bulge': {
                #     'prior': {'min': -1, 'max': 1}, 
                #     'latex_name': '\\Delta_{\\rm x_bulge}'
                #     },
                # 'dy_bulge': {
                #     'prior': {'min': -1, 'max': 1}, 
                #     'latex_name': '\\Delta_{\\rm y_bulge}'
                #     },
                # 'flux': {
                #     'prior': {'min': 1, 'max': 5}, 
                #     'latex_name': r'$\log{(F_{\rm disk})}$'
                #     },
                # 'flux_bulge': {
                #     'prior': {'min': 1, 'max': 5}, 
                #     'latex_name': r'$\log{(F_{\rm bulge})}$'
                #     }, 
                },
            'line_params': {
                # 'dx_vel': {
                #     'prior': {'min': -0.3, 'max': 0.3}, 
                #     'latex_name': '{\\Delta x^{\\mathrm{vel, 1}}}'
                #     }, 
                # 'dx_vel_2': {
                #     'prior': {'min': -0.3, 'max': 0.3}, 
                #     'latex_name': '{\\Delta x^{\\mathrm{vel, 2}}}'
                #     }, 
                # 'I01': {
                #     'prior': {'min': 0, 'max': 1000}, 
                #     'latex_name': r'$I_{\rm 0, Line 1}$'
                #     }, 
                # 'I02': {
                #     'prior': {'min': 0, 'max': 1000}, 
                #     'latex_name': r'$I_{\rm 0, Line 2}$'
                #     }, 
                # 'bkg_level': {
                #     'prior': {'min': 0, 'max': 10}, 
                #     'latex_name': 'BKG'
                #     }, 
                }
            }
        }
    
    #
    
    data_info1 = data_info.copy()
    data_info2 = data_info.copy()
    data_info3 = data_info.copy()
    data_info1['spec'] = []
    data_info1['spec'].append(data_info['spec'][0])
    data_info2['spec'] = []
    data_info2['spec'].append(data_info['spec'][1])
    data_info3['spec'] = []
    data_info3['spec'].append(data_info['spec'][2])
    inference  = UltranestSampler(data_info,  config_dic)
    inference1 = UltranestSampler(data_info1, config_dic)
    inference2 = UltranestSampler(data_info2, config_dic)
    inference3 = UltranestSampler(data_info3, config_dic)
    rough_check_gamma_convergence(inference, inference1, inference2, inference3)
 
    save_path = './'
    # rough_check_gamma_convergence(inference)
