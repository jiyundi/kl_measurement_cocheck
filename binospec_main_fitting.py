import joblib
import pickle
# import yaml
import getdist
import numpy as np
import galsim
from   klm.ultranest_sampler import UltranestSampler
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
matplotlib.use('TkAgg')  # 非 GUI 模式
# matplotlib.use('module://matplotlib_inline.backend_inline')
plt.style.use('default')



mock_folder = 'mock_100/'

for iter_num in [67, 65, 63, 61]:
    slit_name = '095'
    run       = iter_num
    
    # Load
    with open(f'{mock_folder}pkl/mock_{slit_name}_{iter_num}.pkl', "rb") as f:
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
            'line_species': ['OII','OII','OII'], 
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
                    # 'g2': 0.2, 
                    # 'cosi': 0.5, 
                    # 'theta_int': 0.0,
                    # 'vscale': 0.2, 
                    # 'r_hl_disk':  1,
                    # 'r_hl_bulge': 0.8, 
                    
                    # 'gamma_t': 0.2, 
                    
                    # 'vcirc': 142,
                    
                    'dx_disk':  0, 
                    'dy_disk':  0, 
                    'dx_bulge': 0, 
                    'dy_bulge': 0,
                    # 'flux':       3.0, 
                    # 'flux_bulge': 2.0,
                    },
                'OII_params': {
                    'dx_vel':   0, 
                    'dx_vel_2': 0, 
                    # 'I01': 10, 
                    # 'I02': 10, 
                    # 'bkg_level': 1,
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
                'g2': {
                    'prior': {'min': -0.5, 'max': 0.5}, 
                    'latex_name': r'$g_2$'
                    },
                'vcirc': {
                    'prior': {'min': 119, # -3-sigma away from 142
                              'max': 170, # +3-sigma away from 142
                              'norm':
                                  {'loc': 'TFprior.log10_vTF',
                                   'scale': 'TFprior.sigmaTF'}}, 
                    'latex_name': r'$v_{\rm circ}$'
                    },
                # 'gamma_t': {
                #     'prior': {'min': -0.5, 'max': 0.5}, 
                #     'latex_name': 'g_+'
                #     },
                'cosi': {
                    'prior': {'min': 0, 'max': 1},
                    'latex_name': r'$\cos{(i)}$'
                    }, 
                'theta_int': {
                    'prior': {'min': -1*np.pi, 'max': 1*np.pi}, 
                    'latex_name': r'$\theta_{\mathrm{int}}$'
                    }, 
                'vscale': {
                    'prior': {'min': 0.01, 'max': 1}, 
                    'latex_name': r'$r_{\mathrm{vscale}}$'
                    }, 
                'r_hl_disk': {
                    'prior': {'min': 0.15, 'max': 3}, 
                    'latex_name': r'$r_{\mathrm{hl, disk}}$'
                    },
                'r_hl_bulge': {
                    'prior': {'min': 0.15, 'max': 3}, 
                    'latex_name': r'$r_{\mathrm{hl, bulge}}$'
                    },
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
                'flux': {
                    'prior': {'min': 1, 'max': 5}, 
                    'latex_name': r'$\log{(F_{\rm disk})}$'
                    },
                'flux_bulge': {
                    'prior': {'min': 1, 'max': 5}, 
                    'latex_name': r'$\log{(F_{\rm bulge})}$'
                    }, 
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
                'I01': {
                    'prior': {'min': 0, 'max': 1000}, 
                    'latex_name': r'$I_{\rm 0, Line 1}$'
                    }, 
                'I02': {
                    'prior': {'min': 0, 'max': 1000}, 
                    'latex_name': r'$I_{\rm 0, Line 2}$'
                    }, 
                'bkg_level': {
                    'prior': {'min': 0, 'max': 10}, 
                    'latex_name': 'BKG'
                    }, 
                }
            }
        }
    
    #
    inference = UltranestSampler(data_info, config_dic)
    # rough_check_gamma_convergence(data_info)
    
    save_path = './'
    
    # For Pranjal's co-check
    # with open( f'{save_path}'+f'run0.{run}/config.yaml', 'w') as file:
    #     yaml.dump(inference.config.__repr__, file)
    
    ## Output path
    #---------------------------  Run Ultranest --------------------------- #
    """ To see each iter/eval's param set in fitting, debug by setting 
        a breakpoint on Line 2628 in site-packages/ultranest/integrator.py
        or Line 120 in ./ultranest_sampler.py 
    """
    sampler = inference.run(output_dir=f'./run0.{run}/', test_run=False, run_num=run)
    
    # np.save(f'{save_path}/run0.{run}/ultranest_sampler_results.npy', sampler.results)
    # If need to read...
    # samples = np.load('./samples.npy')
    
    with open(f'{save_path}/run0.{run}/ultranest_sampler_results.pkl', 'wb') as f:
        pickle.dump(sampler.results, f)
    # If need to read...
    # with open(save_path, 'rb') as f:
    #     loaded_results = pickle.load(f) # read
    # samples = loaded_results['samples'] # access saved samples
    
    #---------------------------  Load samples --------------------------- #
    sampler.print_results()
    sampler.plot()
    sampler.plot_trace()
    
    samples = sampler.results['samples']
    
    MC_samples = getdist.MCSamples(samples=samples, 
                                   names=inference.config.params.names, 
                                   sampler='nested')
    
    #---------------------------  Best fit results ----------------------- #
    estimates = sampler.results['maximum_likelihood']['point']
    
    best_fit_dict = inference.params.gen_param_dict(inference.config.params.names, 
                                                    estimates)
    
    line = config_dic['galaxy_params']['line_species']
    this_line_dict = {**best_fit_dict['shared_params'], 
                      **best_fit_dict[f'{line[0]}_params']}
    
    best_fit_params = {'best_fit_dict': best_fit_dict,
                       'this_line_dict': this_line_dict}
    with open(f'{save_path}/run0.{run}/ultranest_best_fit_params.pkl', "wb") as f:
        joblib.dump(best_fit_params, f)
    # with open('ultranest_best_fit_params.pkl', "rb") as f:
    #     best_fit_params = joblib.load(f)
    
    #---------------------------  Plot --------------------------- #
    def plot_obs_fit_res(data_info, 
                         inference, best_fit_dict, estimates, this_line_dict,
                         run, save_path):
        nspec = len(data_info['spec'])
        image_fit  = inference.image_model.get_image(best_fit_dict['shared_params']).T
        image_obs  = inference.data_image.T
        image_chi2 = inference.calc_image_loglike(best_fit_dict)
        image_dof  = image_fit.shape[0] * image_fit.shape[1] - len(estimates)
        
        fig   = plt.figure(figsize=(12, 3*(1+nspec)))  # (length, height)
        plt.subplots_adjust(hspace=0.2, wspace=0.2) # h=height
        gs    = fig.add_gridspec(nrows=1+nspec, ncols=3, 
                                 height_ratios=[1]*(1+nspec), 
                                 width_ratios =[1,1,1])
        ax_img_obs = fig.add_subplot(gs[0, 0])
        ax_img_fit = fig.add_subplot(gs[0, 1])
        ax_img_res = fig.add_subplot(gs[0, 2])
        
        noise = np.std(image_obs)
        im1 = ax_img_obs.imshow(image_obs, 
                                vmin=0-noise, vmax=0 + 5*noise, 
                                cmap='viridis', origin='lower', aspect='equal')
        im2 = ax_img_fit.imshow(image_fit, 
                                vmin=0-noise, vmax=0 + 5*noise, 
                                cmap='viridis', origin='lower', aspect='equal')
        im3 = ax_img_res.imshow(image_obs - image_fit, 
                                cmap='viridis', origin='lower', aspect='equal')
        fig.colorbar(im1, ax=ax_img_obs)
        fig.colorbar(im2, ax=ax_img_fit)
        fig.colorbar(im3, ax=ax_img_res)
        
        strk_txt2 = ax_img_res.text(1, 1, 
                                    r"$\chi^2/{\rm DOF}=$"+f"{image_chi2/image_dof:.1f}", 
                                    c='yellow', fontsize=15, weight='bold', 
                                    ha='right', va='top', transform=ax_img_res.transAxes)
        strk_txt2.set_path_effects([path_effects.Stroke(linewidth=4, foreground='blue'),  # stroke
                                    path_effects.Normal()]) # stroked-text
        
        ax_img_obs.set_ylabel(r'$\Delta$ Dec [arcsec]')
        ax_img_obs.set_title('Observation')
        ax_img_fit.set_title('Best Fit')
        ax_img_res.set_title('Residual (obs - fit)')
        ax_img_obs.grid(linestyle=':', color='white', alpha=0.5)
        ax_img_fit.grid(linestyle=':', color='white', alpha=0.5)
        ax_img_res.grid(linestyle=':', color='white', alpha=0.5)
        
        for i in range(nspec):
            inference.spec_model[i]._init_observable(data_info['galaxy'], 
                                                     data_info['spec'][i]['par_meta'])
            spec0_fit = inference.spec_model[i].get_observable(this_line_dict)
            spec0_obs = inference.data_spec[i]
            svar0_obs = inference.var_spec[i]
            cont0_obs = inference.cont_model[i][:, np.newaxis]
            spec0_chi2 = np.sum((spec0_obs - spec0_fit)**2 / (spec0_obs + svar0_obs + cont0_obs))
            spec0_dof  = spec0_fit.shape[0] * spec0_fit.shape[1] - len(estimates)
        
            # spec0_obs[stdv0_obs>25] = 0.
            # spec0_obs[stdv0_obs>25] = 0.
            
            extent = [data_info['spec'][i]['par_meta']['lambda_grid'][0][ 0].value, 
                      data_info['spec'][i]['par_meta']['lambda_grid'][0][-1].value, 
                      inference.spec_model[i].slit_x[0], 
                      inference.spec_model[i].slit_x[-1]]
        
            ax_spe_obs = fig.add_subplot(gs[i+1, 0])
            ax_spe_fit = fig.add_subplot(gs[i+1, 1])
            ax_spe_res = fig.add_subplot(gs[i+1, 2])
            
            noise = np.std(spec0_obs)
            im1 = ax_spe_obs.imshow(spec0_obs, 
                                    vmin=0-noise, vmax=0 + 5*noise, 
                                    origin='lower', extent=extent, 
                                    cmap='viridis', aspect='auto')
            im2 = ax_spe_fit.imshow(spec0_fit, 
                                    vmin=0-noise, vmax=0 + 5*noise, 
                                    origin='lower', extent=extent, 
                                    cmap='viridis', aspect='auto')
            im3 = ax_spe_res.imshow(spec0_fit - spec0_obs, 
                                    origin='lower', extent=extent, 
                                    cmap='viridis', aspect='auto')
            fig.colorbar(im1, ax=ax_spe_obs)
            fig.colorbar(im2, ax=ax_spe_fit)
            fig.colorbar(im3, ax=ax_spe_res)
            
            strk_txt1 = ax_spe_res.text(1, 1, 
                                        r"$\chi^2/{\rm DOF}=$"+f"{spec0_chi2/spec0_dof:.1f}", 
                                        c='yellow', fontsize=15, weight='bold', 
                                        ha='right', va='top', 
                                        transform=ax_spe_res.transAxes)
            strk_txt1.set_path_effects([path_effects.Stroke(linewidth=4, foreground='blue'),  # stroke
                                        path_effects.Normal()]) # stroked-text
            
            ax_spe_obs.xaxis.get_major_formatter().set_useOffset(False)
            ax_spe_fit.xaxis.get_major_formatter().set_useOffset(False)
            ax_spe_res.xaxis.get_major_formatter().set_useOffset(False)
            ax_spe_obs.set_ylabel('Slit Position [arcsec]')
            ax_spe_obs.grid(linestyle=':', color='white', alpha=0.5)
            ax_spe_fit.grid(linestyle=':', color='white', alpha=0.5)
            ax_spe_res.grid(linestyle=':', color='white', alpha=0.5)
        
        plt.savefig(f'{save_path}/run0.{run}/best_fit_spec.png', dpi=100, bbox_inches='tight')
        return
    plot_obs_fit_res(data_info, 
                     inference, best_fit_dict, estimates, this_line_dict,
                     run, save_path)



# Useful functions
def rough_check_gamma_convergence(data_info):
    """ 
    To use, only leave one free param in fitting param dictionary.

    You need to adjust residual dict params to true values.
    """
    data_info1 = data_info.copy()
    data_info2 = data_info.copy()
    data_info3 = data_info.copy()
    data_info1['spec'] = []
    data_info1['spec'].append(data_info['spec'][0])
    data_info2['spec'] = []
    data_info2['spec'].append(data_info['spec'][1])
    data_info3['spec'] = []
    data_info3['spec'].append(data_info['spec'][2])
    inference1 = UltranestSampler(data_info1, config_dic)
    inference2 = UltranestSampler(data_info2, config_dic)
    inference3 = UltranestSampler(data_info3, config_dic)
    
    param_grid = np.linspace(-0.5, 0.5, 20) # gamma_t
    log_like, log_like1, log_like2, log_like3   = [], [], [], []
    for i in range(len(param_grid)):
        # print(param_grid[i])
        likeli = inference.calc_joint_loglike([param_grid[i]])
        log_like.append(likeli)
        # print(param_grid[i])
        likeli1 = inference1.calc_joint_loglike([param_grid[i]])
        log_like1.append(likeli1)
        # print(param_grid[i])
        likeli2 = inference2.calc_joint_loglike([param_grid[i]])
        log_like2.append(likeli2)
        # print(param_grid[i])
        likeli3 = inference3.calc_joint_loglike([param_grid[i]])
        log_like3.append(likeli3)
    
    plt.semilogy(param_grid, -np.array(log_like), label='combined')
    plt.semilogy(param_grid, -np.array(log_like1), ls='-.', label='1')
    plt.semilogy(param_grid, -np.array(log_like2), ls='--', label='2')
    plt.semilogy(param_grid, -np.array(log_like3), ls=':', label='3')
    plt.axvline(x=0)
    plt.legend()
    plt.grid()
    plt.show()
    return

def overwrite_best_fit_params_in_dic(inference):
    # See parameters.py:128 for details
    bestfitdic = inference.params.gen_param_dict(inference.config.params.names, 
                                                 [0.6])
    return bestfitdic

