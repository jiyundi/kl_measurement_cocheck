from safe_plot import setup; setup() # must before plt

import yaml
import joblib
import pickle
import numpy as np
# import astropy.units as u
import galsim
from   klm.ultranest_sampler import UltranestSampler
from   klm.parameters        import Parameters
# from   build_mock      import make_a_mock_only_one_slit
# from   build_mock_plot import make_exam_plots
from   binospec_plot_best_fit_only import load_best_fit_json
from   binospec_plot_best_fit_only import plot_obs_fit_res

# import matplotlib
import matplotlib.pyplot as plt
# import matplotlib.patheffects as path_effects
# matplotlib.use('TkAgg')
# matplotlib.use('module://matplotlib_inline.backend_inline')
plt.style.use('default')
# plt.style.use('classic')
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.serif": "Helvetica",
})

def load_mock(existed_mock_filepath='mock/a.pkl'):
    with open(existed_mock_filepath, "rb") as f:
        data_info = joblib.load(f)
        
        # Recover wcs(galsim.wcs) from ap_wcs
        ap_wcs  = data_info['image']['par_meta']['ap_wcs']
        data_info['image']['par_meta']['wcs'] = galsim.AstropyWCS(wcs=ap_wcs)
    return data_info

def copy_to_cache(file_path, dest_path, cache_dir='./cache'):
    import os
    import shutil
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"Cache folder created: {cache_dir}")

    shutil.copy2(file_path, dest_path)
    print(f"Cache folder created: {dest_path}")
    return

# =================================================================
# =================================================================
# =================================================================
# Useful functions ================================================
# =================================================================
# =================================================================
# =================================================================

def overwrite_best_fit_params_in_dic(inference):
    # See parameters.py:128 for details
    bestfitdic = inference.params.gen_param_dict(inference.config.params.names, 
                                                 [0.6])
    return bestfitdic

# =================================================================
# =================================================================
# =================================================================
# Commands ========================================================
# =================================================================
# =================================================================
# =================================================================
















run = 1004

# Load
data_info = load_mock(existed_mock_filepath='mock_pranjal/mock_data_10.pkl')

# Load YAML file for config
# with open("../config/binospec_fid_params.yaml", "r", encoding="utf-8") as file1:
#     fid_params     = yaml.safe_load(file1)
# with open("../config/binospec_fitting_params.yaml", "r", encoding="utf-8") as file2:
#     fitting_params = yaml.safe_load(file2)

# Make cache fiducial+fitting files for future
# copy_to_cache(file_path= '../config/binospec_fid_params.yaml', 
#               cache_dir= '../config/fid_cache', 
#               dest_path=f'../config/fid_cache/binospec_fid_params_{run}.yaml')
# copy_to_cache(file_path= '../config/binospec_fitting_params.yaml', 
#               cache_dir= '../config/fitting_cache',
#               dest_path=f'../config/fitting_cache/binospec_fitting_params_{run}.yaml')


# config_dic = {
#     'galaxy_params': {
#         'obs_type': 'slit', 
#         'line_species': ['Halpha', 'Halpha'], 
#         'log10_Mstar': 10.5, # np.log10(142) * 3.869 + 1.718, 
#         # None, by log10(vcirc) = (log10_Mstar - 1.718) / 3.869
#         'log10_Mstar_err': 0.1, # None, 
#         'line_profile_path': None,
#         }, 
#     'likelihood': {
#         'fit_image':  True, 
#         'fit_spec':   True, 
#         'set_non_analytic_prior': None,
#         'fid_params': fid_params
#         }, 
#     'TFprior': {
#         'use_TFprior': True, 
#         'log10_vTF':   None, # np.log10(142), 
#         'sigmaTF':     None, 
#         'a':            None, 
#         'b':            None, 
#         'sigmaTF_intr': None, 
#         'relation':     None
#         }, 
#     'params': fitting_params,
#     'truevalues': None
#     }

# If needed, check true values in mock params YAML file
# mock_params = data_info['fid_params']
# mock_params = Parameters._flatten(mock_params, level=1)
# true_params = np.zeros((1,3))
# for shared_or_lines, subdict in fitting_params.items():
#     if subdict:
#         for this_par_name, this_par_dict in subdict.items():
#             for true_name, true_value in mock_params.items():
#                 if this_par_name == true_name.split('-')[1]:
#                     true_params = np.append(true_params,
#                                             [[true_name, 
#                                               this_par_name, 
#                                               true_value]], 
#                                             axis=0)
# true_params = np.delete(true_params, (0), axis=0) 
# config_dic['truevalues'] = true_params[:, 2].astype(float)

with open("mock_pranjal/example_config.yaml", "r", encoding="utf-8") as file1:
    config = yaml.safe_load(file1)

config['galaxy_params']['line_species']    = ['Halpha', 'Halpha']
config['galaxy_params']['obs_type']        = 'slit'
config['galaxy_params']['log10_Mstar']     = data_info['galaxy']['log10_Mstar']
config['galaxy_params']['log10_Mstar_err'] = data_info['galaxy']['log10_Mstar_err']

config['likelihood']['fid_params'] = {
    'shared_params': {
          'beta': 0,
          'v_0': 0,
          'image_snr': 80,
          'spec_snr': 40,
          }
    }


inference = UltranestSampler(data_info, config)

#-------------------------  Run Ultranest --------------------------- #
""" To see each iter/eval's param set in fitting, debug by setting 
    a breakpoint on Line 2628 in site-packages/ultranest/integrator.py
    or Line 120 in ./ultranest_sampler.py 
"""
sampler = inference.run(output_dir=f'./run0.{run}/', test_run=False, run_num=run)

#----------------------- Save sample results --------------------------- #
save_path = './'
with open(f'{save_path}/run0.{run}/ultranest_sampler_results.pkl', 'wb') as f:
    pickle.dump(sampler.results, f)

estimates = sampler.results['maximum_likelihood']['point']

best_fit_dict = inference.params.gen_param_dict(inference.config.params.names, 
                                                estimates)
line = config['galaxy_params']['line_species']
this_line_dict = {**best_fit_dict['shared_params'], 
                  **best_fit_dict[f'{line[0]}_params']}
best_fit_params = {'best_fit_dict': best_fit_dict,
                   'this_line_dict': this_line_dict}
with open(f'{save_path}/run0.{run}/ultranest_best_fit_params.pkl', "wb") as f:
    joblib.dump(best_fit_params, f)

#------------------- Main Plot Module ---------------------- #
sampler.print_results()
sampler.plot()
# sampler.plot_trace()

# samples = sampler.results['samples']

# MC_samples = getdist.MCSamples(samples=samples, 
#                                names=inference.config.params.names, 
#                                sampler='nested')

json_filename = f'{save_path}run0.{run}/run{run}/info/results.json'

estimates, best_fit_params, best_fit_core_params = load_best_fit_json(inference, config['params'], json_filename)

plot_obs_fit_res(data_info, 
                 inference, best_fit_params, best_fit_core_params, 
                 run, save_path)
    