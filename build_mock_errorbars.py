import numpy as np
import pickle
import matplotlib.pyplot as plt
plt.style.use('classic')
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.serif": "Helvetica",
})
colors=["#C82423", # "darkred"
        "#F54C22", # "oranred"
        "#8fc4a5", # "greedrk"
        "#1D3557", # "blueblk"
        "#457B9D", # "blueash"
        "#74A9CF", # "blueook"
        "#A8DADC"] # "bluegre"

def analyze_percentile(samples):
    dic_percent = {}
    nparams = len(samples[0])
    arr = np.zeros((nparams, 3))
    
    for j in range(nparams):
        samp_points = samples[:,j]
        x123 = np.percentile(samp_points, [16, 50, 84])
        err_lo, mean, err_hi = x123[0]-x123[1], x123[1], x123[2]-x123[1]
        arr[j] = np.around([mean, err_lo, err_hi], decimals=4)
    
    dic_percent = {
        'g1': {        'mean': arr[0,0], 'err_lo': arr[0,1], 'err_hi': arr[0,2]},
        'g2': {        'mean': arr[1,0], 'err_lo': arr[1,1], 'err_hi': arr[1,2]},
        'vcirc': {     'mean': arr[2,0], 'err_lo': arr[2,1], 'err_hi': arr[2,2]},
        'cosi': {      'mean': arr[3,0], 'err_lo': arr[3,1], 'err_hi': arr[3,2]}, 
        'theta_int': { 'mean': arr[4,0], 'err_lo': arr[4,1], 'err_hi': arr[4,2]}, 
        'vscale': {    'mean': arr[5,0], 'err_lo': arr[5,1], 'err_hi': arr[5,2]}, 
        'r_hl_disk': { 'mean': arr[6,0], 'err_lo': arr[6,1], 'err_hi': arr[6,2]},
        'r_hl_bulge': {'mean': arr[7,0], 'err_lo': arr[7,1], 'err_hi': arr[7,2]},
        'flux': {      'mean': arr[8,0], 'err_lo': arr[8,1], 'err_hi': arr[8,2]},
        'flux_bulge': {'mean': arr[9,0], 'err_lo': arr[9,1], 'err_hi': arr[9,2]}, 
        'I01': {       'mean': arr[10,0], 'err_lo': arr[10,1], 'err_hi': arr[10,2]}, 
        'I02': {       'mean': arr[11,0], 'err_lo': arr[11,1], 'err_hi': arr[11,2]}, 
        'bkg_level': { 'mean': arr[12,0], 'err_lo': arr[12,1], 'err_hi': arr[12,2]}, 
        }
    return dic_percent

def make_a_dic_for_slits(iter_nums, if_add_noise):
    dic_all_iter = {}
    for i in range(len(iter_nums)):
        for j in range(len(iter_nums[0])):
            iter_num = iter_nums[i][j]
            # if iter_num < 75.5:
            try:
                samples = np.load(f'../../../../RSCH3/kl_github/runs_March_2025/run0.{iter_num}/ultranest_sampler_samples.npy')
            except FileNotFoundError:
                with open(f'../../../../RSCH3/kl_github/runs_March_2025/run0.{iter_num}/ultranest_sampler_results.pkl', 'rb') as f:
                    sample_results = pickle.load(f) # read
                # sample_results = np.load(f'../../../../RSCH3/kl_github/runs_March_2025/run0.{iter_num}/ultranest_sampler_results.npy')
                samples        = sample_results['samples']
            percentile = analyze_percentile(samples)
            dic_all_iter[iter_num] = {
                    "iter_num":   iter_num, 
                    "add_noise":  if_add_noise,
                    "cosi":       cosis[i][j], 
                    "spec_snr":   spec_snrs[i][j],
                    "percentile": percentile
                    }
            # else:
            #     dic_all_iter[iter_num] = {
            #             "iter_num":   iter_num, 
            #             "add_noise":  if_add_noise,
            #             "cosi":       cosis[i][j], 
            #             "spec_snr":   spec_snrs[i][j],
            #             "percentile": percentile
            #             }
    return dic_all_iter

def plot_diagnostics(dic_all_iter, if_add_noise=False):
    fig = plt.figure(figsize=(6*n_param_to_see, 3*n_spec_snr))  # (length, height)
    plt.subplots_adjust(hspace=0.4, wspace=0.4) # h=height
    gs = fig.add_gridspec(nrows=n_spec_snr, ncols=n_param_to_see, 
                          height_ratios=[1]*n_spec_snr, 
                          width_ratios=[1]*n_param_to_see)
    
    for i in range(n_spec_snr):
        this_spec_snr = spec_snrs[i,0]
        for k in range(len(want_to_see)):
            param = want_to_see[k]
            cosi_values = [dic["cosi"] 
                           for i_num, dic in dic_all_iter.items() 
                           if dic["spec_snr"] == this_spec_snr and dic['add_noise'] == if_add_noise]
            g1_values   = [dic["percentile"][param] 
                           for i_num, dic in dic_all_iter.items() 
                           if dic["spec_snr"] == this_spec_snr and dic['add_noise'] == if_add_noise]
            g1_mean     = np.array([g1_this_i['mean'  ] for g1_this_i in g1_values])
            g1_error_lo = np.array([g1_this_i['err_lo'] for g1_this_i in g1_values])
            g1_error_hi = np.array([g1_this_i['err_hi'] for g1_this_i in g1_values])
            g1_mean_off = g1_mean - want_to_see_ref[k]
            g1_error    = [np.abs(g1_error_lo), g1_error_hi]
        
            ax1 = fig.add_subplot(gs[i, k])
            
            ax1.errorbar(cosi_values, g1_mean_off, yerr=g1_error, 
                         fmt=' ', capsize=5, capthick=1, elinewidth=1, 
                         marker='o', markersize=6, color=colors[k], 
                         label=param+f' (mock: {want_to_see_ref[k]})')
            
            xlim_max = cosi_values[0]+(cosi_values[-1]-cosi_values[0])*1.2
            ax1.hlines(y=0, xmin=0, xmax=xlim_max, 
                       color='gray', linestyle='--', linewidth=1)
            ax1.text(0.01, 0.03, 
                     f'Avg offset = {np.mean(g1_mean_off):.2f}   '+
                     f'Avg error = {np.mean(np.array(g1_error).flatten()):.2f}', 
                     fontsize=12, color='brown', ha='left', va='bottom', 
                     transform=ax1.transAxes)
            ax1.set_xlabel(r'Inclination $\cos{(i)}$', labelpad=0)
            ax1.set_ylabel(r'Offset = Fit - Mock setting')
            ax1.grid(linestyle=':', color='black', alpha=0.5)
            if if_add_noise:
                ax1.set_title('Mocks: SNR='+f'{this_spec_snr}, noise added', fontsize=12)
            else:
                ax1.set_title('Mocks: SNR='+f'{this_spec_snr}, noise-free', fontsize=12)
            ax1.legend(prop={'size': 10})
    
    plt.savefig(f'mock_100/mock_chain_diagnostics_noise_{str(if_add_noise)}.png', dpi=150, bbox_inches='tight')
    return

# sampler = inference.run()
# results = sampler.results
# paramnames = results['paramnames']
# data = np.array(results['weighted_samples']['points'])
# weights = np.array(results['weighted_samples']['weights'])

iter_nums = [ [ 1,  2,  3,  4,  5,  6,  7,  8],
              [11, 12, 13, 14, 15, 16, 17, 18],
              [21, 22, 23, 24, 25, 26, 27, 28],
              [31, 32, 33, 34, 35, 36, 37, 38] ]
iter_nums_no_noise = [[41, 42, 43, 44, 45, 46, 47, 48],
                      [51, 52, 53, 54, 55, 56, 57, 58],
                      [61, 62, 63, 64, 65, 66, 67, 68],
                      [71, 72, 73, 74, 75, 76, 77, 78]]
cosis     = np.tile([0.125, 0.25, 0.375, 0.5, 0.675,  0.75, 0.825, 1], 
                    (len(iter_nums), 1))
spec_snrs = np.tile([[10], [25], [40], [55]], 
                    (1, len(iter_nums[0])))
n_cosi, n_spec_snr = len(iter_nums[0]), len(iter_nums)

dic_all_iter_noise_yes = make_a_dic_for_slits(iter_nums, if_add_noise=True)
dic_all_iter_noise_no  = make_a_dic_for_slits(iter_nums_no_noise, if_add_noise=False)

want_to_see     = ['g1','g2','vcirc','theta_int','vscale']
want_to_see_ref = [ 0,   0.2, 142,    0,          0.2]
n_param_to_see  = len(want_to_see)

plot_diagnostics(dic_all_iter_noise_no, if_add_noise=False)