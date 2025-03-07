import joblib
import json
import matplotlib.pyplot as plt
import numpy as np


from emission_line import EmissionLine
from parameters import Parameters

# with open('../data/targets_draft.json') as f:
#     data = json.load(f)

# for mask in data.keys():
#     for i in range(len(data[mask])):
#         slit = data[mask][i][0]
#         files.append(f'../../zzz_spec_095.pkl')

files = ['../zzz_spec_extract_spec_095_T.pkl']
outfilename = 'zzz_spec_extract_095_OII_T2.pkl'

save_path = './'

for f in files:
    data_info = joblib.load(f)

    nspec = len(data_info['spec'])

    extracted_spec = []
    data_filtered = []
    line_pars = {}

    fig, axs = plt.subplots(nspec, 5, figsize=(20, 3*nspec), facecolor='white')

    if nspec==1: axs = np.array([axs])

    axs[0, 0].set_title('Observation')
    axs[0, 1].set_title('Median filtered obs.')
    axs[0, 2].set_title('Extracted spectrum')
    axs[0, 3].set_title('Continuum Model')

    for i in range(nspec):
        line_name = data_info['spec'][i]['par_meta']['line_species']
        z = data_info['galaxy']['redshift']
        line_wav = Parameters(). _get_species(line_name, z)[1]
        try:
            results, errors = EmissionLine.extract_emission(data_info['spec'][i], 
                                                   return_fit=True)
#             data = joblib.load('zzz_spec_extract_spec_095.pkl')[line_name]
            # save .pkl (added by JD)
            with open(outfilename, 'wb') as f: # (added by JD)
                joblib.dump(results, f)        # (added by JD)
            amplitude, mu, sigma1, sigma2 = results[0], results[1], results[2], results[3]
            amplitude_err, mu_err, sigma1_err, sigma2_err = errors[0], errors[1], errors[2], errors[3]
            
            line_pars[line_name] = [amplitude, mu, sigma1, sigma2]
            line_pars[line_name] += [amplitude_err, mu_err, sigma1_err, sigma2_err]

            ax = axs[i]
            ## Save extracted spectrum
            im0 = ax[0].imshow(data_info['spec'][i]['data'], aspect='auto')
            im1 = ax[1].imshow(results[4], aspect='auto')
            im2 = ax[2].imshow(results[5], aspect='auto')

            ax[3].plot(data_info['spec'][i]['cont_model'])

            # Now plot rotation curve
            x_vals = np.linspace(-3, 3, len(mu[:,0]))
            idx = mu[:, 0] !=0
            rc = ((mu[:,0]/line_wav[0]).decompose()-1)*299792.45
            rc_err = (mu_err[:,0]/line_wav[0]).decompose()*299792.45
            rc_err = np.clip(rc_err, 0, 50)
            ax[4].errorbar(x_vals[idx], rc[idx], yerr=rc_err[idx], fmt='o')

            if np.any(mu[:, 1] != 0):
                rc_err = np.clip(rc_err, 0, 50)
                rc = ((mu[:,1]/line_wav[1]).decompose()-1)*299792.45
                rc_err = (mu_err[:,1]/line_wav[1]).decompose()*299792.45
                ax[4].errorbar(x_vals[idx], rc[idx], yerr=rc_err[idx], fmt='o')

            ax[4].set_ylim(np.min(rc[(rc_err<20) & (rc_err>0)])-30, np.max(rc[(rc_err<20) & (rc_err>0)])+30)
            ax[4].set_xlim(-3, 3)
            ax[0].text(10, 10, line_name, c='white', fontsize=20, weight='bold')
            fig.colorbar(im0, ax=ax[0])
            fig.colorbar(im1, ax=ax[1])
            fig.colorbar(im2, ax=ax[2])

        except FileNotFoundError:
            print(f'FileNotFoundError: Failed for {f}')
            continue
    plt.savefig('zzz_spec_extract_095_OII_T2.png', dpi=300, bbox_inches='tight')
    plt.close()
