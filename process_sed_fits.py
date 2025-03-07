import glob
import json
import numpy as np
from scipy import stats

import prospect.io.read_results as reader
import prospect.models

import ipdb

imf = 1
dust = 4

sed_result_paths = glob.glob(f'/xdisk/timeifler/pranjalrs/sed_fit/*parametric_imf{imf}_dust{dust}.h5')

masks = ['a2261aB', 'a2261b', 'a2261c', 'a2261d']
stellar_mass = {m:{} for m in masks}

for f in sed_result_paths:
    result, obs, _ = reader.results_from(f, dangerous=False)
    this_mask, slit = f.split('/')[-1].split('_')[:2]
    
    shape = result['chain'].shape
    chain = result['chain'].reshape(shape[0]*shape[1], shape[2])
    stellar_mass_mean = np.mean(chain[:, 0])
    stellar_mass_err = np.std(np.log10(chain[:, 0]))
    stellar_mass[this_mask][slit] = [stellar_mass_mean, stellar_mass_err]

with open(f'../data/stellar_mass_imf{imf}_dust{dust}.json', 'w') as json_file:
    json.dump(stellar_mass, json_file, indent=4)

