import joblib
# import pickle
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import galsim
from astropy import wcs
# from astropy.io import fits
# from astropy.coordinates import SkyCoord
from scipy.stats import qmc

from parameters  import Parameters
from spec_model  import SlitModel
from image_model import ImageModel
from mock import Mock

plt.style.use('classic')
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.serif": "Helvetica",
})
    
def prepare_a_slitspec(RA_obj, Dec_obj):
    """
        Make a dic for slit data
    """
    slit_len   =  8
    slit_width =  1
    slit_LPA   = 25*u.deg
    slit_LPA   = 90*u.deg - slit_LPA  # degrees east of north. Want w.r.t. east.
    
    # Only select one from these 3 RA-Dec settings
    # (1) Set C
    slit_RA    = RA_obj
    slit_Dec   = Dec_obj
    # (2) Set A ONLY: 1 arcsec offset (from Set C slit)
    # slit_RA    = RA_obj  - slit_width*u.arcsec * np.sin(slit_LPA)
    # slit_Dec   = Dec_obj + slit_width*u.arcsec * np.cos(slit_LPA)
    # (3) Set B ONLY: -1 arcsec offset
    # slit_RA    = RA_obj  + slit_width*u.arcsec * np.sin(slit_LPA)
    # slit_Dec   = Dec_obj - slit_width*u.arcsec * np.cos(slit_LPA)
    
    spec_pix_scale  = 1/3.2     # arcsec/pix # Binospec: 3.2 px/arcsec
    spec_shape      = [32, 40]  # (spatial pixels, N wavelength points)
    lamb_mean    = 7232         # A
    lamb_disp    =    0.62      # A/px
    lamb_min     = lamb_mean - lamb_disp *  int(spec_shape[1]/2)
    lamb_max     = lamb_min  + lamb_disp * (int(spec_shape[1]) - 1) # N-1 segments
    lamb_one_row = np.linspace(lamb_min, lamb_max, int(spec_shape[1]))
    lambda_grid  = np.tile(lamb_one_row, (int(spec_shape[0]), 1))
    meta_spec = {
        'line_species':   'OII', 
        'ngrid':          spec_shape,
        'lambda_grid':    lambda_grid*u.Angstrom,
        'pixScale':       spec_pix_scale,  # arcsec/px
        'rhl':            None, # See make_mock_data() note below
        'slitRA':    slit_RA,
        'slitDec':   slit_Dec,
        'slitWidth': slit_width,
        'slitLen':   slit_len,
        'slitLPA':   slit_LPA,
        'slitWPA':   slit_LPA + 90*u.deg  # Assume rectangular slit
    }
    return meta_spec

def prepare_an_image(RA_obj, Dec_obj):
    """
        Make a dic for image data
    """
    image_shape     = (40, 40)
    image_pix_scale = 0.2     # arcsec/pix: 0.2 for HSC image
    psfFWHM         = 0.6       # arcsec
    ap_wcs           = wcs.WCS(naxis=2) # Create WCS
    ap_wcs.wcs.crpix = np.array([image_shape[0]/2+0.5,
                                 image_shape[0]/2+0.5]) # Cntrl ref pix (0.5-based)
    ap_wcs.wcs.crval = [RA_obj.value, Dec_obj.value] # RA/Dec (deg) central pixel
    ap_wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN'] 
    ap_wcs.wcs.cdelt = [-image_pix_scale/3600, image_pix_scale/3600] # deg/px
    galsim_wcs       = galsim.AstropyWCS(wcs=ap_wcs)
    meta_image = {
        'ngrid':    image_shape,
        'pixScale': image_pix_scale,
        'psfFWHM':  psfFWHM,
        'wcs':      galsim_wcs,
        'ap_wcs':   ap_wcs,
        'RA':       RA_obj.value,
        'Dec':      Dec_obj.value}
    
    return meta_image

def make_mock_data(emi_line_lst, meta_gal, mock_params):
    """
        Start generating data by using dic we created
    """
    meta_imag  = prepare_an_image(  meta_gal['RA'], meta_gal['Dec'])
    meta_spec  = prepare_a_slitspec(meta_gal['RA'], meta_gal['Dec'])
    
    params       = Parameters(line_species=emi_line_lst)
    updated_dict = params.gen_param_dict(mock_params.keys(), 
                                         mock_params.values())
    
    # Note: SlitModel requires a line_profile_path but now I don't have one.
    #       I assume a rhl = average of rhl_disk and rhl_bulge.
    meta_spec['rhl'] = (mock_params['shared_params-r_hl_disk']+
                        mock_params['shared_params-r_hl_bulge'])/2
    spec_snr  = mock_params['shared_params-spec_snr']
    image_snr = mock_params['shared_params-image_snr']
    
    spec_model      = SlitModel(obj_param=meta_gal, 
                                meta_param=meta_spec)
    this_line_dict  = {**updated_dict['shared_params'], 
                       **updated_dict[f'{emi_line_lst[0]}_params']} # merge dict
    spec_data       = spec_model.get_observable(this_line_dict)
    spec_var        = np.ones(spec_data.shape) * 100

    cont_model_spec = np.zeros(spec_data.shape)
    spec_var        = Mock._set_snr(spec_data, spec_var, # Set var to match SNR
                                    spec_snr, 'spec', verbose=False)
    spec_noise      = np.sqrt(spec_var)*np.random.randn(spec_var.shape[0],
                                                        spec_var.shape[1])/2 # if adding noise
    spec_data       = spec_data + spec_noise
    
    image_model   = ImageModel(meta_image=meta_imag)
    image_data    = image_model.get_image(updated_dict['shared_params'])
    image_sky_var = np.ones(image_data.shape) * 100
    image_var     = image_data + image_sky_var
    image_var     = Mock._set_snr(image_data, image_var, # Set var to match SNR
                                  image_snr, 'image', verbose=False)
    image_noise   = np.sqrt(image_var)*np.random.randn(image_var.shape[0],
                                                       image_var.shape[1])/2 # if adding noise
    image_data    = image_data + image_noise
    
    print('Final image SNR:', round(Mock._calculate_snr(image_data, image_var, 'image')))
    print('Final spec  SNR:', round(Mock._calculate_snr(spec_data,  spec_var,  'spec')))
    
    return {'meta_spec':  meta_spec,  'spec_data':  spec_data,  'spec_var':  spec_var,
            'meta_image': meta_imag, 'image_data': image_data, 'image_var': image_var, 
            'cont_model_spec': cont_model_spec }

def save_dic_and_pkl(mask, slit_name, 
                     mock_data, mock_params, meta_gal, iter_num):
    """
        Make a dic and pkl to save all generated data
    """
    spec_data_info = {
        'data':       mock_data['spec_data'],
        'var':        mock_data['spec_var'],
        'cont_model': mock_data['cont_model_spec'],
        'par_meta':   mock_data['meta_spec']
        }
    
    image_data_info = {
        'data':     mock_data['image_data'],
        'var':      mock_data['image_var'],
        'par_meta': mock_data['meta_image']
        }
    
    mock_data_info = {
        'spec':    [spec_data_info],
        'image':   image_data_info,
        'galaxy':  meta_gal,
        'par_fit': mock_params}
    
    # Unfortunately, my galsim.wcs objects cannot be packed in PKL files.
    # To pack galsim.wcs, DELETE it before packing in PKL.
    # To read, always regenerate by using ap_wcs. (by JD)
    mock_data_info['image']['par_meta'].pop('wcs') # delete it!
    
    with open(f'mock_500/pkl/mock_{mask}_{slit_name}_{iter_num}.pkl', "wb") as f:
        joblib.dump(mock_data_info, f)

    return mock_data_info

def make_exam_plots(mock_data_info, mask, slit_name, iter_num):
    fig = plt.figure(figsize=(13, 5))  # (length, height)
    plt.subplots_adjust(hspace=0.4, wspace=0.0) # h=height
    gs = fig.add_gridspec(nrows=1, ncols=2, 
                          height_ratios=[1], 
                          width_ratios=[1, 2])
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], 
                          projection=mock_data_info['image']['par_meta']['ap_wcs'])
    
    spec_data = mock_data_info['spec'][0]['data']
    im_spec = ax1.imshow(spec_data, origin='lower', 
                         extent=[mock_data_info['spec'][0]['par_meta']['lambda_grid'][0, 0].value, # x_left
                                 mock_data_info['spec'][0]['par_meta']['lambda_grid'][0,-1].value, # x_right
                                 -mock_data_info['spec'][0]['par_meta']['pixScale']*len(spec_data)/2,  # y_bottom
                                 +mock_data_info['spec'][0]['par_meta']['pixScale']*len(spec_data)/2], # y_top
                         cmap='viridis', aspect='auto')
    fig.colorbar(im_spec, ax=ax1)
    ax1.set_ylim(-mock_data_info['spec'][0]['par_meta']['pixScale']*len(spec_data)/2,
                 +mock_data_info['spec'][0]['par_meta']['pixScale']*len(spec_data)/2)
    ax1.xaxis.get_major_formatter().set_useOffset(False)
    ax1.set_xlabel(r'Observed Wavelength $\lambda$ ($\AA$)')
    ax1.set_ylabel(r'Spatial Position (arcsec)')
    ax1.grid(linestyle=':', color='white', alpha=0.5)
    ax1.set_title('Mock MMT/Binospec 2D Spec')
    
    def rot_rectangle(ax, x0, y0, dx, dy, rotation, color, ls):
        xUL = x0 + (-dx/2)*np.cos(rotation) - (+dy/2)*np.sin(rotation)
        yUL = y0 + (-dx/2)*np.sin(rotation) + (+dy/2)*np.cos(rotation)
        xUR = x0 + (+dx/2)*np.cos(rotation) - (+dy/2)*np.sin(rotation)
        yUR = y0 + (+dx/2)*np.sin(rotation) + (+dy/2)*np.cos(rotation)
        xLL = x0 + (-dx/2)*np.cos(rotation) - (-dy/2)*np.sin(rotation)
        yLL = y0 + (-dx/2)*np.sin(rotation) + (-dy/2)*np.cos(rotation)
        xLR = x0 + (+dx/2)*np.cos(rotation) - (-dy/2)*np.sin(rotation)
        yLR = y0 + (+dx/2)*np.sin(rotation) + (-dy/2)*np.cos(rotation)
        ax.plot([xUL, xUR, xLR, xLL, xUL], 
                [yUL, yUR, yLR, yLL, yUL], color=color, linestyle=ls)
        return 
    
    image_data = mock_data_info['image']['data']
    objRA      = mock_data_info['image']['par_meta']['RA']
    objDec     = mock_data_info['image']['par_meta']['Dec']
    ap_wcs     = mock_data_info['image']['par_meta']['ap_wcs']
    pixScale   = mock_data_info['image']['par_meta']['pixScale']
    slitRA     = mock_data_info['spec'][0]['par_meta']['slitRA'].value
    slitDec    = mock_data_info['spec'][0]['par_meta']['slitDec'].value
    slitLen    = mock_data_info['spec'][0]['par_meta']['slitLen']
    slitWidth  = mock_data_info['spec'][0]['par_meta']['slitWidth']
    slit_LPA   = mock_data_info['spec'][0]['par_meta']['slitLPA'].value
    im_imag = ax2.imshow(image_data, origin='lower', 
                         cmap='viridis', aspect='equal')
    fig.colorbar(im_imag, ax=ax2)
    x0obj,  y0obj  = ap_wcs.wcs_world2pix([[objRA,  objDec ]], 0)[0]  
    x0slit, y0slit = ap_wcs.wcs_world2pix([[slitRA, slitDec]], 0)[0]  
    ax2.scatter(x0obj, y0obj, marker='x', s=360, color='black', zorder=1)
    rot_rectangle(ax2, x0slit, y0slit, slitWidth/pixScale, slitLen/pixScale, 
                  (90-slit_LPA)/57.3, 'white', '-')
    ax2.set_xlim(left=0, right=image_data.shape[1]-1)
    ax2.set_ylim(bottom=0, top=image_data.shape[0]-1)
    ax2.coords['ra' ].set_major_formatter('dd:mm:ss')
    ax2.coords['dec'].set_major_formatter('dd:mm:ss')
    ax2.set_xlabel('RA')
    ax2.set_ylabel('Dec')
    ax2.grid(linestyle=':', color='white', alpha=0.5)
    ax2.set_title('Mock Subaru Imaging')
    
    ax1txts, ax2txts = '', ''
    for key, arr in mock_data_info['par_fit'].items():
        if key[:14] == 'shared_params-':
            ax2txts += (key[14:] + ' = ' + '{:.3g}'.format(arr) + '\n')
        elif key[:11] == 'OII_params-':
            ax1txts += (key[11:] + ' = ' + '{:.2g}'.format(arr) + '\n')
    ax2.text(1, 1, ax2txts, fontsize=10, color='white', ha='right', va='top', 
             transform=ax2.transAxes)
    ax1.text(1, 1, ax1txts, fontsize=10, color='white', ha='right', va='top', 
             transform=ax1.transAxes)
    
    plt.savefig(f'mock_500/mock_{mask}_{slit_name}_{iter_num}.png', 
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    return




# Now get object data from object catalog
mask      = 'C'
slit_name = '095'
RA_obj     = 42.00292 *u.deg
Dec_obj    = -3.404363*u.deg
redshift   =  0.94
emi_line_lst = ['OII']
meta_gal = {}
meta_gal = {
            'redshift': redshift,
            'RA':      RA_obj,
            'Dec':     Dec_obj,
            'beta':    0*u.deg,
            # 'log10_Mstar':    10.0,
            # 'log10_Mstar_err': 0.2,
            }

iter_num      = 452

mock_params = {
    'shared_params-image_snr': 80,
    'shared_params-spec_snr':  20,
    'shared_params-beta': meta_gal['beta'],
    
    # 'shared_params-gamma_t':    0.2,
    'shared_params-g1':         0.0,
    'shared_params-g2':         0.2,
    'shared_params-cosi':       0.5,
    'shared_params-theta_int':  0.0,
    'shared_params-vscale':     0.2,
    'shared_params-r_hl_disk':  1.0,
    'shared_params-r_hl_bulge': 0.8,
    
    'shared_params-vcirc':    142,
    'shared_params-v_0':        0,
    'shared_params-dx_disk':    0,
    'shared_params-dy_disk':    0,
    'shared_params-dx_bulge':   0,
    'shared_params-dy_bulge':   0,
    'shared_params-flux':       3.0, # = log(F)
    'shared_params-flux_bulge': 2.0, # = log(F_bulge)
    
    'OII_params-dx_vel':      0,
    'OII_params-dx_vel_2':    0,
    'OII_params-I01':        10,
    'OII_params-I02':        10,
    'OII_params-bkg_level':   1
}

mock_data      = make_mock_data(emi_line_lst, meta_gal, mock_params)
mock_data_info = save_dic_and_pkl(mask, slit_name, 
                                  mock_data, mock_params, meta_gal, iter_num)
make_exam_plots(mock_data_info, mask, slit_name, iter_num)



# Sampling method: Latin Hypercube (LHS)
how_many_iter = 10
# sampler = qmc.LatinHypercube(d=8, seed=0)
# sample = sampler.random(n=how_many_iter)
# image_snr = [24, 132]
# spec_snr  = [13,  33]
# gamma_t   = [-0.1, 0.1]
# cosi      = [   0,   1]
# theta_int = [   0, 2*np.pi]
# vscale     = [0.1,  3]
# r_hl_disk  = [0.15, 3]
# r_hl_bulge = [0.15, 3]
# l_bounds = [image_snr[0], spec_snr[0],  gamma_t[0], cosi[0], theta_int[0], 
#             vscale[0],    r_hl_disk[0], r_hl_bulge[0]]
# u_bounds = [image_snr[1], spec_snr[1],  gamma_t[1], cosi[1], theta_int[1], 
#             vscale[1],    r_hl_disk[1], r_hl_bulge[1]]
# for sampled_params in qmc.scale(sample, l_bounds, u_bounds):
#     iter_num += 1
#     mock_params['shared_params-image_snr']  = sampled_params[0]
#     mock_params['shared_params-spec_snr']   = sampled_params[1]
#     mock_params['shared_params-gamma_t']    = sampled_params[2]
#     mock_params['shared_params-cosi']       = sampled_params[3]
#     mock_params['shared_params-theta_int']  = sampled_params[4]
#     mock_params['shared_params-vscale']     = sampled_params[5]
#     mock_params['shared_params-r_hl_disk']  = sampled_params[6]
#     mock_params['shared_params-r_hl_bulge'] = sampled_params[7]
#     mock_data      = make_mock_data(emi_line_lst, meta_gal, mock_params)
#     mock_data_info = save_dic_and_pkl(mask, slit_name, 
#                                       mock_data, mock_params, meta_gal, iter_num)
#     make_exam_plots(mock_data_info, mask, slit_name, iter_num)

# Sampling method: Random in Gaussian
# parameters = [
#     # keyname, mean,sigma,   min, max
#     ("g1",          0,   0.5,  -0.5,  0.5),
#     ("g2",          0,   0.5,  -0.5,  0.5),
#     ("cosi",      0.5,   0.5,     0,    1),
#     ("theta_int",   0, np.pi,-np.pi,np.pi),
#     ("vscale",    1.5,   1.5,   0.1,    3),
#     ("r_hl_disk", 1.5,   1.5,  0.15,    3),
#     ("r_hl_bulge",1.5,   1.5,  0.15,    3),
#     ("flux",        3,     2,     1,    5),
#     ("flux_bulge",  3,     2,     1,    5),
#     ("I01",        25,    25,     0,   50),
#     ("I02",        25,    25,     0,   50),
#     ("bkg",         5,     5,     0,   10),
# ]
# all_sampled_params = {}
# for param, mean, std, lower, upper in parameters:
#     values = []
#     while len(values) < how_many_iter:
#         sample = np.random.normal(mean, std, 1)[0]
#         if lower <= sample <= upper:
#             values.append(sample)
#     all_sampled_params[param] = values
# def transform_to_list(sampled_params):
#     keys = list(sampled_params.keys())
#     num_samples = len(next(iter(sampled_params.values())))
#     b = [
#         {key: sampled_params[key][i] for key in keys}
#         for i in range(num_samples)
#     ]
#     return b
# param_list = transform_to_list(all_sampled_params)
# for i in range(len(param_list)):
#     iter_num += 1
#     mock_params['shared_params-g1']         = param_list[i]['g1']
#     mock_params['shared_params-g2']         = param_list[i]['g2']
#     mock_params['shared_params-cosi']       = param_list[i]['cosi']
#     mock_params['shared_params-theta_int']  = param_list[i]['theta_int']
#     mock_params['shared_params-vscale']     = param_list[i]['vscale']
#     mock_params['shared_params-r_hl_disk']  = param_list[i]['r_hl_disk']
#     mock_params['shared_params-r_hl_bulge'] = param_list[i]['r_hl_bulge']
#     mock_params['shared_params-flux']       = param_list[i]['flux']
#     mock_params['shared_params-flux_bulge'] = param_list[i]['flux_bulge']
#     mock_params['OII_params-I01']           = param_list[i]['I01']
#     mock_params['OII_params-I02']           = param_list[i]['I02']
#     mock_params['OII_params-bkg_level']     = param_list[i]['bkg']
#     mock_data      = make_mock_data(emi_line_lst, meta_gal, mock_params)
#     mock_data_info = save_dic_and_pkl(mask, slit_name, 
#                                       mock_data, mock_params, meta_gal, iter_num)
#     make_exam_plots(mock_data_info, mask, slit_name, iter_num)