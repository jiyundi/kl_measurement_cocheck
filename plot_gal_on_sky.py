## For each slit, overplots contours for all emission lines
import argparse
import joblib
import json
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.patches import Rectangle
import numpy as np
import scipy.stats

import astropy.units as u
from astropy.io import fits
import astropy.wcs as wcs
from astropy.coordinates import SkyCoord
import galsim

import klm.utils as utils

def get_slit_corners(RA, Dec, PA, LEN, WID):
    
    theta = np.pi - PA
    L = LEN/3600.
    dx = WID/3600.

    ul_ra = RA -  L*np.cos(theta)/2. + dx * np.sin(theta)/2.
    ur_ra = RA +  L*np.cos(theta)/2. + dx * np.sin(theta)/2.
    ll_ra = RA -  L*np.cos(theta)/2. - dx * np.sin(theta)/2.
    lr_ra = RA +  L*np.cos(theta)/2. - dx * np.sin(theta)/2.

    ul_dec = Dec + L*np.sin(theta)/2. + dx * np.cos(theta)/2.
    ur_dec = Dec - L*np.sin(theta)/2. + dx * np.cos(theta)/2.
    ll_dec = Dec + L*np.sin(theta)/2. - dx * np.cos(theta)/2.
    lr_dec = Dec - L*np.sin(theta)/2. - dx * np.cos(theta)/2.

    ul_pos = this_wcs.wcs_world2pix(ul_ra, ul_dec, 1)
    ur_pos = this_wcs.wcs_world2pix(ur_ra, ur_dec, 1)
    ll_pos = this_wcs.wcs_world2pix(ll_ra, ll_dec, 1)
    lr_pos = this_wcs.wcs_world2pix(lr_ra, lr_dec, 1)

    pos = np.vstack((ul_pos, ur_pos, lr_pos, ll_pos, ul_pos))
    return pos


run = 'run3'
stats = joblib.load(f'stats_DEIMOS/results_{run}.pkl')


path = '../../Data/klens_data/images/hlsp_clash_subaru_suprimecam_a2261_rc_2004-v20110514_drz.fits'
subaru_image = fits.open(path)

fig = plt.figure(figsize=(20, 20))
this_wcs = wcs.WCS(subaru_image[0].header)
ax = fig.add_subplot(111, projection=this_wcs)



lon = ax.coords[0]
lat = ax.coords[1]
lon.set_major_formatter('d.dd')
lat.set_major_formatter('d.dd')

im = ax.imshow(subaru_image[0].data, cmap=plt.cm.seismic, vmin=-1, vmax=1, rasterized=True)
xlow, ylow = this_wcs.wcs_world2pix(260.57, 32.1, 1)
xup, yup = this_wcs.wcs_world2pix(260.65, 32.16, 1)

xlow, ylow = int(xlow), int(ylow)
xup, yup = int(xup), int(yup)

ax.set_xlim([xup, xlow]) # Don't change
ax.set_ylim([ylow, yup])


a2261_x, a2261_y = this_wcs.wcs_world2pix(260.612917, 32.133889, 1)
ax.scatter(a2261_x, a2261_y, c='royalblue', marker='^')

ax.axvline(a2261_x, c='gray', ls=':')
ax.axhline(a2261_y, c='gray', ls=':')

ax.set(xlabel='R. A.', ylabel='Dec')

inset_axes_pos = {'a2261aB_024': [0.65, 0.78, 0.2, 0.2],
                'a2261b_007': [0.65, 0.02, 0.2, 0.2],
                'a2261c_007': [0.2, 0.2, 0.2, 0.2],
                'a2261c_037': [0.9, 0.5, 0.2, 0.2],
                'a2261d_027': [0.8, 0.25, 0.2, 0.2]}

# axins = zoomed_inset_axes(ax, 4) # zoom = 6
# Plot object
for i, obj in enumerate(list(stats.keys())):
    if obj in ['a2261b_012', 'a2261d_025', 'a2261b_065', 'a2261c_060']:
        continue

    data_info = joblib.load(f'../data/data_info/{obj}.pkl')
    beta = data_info['galaxy']['beta'].to(u.radian).value
    RA, Dec = data_info['galaxy']['RA'].value, data_info['galaxy']['Dec'].value
    
    meta_pars =  data_info['spec'][0]['par_meta']
    slitRA, slitDec = meta_pars['slitRA'].value, meta_pars['slitDec'].value
    slitLPA =  meta_pars['slitLPA'].to(u.radian).value

    ## Show object name
    pix_x, pix_y = this_wcs.wcs_world2pix(RA, Dec, 1)
    ax.text(x=pix_x, y=pix_y+30, s=obj, color='k', weight='bold', fontsize=9)


    ## Make zoomed in window
    axins = ax.inset_axes(inset_axes_pos[obj], projection=this_wcs)
    x1, x2, y1, y2 = int(pix_x-25), int(pix_x+25), int(pix_y-25), int(pix_y+25)
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    axins.imshow(subaru_image[0].data, cmap=plt.cm.seismic, vmin=-0.4, vmax=0.4, rasterized=True)

    # subregion of the original image
    ax.indicate_inset_zoom(axins)
    lon = axins.coords[0]
    lat = axins.coords[1]
    lon.set_major_formatter('dd:mm:ss')
    lat.set_major_formatter('dd:mm:ss.s')
    axins.tick_params(axis='x', bottom=True, top=False)
    axins.tick_params(axis='y', left=True, right=False)
    axins.set(xlabel=' ', ylabel=' ')


    ## ------------------ Plot slit --------------------##
    LEN = meta_pars['slitLen']
    WID = meta_pars['slitWidth']

    corners = get_slit_corners(slitRA, slitDec, slitLPA, LEN, WID)

    axins.plot(corners[:, 0], corners[:, 1], c='royalblue')
    axins.text(x=pix_x-24, y=pix_y+22, s=f'Slit Angle={slitLPA*180/np.pi:.0f}$\degree$', color='k', fontsize=8)


    ## ------------------ Plot predicted gamma t--------------------##
    ## Show direction of gammat
    g1 = stats[obj][1]['g1'].value
    g2 = stats[obj][1]['g2'].value

    alpha = np.pi - 0.5*np.arctan2(g2, g1) #- np.pi/2
    ax.quiver(pix_x, pix_y, -np.cos(alpha), -np.sin(alpha), angles='xy', color='royalblue', scale=20, width=0.001)

    ## ------------------ Plot predicted gamma t--------------------##
    path = '../../Data/CLASH/'
    g1_predict, g2_predict = utils.predcit_shear_from_map(RA, Dec, path=path)
    alpha = np.pi - 0.5*np.arctan2(-g2_predict, g1_predict)# - np.pi/2
    # gamma = (g1_predict**2 + g2_predict**2)**0.5
    # new_RA, new_Dec = RA + gamma*np.sin(2*beta), Dec + gamma*np.cos(2*beta)
    # new_pix_x, new_pix_y = this_wcs.wcs_world2pix(new_RA, new_Dec, 1)
    # axins.plot([pix_x, new_pix_x], [pix_y, new_pix_y], c='k')

    #axins.quiver(pix_x, pix_y, -g1_predict*np.cos(2*beta),  +g2_predict*np.sin(2*beta), angles='xy', scale=1)
    ax.quiver(pix_x, pix_y, -np.cos(alpha), -np.sin(alpha), angles='xy', scale=20, width=0.001)
    # ax.quiver(pix_x, pix_y, (a2261_x-pix_x)/1000, (a2261_y-pix_y)/1000, angles='xy', scale=5)

    axins.text(x=pix_x-24, y=pix_y+20, s=f'$\phi$ w.r.t cluster ={beta*180/np.pi:.0f}$\degree$', color='k', fontsize=8)


plt.savefig('diagnostic_plot_image.pdf', bbox_inches='tight')