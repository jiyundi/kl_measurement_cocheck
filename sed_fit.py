import argparse
import numpy as np
import matplotlib.pyplot as plt
import time, sys

import dynesty

from prospect.fitting import fit_model, lnprobfn
from prospect.fitting.fitting import run_emcee
from prospect.io import write_results as writer

from prospect.models.templates import TemplateLibrary, adjust_dirichlet_agebins
from prospect.models import priors, sedmodel
from sedpy.observate import load_filters
from prospect.sources import CSPSpecBasis, FastStepBasis
from prospect.utils.obsutils import fix_obs

from klm.observations import Deimos
import klm.photometry as photometry

import ipdb

parser = argparse.ArgumentParser()
parser.add_argument('--mask', type=str)
parser.add_argument('--slit', type=str)
parser.add_argument('--results_type', default='emcee')
parser.add_argument('--imf_type', default=1, type=int)
parser.add_argument('--dust_type', default=4, type=int)
parser.add_argument('--useSubaru', default=1, type=int)
parser.add_argument('--useHST', default=0, type=int)
parser.add_argument('--useInfrared', default=0, type=int)
parser.add_argument('--Infraredband', default='both', type=str)
parser.add_argument('--parametric', default=1, type=int)

def init_dust_model(model_params, dust_type):
    #model_params['dust1'] = dict(name= "dust1", N=1, isfree=True, init=0.2, prior=priors.ClippedNormal(mini=0, maxi=2, mean=1, sigma=0.3),
    #init_disp=0.4, disp_floor=0.4)
    model_params['dust2'] = dict(name= "dust2", N=1, isfree=True, init=0.2, prior=priors.ClippedNormal(mini=0, maxi=4, mean=0.3, sigma=1),
    init_disp=0.4, disp_floor=0.4)

    # Power law
    if dust_type == 0:
        model_params['dust_index'] = dict(name= "dust_index", N=1, isfree=True, init=-0.7, prior=priors.TopHat(mini=-3, maxi=3),
    init_disp=0.2, disp_floor=0.2)
        model_params['dust1_index'] = dict(name= "dust1_index", N=1, isfree=True, init=-0.7, prior=priors.TopHat(mini=-3, maxi=3),
    init_disp=0.2, disp_floor=0.2)

    # Milky way extinction
    if dust_type == 1:
        model_params['mwr'] = dict(name= "mwr", N=1, isfree=True, init=3.1, prior=priors.TopHat(mini=0.1, maxi=6),
    init_disp=0.4, disp_floor=0.4)
        model_params['uvb'] = dict(name= "uvb", N=1, isfree=True, init=3.1, prior=priors.TopHat(mini=0.1, maxi=6),
    init_disp=0.4, disp_floor=0.4)

    # Calzetti 2000: dust1, dust2 already free parameters
    if dust_type == 2:
        model_params['dust1'] = dict(name= "dust1", N=1, isfree=False, init=0.0)

    # Kriek & Conroy 2013
    if dust_type == 4:
        model_params['dust_index'] = dict(name= "dust_index", N=1, isfree=True, init=-0.7, prior=priors.TopHat(mini=-3, maxi=1),
    init_disp=0.2, disp_floor=0.2)
        # model_params['dust1_index'] = dict(name= "dust1_index", N=1, isfree=True, init=-0.7, prior=priors.TopHat(mini=-5, maxi=1),
    # init_disp=0.2, disp_floor=0.2)

    ## Select dust model
    model_params['dust_type']['init'] = dust_type

    return model_params

def build_model_nonparametric(object_redshift=0.0, fixed_metallicity=None, add_duste=True,
                add_neb=False, **kwargs):
    model_params = TemplateLibrary['dirichlet_sfh']

    nbins_sfh = 10
    age_bins = np.linspace(np.log10(300e6), 13.1, nbins_sfh-1)
    a = np.array([0, np.log10(100e6)])
    age_bins = np.append(a, age_bins)

    model_params = adjust_dirichlet_agebins(model_params, age_bins)
    model_params = init_dust_model(model_params, kwargs['dust_type'])
    # Adjust model initial values (only important for optimization or emcee)
    model_params["zred"]['init'] = object_redshift

    # adjust priors
    model_params["logzsol"] = dict(N=1, isFree=True, init=-0.3, prior=priors.LogUniform(mini=-2, maxi=0.19),
                                            init_disp=0.01, disp_floor=0.1)
    model_params['z_fraction'] = priors.Beta(alpha=np.full(nbins_sfh-1, 0.7), beta=np.ones(nbins_sfh-1), mini=0.0, maxi=1.0)
    model_params["total_mass"] = dict(N=1, isFree=True, init=1e11, prior=priors.LogUniform(mini=1e6, maxi=1e13),
                                        init_disp=5e10, disp_floor=5e9)

    model_params['imf_type']['init'] = kwargs['imf_type']

    if add_duste:
        # Add dust emission (with fixed dust SED parameters)
        model_params.update(TemplateLibrary["dust_emission"])

    if add_neb:
        # Add nebular emission (with fixed parameters)
        model_params.update(TemplateLibrary["nebular"])

    # Now instantiate the model using this new dictionary of parameter specifications
    model = sedmodel.SedModel(model_params)

    return model


def build_model(object_redshift=0.0, fixed_metallicity=None, add_duste=False,
                add_neb=False, **kwargs):
    """Construct a model.  This method defines a number of parameter
    specification dictionaries and uses them to initialize a
    `models.sedmodel.SedModel` object.

    :param object_redshift:
        If given, given the model redshift to this value.

    :param add_dust: (optional, default: False)
        Switch to add (fixed) parameters relevant for dust emission.

    :param add_neb: (optional, default: False)
        Switch to add (fixed) parameters relevant for nebular emission, and
        turn nebular emission on.
    """
    model_params = TemplateLibrary["parametric_sfh"]
    model_params = init_dust_model(model_params, kwargs['dust_type'])

    # Adjust model initial values (only important for optimization or emcee)
    model_params["tage"]["init"] = 13.

    # If we are going to be using emcee, it is useful to provide an
    # initial scale for the cloud of walkers (the default is 0.1)
    # For dynesty these can be skipped
    model_params["tau"]["init_disp"] = 3.0
    model_params["tage"]["init_disp"] = 5.0
    model_params["tage"]["disp_floor"] = 2.0

    # adjust priors
    model_params["logzsol"] = dict(N=1, isfree=True, init=-0.3, prior=priors.Uniform(mini=-2, maxi=0.19))
    model_params["tau"]= dict( N=1, isfree=True, init=5.3, prior=priors.LogUniform(mini=1e-1, maxi=10), disp_floor=0.5)
    model_params["mass"]= dict(N=1, isfree=True, init=1e10, prior=priors.Uniform(mini=1e7, maxi=10**12.5), disp_floor=5e6, units="mstar")


    ## Redshift
    model_params["zred"]['isfree'] = False
    model_params["zred"]['init'] = object_redshift

    model_params['imf_type']['init'] = kwargs['imf_type']

    if add_duste:
        # Add dust emission (with fixed dust SED parameters)
        model_params.update(TemplateLibrary["dust_emission"])

    if add_neb:
        # Add nebular emission (with fixed parameters)
        model_params.update(TemplateLibrary["nebular"])

    # Now instantiate the model using this new dictionary of parameter specifications
    model = sedmodel.SedModel(model_params)

    return model


def build_obs(phot, phot_error, is_maggies=False, luminosity_distance=None, **kwargs):
    filternames = filtersets
    # And here we loop over the magnitude columns

    # And since these are absolute mags, we can shift to any distance.
    if luminosity_distance is not None:
        dm = 25 + 5 * np.log10(luminosity_distance)
        mags += dm

    # Build output dictionary.
    obs = {}
    # This is a list of sedpy filter objects.    See the
    # sedpy.observate.load_filters command for more details on its syntax.
    obs['filters'] = load_filters(filternames)
    # This is a list of maggies, converted from mags.  It should have the same
    # order as `filters` above.
    obs['maggies'] = np.squeeze(10**(-phot/2.5))
    # HACK.  You should use real flux uncertainties
    obs['maggies_unc'] = (np.array(phot_error)*obs['maggies']*np.log(10)/2.5)

    if is_maggies:
        obs['maggies'] = mags
        obs['maggies_unc'] = error

    obs["phot_wave"] = np.array([f.wave_effective for f in obs["filters"]])

    # Here we mask out any NaNs or infs
    obs['phot_mask'] = np.isfinite(np.squeeze(mags))
    # We have no spectrum.
    obs['wavelength'] = None
    obs['spectrum'] = None


    # This ensures all required keys are present and adds some extra useful info
    obs = fix_obs(obs)

    return obs

# --------------
# SPS Object
# --------------

def build_sps(zcontinuous=1, **extras):
    """
    :param zcontinuous:
        A vlue of 1 insures that we use interpolation between SSPs to
        have a continuous metallicity parameter (`logzsol`)
        See python-FSPS documentation for details
    """

    if extras['model_parametric'] == 1:
        sps = CSPSpecBasis(zcontinuous=zcontinuous, add_stellar_remnants=False)

    elif extras['model_parametric'] == 0:
        sps = FastStepBasis(zcontinuous=zcontinuous, add_stellar_remnants=False)

    return sps


# -----------------
# Noise Model
# ------------------

def build_noise(**extras):
    return None, None

# -----------
# Everything
# ------------

def build_all(**kwargs):

    if kwargs['model_parametric'] == 1:
        return (build_obs(**kwargs), build_model(**kwargs),
                build_sps(**kwargs), build_noise(**kwargs))

    elif kwargs['model_parametric'] == 0:
        return (build_obs(**kwargs), build_model_nonparametric(**kwargs),
                build_sps(**kwargs), build_noise(**kwargs))

if __name__ == '__main__':
    args = parser.parse_args()
    imf_type = args.imf_type
    dust_type = args.dust_type

    mask = args.mask
    slit = args.slit

    deimos = Deimos('/xdisk/timeifler/pranjalrs/KL_data/')
    RA, DEC = deimos.getCoords_from_slit_name(mask=mask, slit_name=slit)

    all_mags = []
    all_error = []
    filtersets = []

    if args.useSubaru == 1:
        filters, mag, error = photometry.get_photometry_Subaru(RA, DEC, '/xdisk/timeifler/pranjalrs/KL_data/')
        filtersets += filters
        all_mags += mag
        all_error += error

    if args.useHST == 1:
        filters, mag, error = photometry.get_photometry_HST(RA, DEC)
        filtersets += filters
        all_mags += mag
        all_error += error


    if args.useInfrared == 1:
        filters, mag, error = photometry.get_photometry_infrared(mask, slit, args.Infraredband)
        ## Hack
        if mask=='a2261b' and slit=='0089':
            mag = [16.617, 16.495, 11.819]
            error = [0.065, 0.174, 0.259]

        elif mask=='a2261b' and slit=='0079':
            if args.Infraredband=='WISE':
                mag = [16.09, 16.431, 12.462]
                error = [0.045, 0.228, 0.369]

            elif args.Infraredband=='both':
                mag[-3:] = [16.09, 16.431, 12.462]
                error[-3:] = [0.045, 0.228, 0.369]

        #else:
        #    raise NotImplementedError('WISE magnitudes for object not specified')

        filtersets += filters
        all_mags += mag
        all_error += error

    run_params = {}
    ### Hack for using Legacy Survey Imaging
    bandpass_dict = {'g': 'decam_g', 'r': 'decam_r', 'i': 'decam_i', 'z': 'decam_z', 'w1': 'wise_w1', 'w2': 'wise_w2', 'w3': 'wise_w3', 'w4': 'wise_w4'}
    legacy_survey_data = np.loadtxt(f'../data/LS_{mask}_{slit}.txt', skiprows=1, delimiter=',')
    bands = ['g', 'r', 'i', 'z', 'w1', 'w2', 'w3', 'w4']
    fluxes = legacy_survey_data[2:2+len(bands)]
    ivar = legacy_survey_data[2+len(bands):]

    all_mags, all_errors = [], []
    filtersets = []
    for i, band in bands:
        # Fluxes are in nanomaggies
        if ivar[i] != 0:
            all_mags.append(fluxes[i] * 1e-9)
            all_errors.append(1/np.sqrt(ivar[i]) * 1e-9)
            filtersets.append(bandpass_dict[band])

    run_params['is_maggies'] = True
    t


    redshift = deimos.get_redshift_from_slit_name(mask, slit)


    run_params['mags'] = np.array(all_mags)
    run_params['error'] = np.array(all_error)
    run_params['object_redshift'] = redshift
    run_params['add_neb'] = False
    run_params["zcontinuous"] = 1
    run_params['model_parametric'] = args.parametric
    run_params['imf_type'] = imf_type  # Chabrier: 1, Kroupa: 2 https://github.com/bd-j/prospector/issues/166#issuecomment-611085641
    run_params['dust_type'] = dust_type # Calzetti: 2, Kriek & Conroy: 4

    ## Build everything for MCMC
    obs, model, sps, noise = build_all(**run_params)

    ####### Sampling #########
    results_type = args.results_type


    if results_type == 'emcee':
        run_params["optimize"] = False
        run_params["emcee"] = True
        run_params["dynesty"] = False
        # Number of emcee walkers
        run_params["nwalkers"] = 30
        # Number of iterations of the MCMC sampling
        run_params["niter"] = 4000
        # Number of iterations in each round of burn-in
        # After each round, the walkers are reinitialized based on the
        # locations of the highest probablity half of the walkers.
        run_params["nburn"] = [1300, 1300, 1300]


    if results_type == 'dynesty':
        run_params["dynesty"] = True
        run_params["optmization"] = False
        run_params["emcee"] = False
        run_params["nested_method"] = "rwalk"
        run_params["nlive_init"] = 400
        run_params["nlive_batch"] = 200
        run_params["nested_dlogz_init"] = 0.05
        run_params["nested_posterior_thresh"] = 0.05
        run_params["nested_maxcall"] = int(1e7)

    print(run_params)

    # Set up MPI communication
    try:
        import mpi4py
        from mpi4py import MPI
        from schwimmbad import MPIPool

        mpi4py.rc.threads = False
        mpi4py.rc.recv_mprobe = False

        comm = MPI.COMM_WORLD
        size = comm.Get_size()

        withmpi = comm.Get_size() > 1
    except ImportError:
        print('Failed to start MPI; are mpi4py and schwimmbad installed? Proceeding without MPI.')
        withmpi = False

    # Evaluate SPS over logzsol grid in order to get necessary data in cache/memory
    # for each MPI process. Otherwise, you risk creating a lag between the MPI tasks
    # caching data depending which can slow down the parallelization
    if (withmpi) & ('logzsol' in model.free_params):
        dummy_obs = dict(filters=None, wavelength=None)

        logzsol_prior = model.config_dict["logzsol"]['prior']
        lo, hi = logzsol_prior.range
        logzsol_grid = np.around(np.arange(lo, hi, step=0.1), decimals=2)

        sps.update(**model.params)  # make sure we are caching the correct IMF / SFH / etc
        for logzsol in logzsol_grid:
            model.params["logzsol"] = np.array([logzsol])
            _ = model.predict(model.theta, obs=dummy_obs, sps=sps)


    # ensure that each processor runs its own version of FSPS
    # this ensures no cross-over memory usage
    from functools import partial
    lnprobfn_fixed = partial(lnprobfn, sps=sps)

    if withmpi:
        run_params["using_mpi"] = True
        with MPIPool() as pool:

            # The dependent processes will run up to this point in the code
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
            nprocs = pool.size
            print('Running MCMC now...')
            print(model)
            # The parent process will oversee the fitting
            output = fit_model(obs, model, sps, noise, pool=pool, queue_size=nprocs, lnprobfn=lnprobfn_fixed, **run_params)
    else:
        # without MPI we don't pass the pool
        print('Running MCMC now...')
        print(model)
        output = fit_model(obs, model, sps, noise, lnprobfn=lnprobfn_fixed, **run_params)

    # Set up an output file and write
    if run_params['emcee'] is True:
        sampler = 'emcee'

    elif run_params['dynesty'] is True:
        sampler = 'dynesty'

    if args.useSubaru + args.useHST + args.useInfrared == 2:
        base_path = f'/xdisk/timeifler/pranjalrs/sed_fit'

    elif args.useSubaru == 1:
        base_path = f'/xdisk/timeifler/pranjalrs/sed_fit_Subaru'

    elif args.useHST == 1:
        base_path = f'/xdisk/timeifler/pranjalrs/sed_fit_HST'

    elif args.useInfrared == 1:
        base_path = f'/xdisk/timeifler/pranjalrs/sed_fit_Infrared'
        if args.Infraredband!='both':
            base_path = f'/xdisk/timeifler/pranjalrs/sed_fit_'+args.Infraredband

    if run_params['model_parametric'] == 1:
        hfile = f"{base_path}/{mask}_{slit}_parametric_imf{imf_type}_dust{dust_type}.h5"

    elif run_params['model_parametric'] == 0:
        hfile = f"{base_path}/{mask}_{slit}_imf{imf_type}_dust{dust_type}.h5"

    writer.write_hdf5(hfile, run_params, model, obs,
                    output["sampling"][0], output["optimization"][0],
                    tsample=output["sampling"][1],
                    toptimize=output["optimization"][1],
                    sps=sps)
    try:
        hfile.close()
    except(AttributeError):
        pass
