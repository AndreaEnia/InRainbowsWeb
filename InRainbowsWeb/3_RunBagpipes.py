import os, sys
import bagpipes as pipes
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from _utils import IO_funcs as IO

galaxy_name = 'test_new'
# Read photometry .csv
df = pd.read_csv('Galaxies/{}/photometries_bagpipes.csv'.format(galaxy_name))

# Run bagpipes
def load_phot(ID):
    subdf = df[df['ID'] == float(ID)]
    subdf = subdf.drop(columns = ['ID', 'PIX_X', 'PIX_Y', 'RA', 'DEC'])
    phot = np.array(subdf).reshape(-1,2)
    return phot

filter_list = np.loadtxt('Filters/JWST_filters.txt', dtype='str')

delayed = {}                          # Tau model te^-(t/tau)
delayed["age"] = (0.001, 14.)         # Time since SF began: Gyr
delayed["tau"] = (0.01, 10.)          # Timescale of decrease: Gyr
delayed["metallicity"] = (0.005, 2.5) # 
delayed["massformed"] = (1., 12.5)    # log_10(M*/M_solar)

nebular = {}
nebular["logU"] = (-4, -2)            # Log_10 of the ionization parameter.

dust = {}                         
dust["type"] = "Calzetti"         
dust["Av"] = (0., 6.)                  # Prova no-dust (=0), sennò: (0., 6.)

fit_info = {}                           # The fit instructions dictionary
fit_info["redshift"] = 0.345           # Spec-z
#fit_info["redshift"] = (0., 15.)       # Unknown photo-z
fit_info["delayed"] = delayed
fit_info["dust"] = dust
fit_info["nebular"] = nebular

from joblib import Parallel, delayed

def fit_catalog_parallel(args):
    cell_ID, fi, fl = args
    if os.path.exists('pipes/plots/{0}/{0}_corner.pdf'.format(cell_ID)): return
    galaxy = pipes.galaxy(cell_ID, load_phot, phot_units = 'mujy', filt_list = fl, spectrum_exists = False)
    fit = pipes.fit(galaxy, fi, run = '{}'.format(cell_ID))
    try: fit.fit(verbose=False, n_live=512)
    except: return
    try: fit.plot_sfh_posterior()
    except: pass
    fit.plot_1d_posterior()
    fit.plot_corner()
    _ = fit.plot_spectrum_posterior()
    return

# Fit the cells in ||.
args_list = [(cell_ID, fit_info, filter_list) for cell_ID in df.ID.values]
n_jobs = 12
Parallel(n_jobs=n_jobs)(delayed(fit_catalog_parallel)(args) for args in args_list)

#######################################
# Generate final catalogue
# QUANTO SOTTO E' MOLTO BRUTTO E DEVE ESSERE
# MODIFICATO IN FUTURO PERCHE' OLTRE CHE FUNZIONALI
# LE COSE DEVONO ANCHE ESSERE BELLE DA VEDERE MINCHIA

import copy
this_ID = df.ID[0] # Safe exit
galaxy = pipes.galaxy(str(this_ID), load_phot, phot_units='mujy', \
                      filt_list = filter_list, spectrum_exists = False)
fit = pipes.fit(galaxy, {}, run = '{}'.format(this_ID))
fit.posterior.get_basic_quantities()
fit.posterior.get_advanced_quantities()

# Setup catalogue
s_vars = copy.copy(fit.fitted_model.params)
s_vars += ["stellar_mass", "formed_mass", "sfr", "ssfr", "nsfr", "mass_weighted_age", "tform", "tquench"]
cols = ["#ID"]
for var in s_vars: cols += [var + "_16", var + "_50", var + "_84"]
cols += ["input_redshift", "log_evidence", "log_evidence_err"]
cols += ["chisq_phot", "n_bands"]
cat = pd.DataFrame(np.zeros((df.ID.shape[0], len(cols))), columns=cols)
cat.loc[:, "#ID"] = df['ID'].values
cat.index = df.ID

def update_cat(ID, cat):
    galaxy = pipes.galaxy(str(ID), load_phot, phot_units='mujy', \
                          filt_list = filter_list, spectrum_exists = False)
    try: fit = pipes.fit(galaxy, {}, run = '{}'.format(ID))
    except:
        print('BIOPARCO')
        return cat.loc[this_ID] # Safe exit just in case
    fit.posterior.get_basic_quantities()
    fit.posterior.get_advanced_quantities()
    samples = fit.posterior.samples
    for v in s_vars:
        values = samples[v]
        cat.loc[ID, v + "_16"] = np.percentile(values, 16)
        cat.loc[ID, v + "_50"] = np.percentile(values, 50)
        cat.loc[ID, v + "_84"] = np.percentile(values, 84)   
    results = fit.results
    cat.loc[ID, "log_evidence"] = results["lnz"]
    cat.loc[ID, "log_evidence_err"] = results["lnz_err"]
    cat.loc[ID, "chisq_phot"] = np.min(samples["chisq_phot"])
    return cat.loc[ID]

from functools import partial
from concurrent.futures import ProcessPoolExecutor

# Process the results in ||.
with ProcessPoolExecutor(max_workers = 14) as executor:
    results = list(executor.map(partial(update_cat, cat=cat), df.ID.values))
    
results_df = pd.concat(results, axis = 1).T.reset_index(drop = True)
results_df = pd.concat([df, results_df], axis = 1)
results_df['PIX_X'] = results_df['PIX_X'].astype('int')
results_df['PIX_Y'] = results_df['PIX_Y'].astype('int')
results_df.to_csv('pipes/cats/{0}_final_results.csv'.format(galaxy_name), index = False)

from matplotlib import cm, rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
from matplotlib import cm, gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.axes as maxes
fits_base = IO.FitsUtils('Galaxies/{0}/f444w_SNR_POOLED.fits'.format(galaxy_name), 'f444w', 'sci')

def plot_quantity(fits_base, quantity, label, savepath, xmin, xmax, ymin, ymax, \
                  chisq_mask = False, chisq_map = None, chisq_threshold = 30, \
                  cmap = cm.inferno):
    fig = plt.figure(figsize=(10,10))
    ax = plt.subplot(111, projection = fits_base.wcs)
    ax.coords[0].set_axislabel('RA [deg]', size = 20), ax.coords[1].set_axislabel('DEC [deg]', size = 20)
    ax.coords[0].set_ticklabel(size=15), ax.coords[1].set_ticklabel(size=15)
    if chisq_mask:
        c_BAD, c_GOOD = np.where(chisq_map > chisq_threshold), np.where(chisq_map <= chisq_threshold)
        masked_quantity = np.copy(quantity)
        masked_quantity[c_BAD] = np.nan
        ax.imshow(quantity, origin = 'lower', interpolation = 'nearest', cmap = cm.Greys)
        im = ax.imshow(masked_quantity, origin = 'lower', interpolation = 'nearest', cmap = cmap)
    else: im = ax.imshow(quantity, origin = 'lower', interpolation = 'nearest', cmap = cmap)
    ax.axis('equal')
    if xmin != 0: ax.set_xlim(xmin, xmax), ax.set_ylim(ymin, ymax)
    else: pass
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", axes_class=maxes.Axes, pad=0.0)
    cax.tick_params(direction='in') 
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.outline.set_edgecolor('black')
    cax.xaxis.set_ticks_position('top')
    cax.tick_params(axis='both', which='major', length = 4.0, labelsize=15)
    cax.set_title(label, fontsize = 25, pad = 30)
    fig.savefig(savepath, bbox_inches = 'tight')
    
def save_fits(fits_base, quantity, label, code, savepath):
    from datetime import datetime
    from astropy.io import fits
    newhdr = fits_base.hdr
    newhdr['BUNIT'] = label
    newhdr['CODE'] = code
    newhdr['WHO'] = 'A. Enia, L. Scaloni'
    newhdr['WHEN'] = datetime.now().strftime("%d%b%Y %H:%M:%S")
    hdu = fits.PrimaryHDU(data = quantity, header = newhdr)
    hdu.writeto(savepath, overwrite=True)
    return

# Questo (plot + fits) DEVE diventare un unico loop
nx, ny = fits_base.hdr['NAXIS1'], fits_base.hdr['NAXIS2']
xmin, xmax = 0+15, nx-15 # CHISTU HAV'I A CANCIARI QUANNU
ymin, ymax = 0+15, ny-15 # HAVI A FARI 'U CATALOGO FINALE
chisq_threshold = 100

chisqmap = np.zeros((nx, ny))
chisqmap[chisqmap == 0] = np.nan
chisqmap[results_df.PIX_Y.values, results_df.PIX_X.values] = results_df.chisq_phot.values
plot_quantity(fits_base, chisqmap, r'$\chi^2$', 'pipes/{}_chisq.pdf'.format(galaxy_name), xmin, xmax, ymin, ymax)

f444w = np.zeros((nx, ny))
f444w[f444w == 0] = np.nan
f444w[results_df.PIX_Y.values, results_df.PIX_X.values] = results_df.f444w.values
plot_quantity(fits_base, f444w, r'Flux [$\mu$Jy]', 'pipes/{}_f444w_chisq.pdf'.format(galaxy_name), xmin, xmax, ymin, ymax, \
              chisq_mask = True, chisq_map = chisqmap, chisq_threshold = chisq_threshold)

#zmap = np.zeros((nx, ny))
#zmap[zmap == 0] = np.nan
#zmap[results_df.PIX_Y.values, results_df.PIX_X.values] = results_df.redshift_50.values
#plot_quantity(fits_base, zmap, r'$z_{\rm phot}$', 'pipes/{}_photoz.pdf'.format(galaxy_name), xmin, xmax, ymin, ymax, \
#chisq_mask = True, chisq_map = chisqmap, chisq_threshold = chisq_threshold)

massmap = np.zeros((nx, ny))
massmap[massmap == 0] = np.nan
massmap[results_df.PIX_Y.values, results_df.PIX_X.values] = results_df.formed_mass_50.values
plot_quantity(fits_base, massmap, r'$\log {\rm M}_* [M_{\odot}]$', 'pipes/{}_mass.pdf'.format(galaxy_name), xmin, xmax, ymin, ymax, \
              chisq_mask = True, chisq_map = chisqmap, chisq_threshold = chisq_threshold)

sfrmap = np.zeros((nx, ny))
sfrmap[sfrmap == 0] = np.nan
sfrmap[results_df.PIX_Y.values, results_df.PIX_X.values] = np.log10(results_df.sfr_50.values)
plot_quantity(fits_base, sfrmap, r'$\log {\rm SFR} [M_{\odot}/yr]$', 'pipes/{}_sfr.pdf'.format(galaxy_name), xmin, xmax, ymin, ymax, \
              chisq_mask = True, chisq_map = chisqmap, chisq_threshold = chisq_threshold)

avmap = np.zeros((nx, ny))
avmap[avmap == 0] = np.nan
avmap[results_df.PIX_Y.values, results_df.PIX_X.values] = results_df['dust:Av_50'].values
plot_quantity(fits_base, avmap, r'$A_V$ [mag]', 'pipes/{}_Av.pdf'.format(galaxy_name), xmin, xmax, ymin, ymax, \
              chisq_mask = True, chisq_map = chisqmap, chisq_threshold = chisq_threshold)

save_fits(fits_base, chisqmap, 'chi2', 'BAGPIPES', 'pipes/{}_chi2.fits'.format(galaxy_name))
#save_fits(fits_base, zmap, 'photo-z', 'BAGPIPES', 'pipes/{}_photoz.fits'.format(galaxy_name))
save_fits(fits_base, massmap, 'logMstar', 'BAGPIPES', 'pipes/{}_mass.fits'.format(galaxy_name))
save_fits(fits_base, sfrmap, 'logSFR', 'BAGPIPES', 'pipes/{}_sfr.fits'.format(galaxy_name))
save_fits(fits_base, avmap, 'Av', 'BAGPIPES', 'pipes/{}_Av.fits'.format(galaxy_name))