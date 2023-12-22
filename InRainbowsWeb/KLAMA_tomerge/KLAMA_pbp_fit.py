import os, sys, subprocess
import subprocess
import numpy as np
np.seterr(invalid='ignore')
import pandas as pd
from astropy.io import fits, ascii
from astropy import units as u
from astropy.constants import c as v_lux
from astropy.cosmology import Planck15 as cosmo
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from tqdm import trange, tqdm

import bagpipes as pipes
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

from math import pow
import AbdFunc

class FitsUtils:
    def __init__(self, signal_path, filtername, kind = 'sci'):
        import numpy as np
        from astropy.io import fits, ascii
        from astropy.wcs import WCS
        self.fits_path = signal_path
        self.fits_file = fits.open(signal_path)
        self.signal = fits.getdata(self.fits_path)
        self.hdr = fits.getheader(self.fits_path)
        self.wcs = WCS(self.hdr)
        self.filtername = filtername
        self.kind = kind
        
    def remove_nans(self):
        self.signal_with_nans = self.signal.copy()
        self.signal = np.nan_to_num(self.signal_with_nans)

    def get_pixel_scale(self):
        import numpy as np
        return (self.hdr['CD2_2']*u.deg).to('arcsec').value
    
    def get_wavelength(self):
        return Wvlghts_dictionary[self.bandname]

    def convert_wht_to_err(self):
        self.signal = np.sqrt(np.absolute(1/self.signal))
        if hasattr(self, 'signal_with_nans'): self.signal_with_nans = np.sqrt(np.absolute(1/self.signal_with_nans))
        if hasattr(self, 'signal_convolved'): self.signal_convolved = np.sqrt(np.absolute(1/self.signal_convolved))
        if hasattr(self, 'signal_convolved_with_nans'): self.signal_convolved_with_nans = np.sqrt(np.absolute(1/self.signal_convolved_with_nans))
        
    def degrade_to_worst(self, kernel_basepath, resize_kernel = False, save_convolved_fits = False):
        from astropy.convolution import convolve
        # Do not convolve if already f444w
        if self.filtername == 'f444w':
            self.signal_convolved = self.signal.copy()
            return
        # Load the convolution kernel
        sb, eb = 'JWST_NIRCam_{}'.format(self.filtername.upper()), 'JWST_NIRCam_F444W'
        try:
            self.kernel_path = kernel_basepath+'Kernel_HiRes_{0}_to_{1}.fits.gz'.format(sb, eb)
            self.kernel = fits.getdata(self.kernel_path)
        except: 
            self.kernel_path = kernel_basepath+'Kernel_HiRes_{0}_to_{1}.fits'.format(sb, eb)
            self.kernel = fits.getdata(self.kernel_path)
        # Eventually resize kernel
        if resize_kernel:
            from skimage.transform import resize
            kernel_2conv = resize(self.kernel, (631, 631), preserve_range = True)
        else: kernel_2conv = self.kernel.copy()
        # Square kernel for err (also square map) or variance map
        if self.kind == 'sci': signal_2conv = self.signal.copy()            
        elif self.kind == 'err': signal_2conv, kernel_2conv = self.signal**2, kernel_2conv**2
        elif self.kind == 'wht': signal_2conv, kernel_2conv = self.signal.copy(), kernel_2conv**2
        # Convolve, square root signal if error map
        sign_conv = convolve(signal_2conv, kernel = kernel_2conv, boundary = 'fill', preserve_nan = True)
        if self.kind == 'err': sign_conv = np.sqrt(sign_conv)
        self.signal_convolved = sign_conv.copy()
        # Store also the nan version of the signal
        if hasattr(self, 'signal_with_nans') == True:
            ok_nan = np.where(np.nan_to_num(self.signal_with_nans-1) == 0) # I know, can't do anything 'bout it
            sign_conv[ok_nan] = np.nan
            self.signal_convolved_with_nans = sign_conv
        # Eventually save the .fits if it takes too much time to convolve it every time
        if save_convolved_fits:
            newpath = temp.fits_path.split('.')
            newpath[0] += '_convolved'
            newpath = '.'.join(newpath)
            hdu = fits.PrimaryHDU(data = self.signal_convolved, header = self.hdr)
            hdu.writeto(newpath, overwrite=False)
            print('Convolved .fits saved to {}'.format(newpath))

working_bands = ['f090w', 'f150w', 'f200w', 'f277w', 'f356w', 'f444w']
working_wvl = ([0.9, 1.5, 2.0, 2.77, 3.56, 4.44]*u.um).to('Angstrom')

# Evaluate photometry
# PSF degradation
map_tuple, errormap_tuple = [], []
for band in tqdm(working_bands):
    temp = FitsUtils('Cutouts/{0}.fits'.format(band), band, 'sci')
    temp.remove_nans()
    temp.degrade_to_worst('Kernels/', resize_kernel = True, save_convolved_fits = False)
    map_tuple.append(temp)
    temp = FitsUtils('Cutouts/{0}_wht.fits'.format(band), band, 'wht')
    temp.remove_nans()
    temp.degrade_to_worst('Kernels/', resize_kernel = True, save_convolved_fits = False)
    temp.convert_wht_to_err()
    errormap_tuple.append(temp) 

pool_factor = 3

# Build Photometry dataframe on original pixel scale
signal_tuple = [m.signal_convolved for m in map_tuple]
datacube = np.stack(signal_tuple, axis=2) # In 10*nJy, apparently
datacube = ((10*datacube)*u.nJy).to('uJy').value

# Resample signal with a sum-pooling factor (here, 3)
new_shape = (int(datacube.shape[0]/pool_factor), int(datacube.shape[1]/pool_factor), datacube.shape[2])
view_shape = (new_shape[0], pool_factor, new_shape[1], pool_factor, datacube.shape[2])
strides = (datacube.strides[0]*pool_factor, datacube.strides[0], datacube.strides[1]*pool_factor, datacube.strides[1], datacube.strides[2])
view = np.lib.stride_tricks.as_strided(datacube, shape=view_shape, strides=strides)
datacube = np.sum(np.sum(view, axis=1), axis=2)

# Build error  dataframe on original pixel scale
error_tuple = [em.signal_convolved for em in errormap_tuple]
error_datacube = np.stack(error_tuple, axis=2) # In 10*nJy, apparently
error_datacube = ((10*error_datacube)*u.nJy).to('uJy').value

# Resample error with a SQUARE-SUM-pooling factor
new_shape = (int(error_datacube.shape[0]/pool_factor), int(error_datacube.shape[1]/pool_factor), error_datacube.shape[2])
view_shape = (new_shape[0], pool_factor, new_shape[1], pool_factor, error_datacube.shape[2])
strides = (error_datacube.strides[0]*pool_factor, error_datacube.strides[0], error_datacube.strides[1]*pool_factor, error_datacube.strides[1], error_datacube.strides[2])
view = np.lib.stride_tricks.as_strided(error_datacube, shape=view_shape, strides=strides)
error_datacube = np.sqrt(np.sum(np.sum(view**2, axis = 1), axis = 2))

# Merge dataframes
datacube = np.concatenate((datacube, error_datacube), axis = 2)

# Adjust hdr/wcs accordingly
newhdr = FitsUtils('Cutouts/f444w.fits', 'f444w').hdr.copy()
newhdr['NAXIS1'] = datacube.shape[0]
newhdr['NAXIS2'] = datacube.shape[1]
newhdr['CD1_1'] *= pool_factor
newhdr['CD2_2'] *= pool_factor
newhdr['CRPIX1'] = int(newhdr['CRPIX1']/pool_factor) + .5
newhdr['CRPIX2'] = int(newhdr['CRPIX2']/pool_factor) + .5 + 1
wcs = WCS(newhdr)

# Here I'm taking advantage of the fact that the maps have the same pixel scale and nx/ny, so instead of doing
# photometry on each position RA/DEC, I just stack everything togheter and then evaluate the RA/DEC in deg for
# each pixel. This gets later placed in a single dataframe with ID, RA, DEC and photometry.
x_indices, y_indices = np.meshgrid(np.arange(datacube.shape[0]), np.arange(datacube.shape[1]))
coords = pixel_to_skycoord(x_indices, y_indices, wcs)
cube_ra = np.stack([coord.ra.value for coord in coords], axis = 1)
cube_dec = np.stack([coord.dec.value for coord in coords], axis = 1)
datacube = np.concatenate((cube_ra[..., np.newaxis], cube_dec[..., np.newaxis], datacube), axis=2)
datacube = np.concatenate((x_indices[..., np.newaxis], y_indices[..., np.newaxis], datacube), axis=2)

# Generate a DataFrame with positions and photometry
column_names = ['PIX_X', 'PIX_Y', 'RA', 'DEC'] + working_bands + [w+'_err' for w in working_bands]
df = pd.DataFrame(np.reshape(datacube, (-1, datacube.shape[-1])), columns=column_names)
df = df.reset_index().rename(columns={'index': 'ID'})

# Take only the KLAMA sources, the two galaxies above and the bridge in between
mask_ra = [110.7025790, 110.7030433, 110.7021852, 110.702559]*u.deg
mask_dec = [-73.4846946, -73.48422, -73.4841961, -73.4844189]*u.deg
mark_radius = [0.75, 0.7, 0.7, 0.5]*u.arcsec
cond = np.logical_or.reduce([np.reshape(coords.separation(SkyCoord(cr, cd, frame = 'icrs')).arcsec, (-1, 1)) < rad.value \
                             for cr, cd, rad in zip(mask_ra, mask_dec, mark_radius)])
df = df[cond]

# Filter out rows where more than 3 values are upper limits
threshold = 3
df = df[(df[working_bands] < 0).sum(axis=1) <= threshold]
df = df.reset_index(drop = True)

## Add rms per pixel
#for i, band in enumerate(working_bands): df['{}_err'.format(band)] = np.sqrt(rms_per_band[i]**2 + df['{}_err'.format(band)]**2)
    
# Correct per dust extinction at galaxy position
Gal_EBV = AbdFunc.EBV_foreground_dust(SkyCoord(mask_ra[0], mask_dec[0], frame = 'icrs'))
Gal_dust_corr_factor = {}
for b, leff in zip(working_bands, working_wvl): Gal_dust_corr_factor[b] = pow(10.0, 0.4*AbdFunc.k_lmbd_Fitz1986_LMC(leff.value)*Gal_EBV)
for b in working_bands:
    df[b] *= Gal_dust_corr_factor[b]
    df[b+'_err'] *= Gal_dust_corr_factor[b]

# Order
ordered_columns = ['ID', 'PIX_X', 'PIX_Y', 'RA', 'DEC', 'f090w', 'f090w_err', 'f150w', 'f150w_err', \
                   'f200w', 'f200w_err', 'f277w', 'f277w_err', 'f356w', 'f356w_err', 'f444w', 'f444w_err']
df = df[ordered_columns]

# Run bagpipes
def load_phot(ID):
    subdf = df[df['ID'] == float(ID)]
    subdf = subdf.drop(columns = ['ID', 'PIX_X', 'PIX_Y', 'RA', 'DEC'])
    phot = np.array(subdf).reshape(-1,2)
    return phot
    
filter_list = np.loadtxt('Filters/JWST_filters.txt', dtype='str')

exponential = {}                          # Tau model e^-(t/tau)
exponential["age"] = (0.001, 14.)         # Time since SF began: Gyr
exponential["tau"] = (0.01, 10.)          # Timescale of decrease: Gyr
exponential["metallicity"] = (0.005, 2.5) # 
exponential["massformed"] = (1., 12.5)    # log_10(M*/M_solar)

nebular = {}
nebular["logU"] = (-4, -2)                # Log_10 of the ionization parameter.

dust = {}                         
dust["type"] = "Calzetti"         
dust["Av"] = (0., 6.)

fit_info = {}                             # The fit instructions dictionary
fit_info["redshift"] = (0., 15.)          # Unknown photo-z
fit_info["exponential"] = exponential
fit_info["dust"] = dust
fit_info["nebular"] = nebular

from joblib import Parallel, delayed

def fit_catalog_parallel(args):
    cell_ID, fi, fl = args
    if os.path.exists('pipes/plots/{0}/{0}_corner.pdf'.format(cell_ID)): return
    galaxy = pipes.galaxy(cell_ID, load_phot, phot_units = 'mujy', filt_list = fl, spectrum_exists = False)
    fit = pipes.fit(galaxy, fi, run = '{}'.format(cell_ID))
    try: fit.fit(verbose=False, n_live=1024)
    except: return
    try: fit.plot_sfh_posterior()
    except: pass
    fit.plot_1d_posterior()
    fit.plot_corner()
    _ = fit.plot_spectrum_posterior()
    return

# Fit the cells in ||.
args_list = [(cell_ID, fit_info, filter_list) for cell_ID in df.ID.values]
n_jobs = 14
Parallel(n_jobs=n_jobs)(delayed(fit_catalog_parallel)(args) for args in args_list)

# Store
df.to_csv('pipes/cats/photometry.csv', index = False)

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
        return cat.loc[5000] # Safe exit just in case
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
results_df.to_csv('pipes/cats/final_results.csv', index = False)

from matplotlib import cm
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
from matplotlib import cm, gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.axes as maxes
fits_base = FitsUtils('Cutouts/f444w.fits', 'f444w', 'sci')

def plot_quantity(quantity, hdr, label, savepath, xmin, xmax, ymin, ymax, \
                  chisq_mask = False, chisq_map = None, chisq_threshold = 30, \
                  cmap = cm.inferno):
    fig = plt.figure(figsize=(10,10))
    ax = plt.subplot(111, projection = WCS(hdr))
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
    return

xmin, xmax = 14, 40
ymin, ymax = 22, 48

chisqmap = np.zeros((datacube.shape[0], datacube.shape[1]))
chisqmap[chisqmap == 0] = np.nan
chisqmap[results_df.PIX_Y.values, results_df.PIX_X.values] = results_df.chisq_phot.values
plot_quantity(chisqmap, newhdr, r'$\chi^2$', 'pipes/KLAMA_chisq.pdf', xmin, xmax, ymin, ymax)

red_chisqmap = np.zeros((datacube.shape[0], datacube.shape[1]))
red_chisqmap[red_chisqmap == 0] = np.nan
red_chisqmap[results_df.PIX_Y.values, results_df.PIX_X.values] = np.sqrt(results_df.chisq_phot.values)/len(working_bands)
plot_quantity(red_chisqmap, newhdr, r'$\chi^2_r$', 'pipes/KLAMA_redchisq.pdf', xmin, xmax, ymin, ymax)

chisq_threshold = 10
f444w = np.zeros((datacube.shape[0], datacube.shape[1]))
f444w[f444w == 0] = np.nan
f444w[results_df.PIX_Y.values, results_df.PIX_X.values] = results_df.f444w.values
plot_quantity(f444w, newhdr, r'$Flux [10*nJy]$', 'pipes/KLAMA_f444w_chisq.pdf', xmin, xmax, ymin, ymax, \
              chisq_mask = True, chisq_map = red_chisqmap, chisq_threshold = chisq_threshold)

zmap = np.zeros((datacube.shape[0], datacube.shape[1]))
zmap[zmap == 0] = np.nan
zmap[results_df.PIX_Y.values, results_df.PIX_X.values] = results_df.redshift_50.values
plot_quantity(zmap, newhdr, r'$z_{\rm phot}$', 'pipes/KLAMA_photoz.pdf', xmin, xmax, ymin, ymax, \
              chisq_mask = True, chisq_map = red_chisqmap, chisq_threshold = chisq_threshold)

massmap = np.zeros((datacube.shape[0], datacube.shape[1]))
massmap[massmap == 0] = np.nan
massmap[results_df.PIX_Y.values, results_df.PIX_X.values] = results_df.formed_mass_50.values
plot_quantity(massmap, newhdr, r'$\log {\rm M}_* [M_{\odot}]$', 'pipes/KLAMA_mass.pdf', xmin, xmax, ymin, ymax, \
              chisq_mask = True, chisq_map = red_chisqmap, chisq_threshold = chisq_threshold)

sfrmap = np.zeros((datacube.shape[0], datacube.shape[1]))
sfrmap[sfrmap == 0] = np.nan
sfrmap[results_df.PIX_Y.values, results_df.PIX_X.values] = np.log10(results_df.sfr_50.values)
plot_quantity(sfrmap, newhdr, r'$\log {\rm SFR} [M_{\odot}/yr]$', 'pipes/KLAMA_sfr.pdf', xmin, xmax, ymin, ymax, \
              chisq_mask = True, chisq_map = red_chisqmap, chisq_threshold = chisq_threshold)

avmap = np.zeros((datacube.shape[0], datacube.shape[1]))
avmap[avmap == 0] = np.nan
avmap[results_df.PIX_Y.values, results_df.PIX_X.values] = np.log10(results_df['dust:Av_50'].values)
plot_quantity(avmap, newhdr, r'$A_V$ [mag]', 'pipes/KLAMA_Av.pdf', xmin, xmax, ymin, ymax, \
              chisq_mask = True, chisq_map = red_chisqmap, chisq_threshold = chisq_threshold)

def save_fits(quantity, quantity_hdr, label, code, savepath, who = 'A. Enia'):
    from datetime import datetime
    from astropy.io import fits
    quantity_hdr['BUNIT'] = label
    quantity_hdr['CODE'] = code
    quantity_hdr['WHO'] =  who
    quantity_hdr['WHEN'] = datetime.now().strftime("%d%b%Y %H:%M:%S")
    hdu = fits.PrimaryHDU(data = quantity, header = quantity_hdr)
    hdu.writeto(savepath, overwrite=True)
    return

save_fits(zmap, newhdr, 'photo-z', 'BAGPIPES', 'pipes/KLAMA_photoz.fits')
save_fits(massmap, newhdr, 'logMstar', 'BAGPIPES', 'pipes/KLAMA_mass.fits')
save_fits(sfrmap, newhdr, 'logSFR', 'BAGPIPES', 'pipes/KLAMA_sfr.fits')
save_fits(avmap, newhdr, 'Av', 'BAGPIPES', 'pipes/KLAMA_Av.fits')