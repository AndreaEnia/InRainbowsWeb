import os, sys, subprocess
import subprocess
import numpy as np
np.seterr(invalid='ignore')
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from astropy.io import fits, ascii
from astropy import units as u
from astropy.constants import c as v_lux
from astropy.cosmology import Planck15 as cosmo
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from tqdm import trange, tqdm

# My scripts
import config
from _utils import IO_funcs as IO
from _utils import AbdFunc
from math import pow

space = '       '

def reduce_maps(galaxy_name, working_bands, kernel_path = config.paths_dict['kernels']):
    # PSF degradation
    print('PSF degradation.')
    map_tuple, errormap_tuple = [], []
    for band in working_bands:
        print('{0}{1} band'.format(space, band))
        temp = IO.FitsUtils('Galaxies/{0}/{1}_sci.fits'.format(galaxy_name, band), band, 'sci')
        temp.remove_nans()
        temp.convert_MJysr_to_uJy(8.4545E-14) # !!!!! #
        temp.degrade_to_worst(kernel_path)
        map_tuple.append(temp)
        temp = IO.FitsUtils('Galaxies/{0}/{1}_err.fits'.format(galaxy_name, band), band, 'err')
        temp.remove_nans()
        temp.convert_MJysr_to_uJy(8.4545E-14) # !!!!! #
        temp.degrade_to_worst(kernel_path)
        errormap_tuple.append(temp)
    print()
    return map_tuple, errormap_tuple

def segmentation_map(map_tuple, errormap_tuple, \
                     thresh = 3, minarea = 500, deblend_nthresh = 32, deblend_cont = 0.005,
                     plot_segm = True):
    import sep
    import matplotlib.pyplot as plt
    
    list_segm_map = []
    for _map, _err in zip(map_tuple, errormap_tuple):
        data_img = _map.signal.copy().byteswap(inplace=True).newbyteorder()
        data_err = _err.signal.copy().byteswap(inplace=True).newbyteorder()
        _, segm_map = sep.extract(data = data_img, thresh = thresh, err = np.median(data_err), \
                                  minarea = minarea, deblend_nthresh = deblend_nthresh, deblend_cont = deblend_cont, segmentation_map = True)
        segm_map[segm_map >= 1] = 1
        list_segm_map.append(segm_map)
    # Put toghether all the segmentation maps
    segm_map = np.zeros((data_img.shape[0], data_img.shape[1]))
    segm_map[np.sum(list_segm_map, axis = 0) > 0] = 1
    # Eventually plot the results
    if plot_segm:
        plt.figure(figsize=(12,10))
        plt.subplot(121, projection = map_tuple[-1].wcs)
        plt.imshow(map_tuple[-1].signal, origin = 'lower', interpolation = 'nearest')
        plt.contour(segm_map)
    return segm_map

def build_photometry(galaxy_name, map_tuple, errormap_tuple, working_bands, \
                     is_segmented = False, segmentation_map = False, masking = False, mask_ra = False, mask_dec = False, mask_radius = False, \
                     ulim_thresh = 3, pool_factor = 3, store = True):
    
    # Build Photometry dataframe on original pixel scale
    try: signal_tuple = [m.signal_convolved_with_nans for m in map_tuple]
    except: signal_tuple = [m.signal_convolved for m in map_tuple]
    datacube = np.stack(signal_tuple, axis=2) # Already in uJy, but recheck the conversion bw MJy/sr and uJy/px
    # Resample signal with a sum-pooling factor (default, 3)
    new_shape = (int(datacube.shape[0]/pool_factor), int(datacube.shape[1]/pool_factor), datacube.shape[2])
    view_shape = (new_shape[0], pool_factor, new_shape[1], pool_factor, datacube.shape[2])
    strides = (datacube.strides[0]*pool_factor, datacube.strides[0], datacube.strides[1]*pool_factor, datacube.strides[1], datacube.strides[2])
    view = np.lib.stride_tricks.as_strided(datacube, shape=view_shape, strides=strides)
    datacube = np.sum(np.sum(view, axis=1), axis=2)
    
    # Build error  dataframe on original pixel scale
    try: error_tuple = [em.signal_convolved_with_nans for em in errormap_tuple]
    except: error_tuple = [em.signal_convolved for em in errormap_tuple]
    error_datacube = np.stack(error_tuple, axis=2) # Already in uJy, but recheck the conversion bw MJy/sr and uJy/px
    # Resample error with a SQUARE-SUM-pooling factor (same as above)
    new_shape = (int(error_datacube.shape[0]/pool_factor), int(error_datacube.shape[1]/pool_factor), error_datacube.shape[2])
    view_shape = (new_shape[0], pool_factor, new_shape[1], pool_factor, error_datacube.shape[2])
    strides = (error_datacube.strides[0]*pool_factor, error_datacube.strides[0], error_datacube.strides[1]*pool_factor, error_datacube.strides[1], error_datacube.strides[2])
    view = np.lib.stride_tricks.as_strided(error_datacube, shape=view_shape, strides=strides)
    error_datacube = np.sqrt(np.sum(np.sum(view**2, axis = 1), axis = 2))
    
    # Merge dataframes
    datacube = np.concatenate((datacube, error_datacube), axis = 2)

    # Adjust hdr/wcs accordingly
    newhdr = map_tuple[-1].hdr.copy() # Copy the f444w header
    newhdr['NAXIS1'] = datacube.shape[0]
    newhdr['NAXIS2'] = datacube.shape[1]
    newhdr['CDELT1'] *= pool_factor
    newhdr['CDELT2'] *= pool_factor
    newhdr['CRPIX1'] = int(newhdr['CRPIX1']/pool_factor) + .5
    newhdr['CRPIX2'] = int(newhdr['CRPIX2']/pool_factor) + .5 + 1
    newwcs = WCS(newhdr)
        
    # Resample segmentation map accordingly
    if is_segmented:
        new_shape = (int(segmentation_map.shape[0]/pool_factor), int(segmentation_map.shape[1]/pool_factor))
        view_shape = (new_shape[0], pool_factor, new_shape[1], pool_factor)
        strides = (segmentation_map.strides[0]*pool_factor, segmentation_map.strides[0], segmentation_map.strides[1]*pool_factor, segmentation_map.strides[1])
        view = np.lib.stride_tricks.as_strided(segmentation_map, shape=view_shape, strides=strides)
        segmentation_map = np.sum(np.sum(view, axis = 1), axis = 2)
        segmentation_map[segmentation_map > 1] = 1
    else: segmentation_map = np.zeros((datacube.shape[0], datacube.shape[1])) + 1
    datacube = np.concatenate((datacube, segmentation_map[..., np.newaxis]), axis = 2)

    # Here I'm taking advantage of the fact that the maps have the same pixel scale and nx/ny, so instead of doing
    # photometry on each position RA/DEC, I just stack everything togheter and then evaluate the RA/DEC in deg for
    # each pixel. This gets later placed in a single dataframe with ID, RA, DEC and photometry.
    x_indices, y_indices = np.meshgrid(np.arange(datacube.shape[0]), np.arange(datacube.shape[1]))
    coords = pixel_to_skycoord(x_indices, y_indices, newwcs)
    cube_ra = np.stack([coord.ra.value for coord in coords], axis = 1)
    cube_dec = np.stack([coord.dec.value for coord in coords], axis = 1)
    datacube = np.concatenate((cube_ra[..., np.newaxis], cube_dec[..., np.newaxis], datacube), axis=2)
    datacube = np.concatenate((x_indices[..., np.newaxis], y_indices[..., np.newaxis], datacube), axis=2)

    # Generate a DataFrame with positions and photometry
    column_names = ['PIX_X', 'PIX_Y', 'RA', 'DEC'] + working_bands + [w+'_err' for w in working_bands] + ['SEGM']
    df = pd.DataFrame(np.reshape(datacube, (-1, datacube.shape[-1])), columns=column_names)
    df = df.reset_index().rename(columns={'index': 'ID'})
    
    # Mask the source, discard other pixels
    if masking == True:
        center_mask = SkyCoord(mask_ra, mask_dec, frame = 'icrs')
        df['sep'] = np.reshape(coords.separation(center_mask).arcsec, (-1, 1))
        cond_sep = df['sep'] < mask_radius.value
    else: cond_sep = numpy.full((1, len(df)), True)
    cond_segm = df['SEGM'] == 1
    df = df[np.logical_and(cond_sep, cond_segm)]
    
    # Filter out rows where more than threshold values are upper limits
    df = df[(df[working_bands] < 0).sum(axis=1) <= ulim_thresh]
    df = df.reset_index(drop = True)
    
    # Correct per dust extinction at galaxy position
    Gal_EBV = AbdFunc.EBV_foreground_dust(center_mask)
    Gal_dust_corr_factor = {}
    for b in working_bands:
        leff = (config.band_wvl_association[b]*u.um).to('angstrom').value
        corr = pow(10.0, 0.4*AbdFunc.k_lmbd_Fitz1986_LMC(leff)*Gal_EBV)
        df[b] *= corr
        df[b+'_err'] *= corr
    
    # Order
    ordered_columns = ['ID', 'PIX_X', 'PIX_Y', 'RA', 'DEC', \
                       'f090w', 'f090w_err', 'f150w', 'f150w_err', 'f200w', 'f200w_err', \
                       'f277w', 'f277w_err', 'f356w', 'f356w_err', 'f410m', 'f410m_err', \
                       'f444w', 'f444w_err']
    df = df[ordered_columns]
    
    # Save pooled array 
    pooled_arr = np.zeros((datacube.shape[0], datacube.shape[1]))
    pooled_arr[pooled_arr == 0] = np.nan
    pooled_arr[df.PIX_Y.astype('int').values, df.PIX_X.astype('int').values] = df.f444w.values/df.f444w_err.values
    hdu = fits.PrimaryHDU(data = pooled_arr, header = newhdr)
    hdu.writeto('Galaxies/{}/f444w_SNR_POOLED.fits'.format(galaxy_name), overwrite=True)
    
    # Store
    if store == True: df.to_csv('Galaxies/{}/photometries.csv'.format(galaxy_name), index = False)
    return df

# --------

def build_photometry_old(galaxy_name, map_tuple, errormap_tuple, working_bands, mask_ra, mask_dec, mask_radius, \
                    ulim_thresh = 3, store = True):
    # Here I'm taking advantage of the fact that the maps have the same pixel scale and nx/ny, so instead of doing
    # photometry on each position RA/DEC, I just stack everything togheter and then evaluate the RA/DEC in deg for
    # each pixel. This gets later placed in a single dataframe with ID, RA, DEC and photometry.
    nx, ny = map_tuple[0].hdr['NAXIS1'],  map_tuple[0].hdr['NAXIS2']
    x_indices, y_indices = np.meshgrid(np.arange(nx), np.arange(ny))
    coords = pixel_to_skycoord(y_indices, x_indices, map_tuple[0].wcs)
    cube_ra = np.stack([coord.ra.value for coord in coords], axis = 1)
    cube_dec = np.stack([coord.dec.value for coord in coords], axis = 1)
        
    # Build Photometry dataframe
    signal_tuple = [m.signal_convolved_with_nans for m in map_tuple]
    datacube = np.stack(signal_tuple, axis=2) # Already in uJy, but recheck the conversion bw MJy/sr and uJy/px
    error_tuple = [em.signal_convolved_with_nans for em in errormap_tuple]
    error_datacube = np.stack(error_tuple, axis=2) # Already in uJy, but recheck the conversion bw MJy/sr and uJy/px
    datacube = np.concatenate((datacube, error_datacube), axis = 2)
    datacube = np.concatenate((cube_ra[..., np.newaxis], cube_dec[..., np.newaxis], datacube), axis=2)
    datacube = np.concatenate((x_indices[..., np.newaxis], y_indices[..., np.newaxis], datacube), axis=2)
    
    # Generate a DataFrame with positions and photometry
    column_names = ['PIX_X', 'PIX_Y', 'RA', 'DEC'] + working_bands + [w+'_err' for w in working_bands]
    df = pd.DataFrame(np.reshape(datacube, (-1, datacube.shape[-1])), columns=column_names)
    df = df.reset_index().rename(columns={'index': 'ID'})
    
    # Mask the source, discard other pixels
    center_mask = SkyCoord(mask_ra, mask_dec, frame = 'icrs')
    df['sep'] = np.reshape(coords.separation(center_mask).arcsec, (-1, 1))
    df = df[df['sep'] < mask_radius.value]
    
    # Filter out rows where more than 3 values are upper limits
    df = df[(df[working_bands] < 0).sum(axis=1) <= ulim_thresh]
    df = df.reset_index(drop = True)
    
    # Correct per dust extinction at galaxy position
    Gal_EBV = AbdFunc.EBV_foreground_dust(center_mask)
    Gal_dust_corr_factor = {}
    for b in working_bands:
        leff = (config.band_wvl_association[b]*u.um).to('angstrom').value
        corr = pow(10.0, 0.4*AbdFunc.k_lmbd_Fitz1986_LMC(leff)*Gal_EBV)
        df[b] *= corr
        df[b+'_err'] *= corr
    
    # Order
    ordered_columns = ['ID', 'PIX_X', 'PIX_Y', 'RA', 'DEC', \
                       'f090w', 'f090w_err', 'f150w', 'f150w_err', 'f200w', 'f200w_err', \
                       'f277w', 'f277w_err', 'f356w', 'f356w_err', 'f410m', 'f410m_err', \
                       'f444w', 'f444w_err']
    df = df[ordered_columns]
    
    # Store
    if store == True: df.to_csv('Galaxies/{}/photometries.csv'.format(galaxy_name), index = False)
    return df
