import os, sys, subprocess
import numpy as np
from astropy.io import fits, ascii
from astropy import units as u
from astropy.constants import c as v_lux
from astropy.cosmology import Planck15 as cosmo
from astropy.coordinates import SkyCoord

def cwave_filters(filters):
    """Function for retrieving central wavelengths of a set of filters

    :param filters:
        A list of filters names

    :returns cwaves:
        A list of central wavelengths of the filters
    """
    import h5py
    f = h5py.File('_utils/filters_w.hdf5', 'r')
    nbands = len(filters)

    if nbands>1:
        cwaves = np.zeros(nbands)
        for bb in range(0,nbands):
            str_temp = 'cw_%s' % filters[bb]
            cwaves[bb] = f[filters[bb]].attrs[str_temp]
    else:
        str_temp = 'cw_%s' % filters
        cwaves = f[filters].attrs[str_temp]
    f.close()

    return cwaves

def k_lmbd_Fitz1986_LMC(wavelength_Ang):
    """A function for calculting dust extc. curve of Fitzpatrick et al. 1986.
    To be used for correction of foreground Galactic dust ettenuation 
    """

    if np.isscalar(wavelength_Ang)==False:
        lmbd_micron = np.asarray(wavelength_Ang)/1e+4
        inv_lmbd_micron = 1.0/lmbd_micron

        k = np.zeros(len(wavelength_Ang))

        idx = np.where(inv_lmbd_micron>=5.9)
        par1 = np.square(inv_lmbd_micron[idx[0]]-(4.608*4.608*lmbd_micron[idx[0]]))
        k[idx[0]] = -0.69 + (0.89/lmbd_micron[idx[0]]) + (2.55/(par1+(0.994*0.994))) + (0.5*((0.539*(inv_lmbd_micron[idx[0]]-5.9)*(inv_lmbd_micron[idx[0]]-5.9)) + (0.0564*(inv_lmbd_micron[idx[0]]-5.9)*(inv_lmbd_micron[idx[0]]-5.9)*(inv_lmbd_micron[idx[0]]-5.9)))) + 3.1
        
        idx = np.where((inv_lmbd_micron<5.9) & (inv_lmbd_micron>3.3))
        par1 = np.square(inv_lmbd_micron[idx[0]]-(4.608*4.608*lmbd_micron[idx[0]]))
        k[idx[0]] = -0.69 + (0.89/lmbd_micron[idx[0]]) + (3.55/(par1+(0.994*0.994))) + 3.1

        idx = np.where((inv_lmbd_micron<=3.3) & (inv_lmbd_micron>=1.1))
        yy = inv_lmbd_micron[idx[0]]-1.82
        ax = 1 + (0.17699*yy) - (0.50447*yy*yy) - (0.02427*yy*yy*yy) + (0.72085*yy*yy*yy*yy) + (0.01979*yy*yy*yy*yy*yy) - (0.77530*yy*yy*yy*yy*yy*yy) + (0.32999*yy*yy*yy*yy*yy*yy*yy)
        bx = (1.41338*yy) + (2.28305*yy*yy) + (1.07233*yy*yy*yy) - (5.38434*yy*yy*yy*yy) - (0.62251*yy*yy*yy*yy*yy) + (5.30260*yy*yy*yy*yy*yy*yy) - (2.09002*yy*yy*yy*yy*yy*yy*yy)
        k[idx[0]] = (3.1*ax) + bx

        idx = np.where(inv_lmbd_micron<1.1)
        ax = 0.574*np.power(inv_lmbd_micron[idx[0]],1.61)
        bx = -0.527*np.power(inv_lmbd_micron[idx[0]],1.61)
        k[idx[0]] = (3.1*ax) + bx
    else:
        lmbd_micron = wavelength_Ang/10000.0
        inv_lmbd_micron = 1.0/lmbd_micron

        if inv_lmbd_micron>=5.9:
            par1 = (inv_lmbd_micron-(4.608*4.608*lmbd_micron))*(inv_lmbd_micron-(4.608*4.608*lmbd_micron))
            k = -0.69 + (0.89/lmbd_micron) + (2.55/(par1+(0.994*0.994))) + (0.5*((0.539*(inv_lmbd_micron-5.9)*(inv_lmbd_micron-5.9)) + (0.0564*(inv_lmbd_micron-5.9)*(inv_lmbd_micron-5.9)*(inv_lmbd_micron-5.9)))) + 3.1
        elif inv_lmbd_micron<5.9 and inv_lmbd_micron>3.3:
            par1 = (inv_lmbd_micron-(4.608*4.608*lmbd_micron))*(inv_lmbd_micron-(4.608*4.608*lmbd_micron))
            k = -0.69 + (0.89/lmbd_micron) + (3.55/(par1+(0.994*0.994))) + 3.1
        elif inv_lmbd_micron<=3.3 and inv_lmbd_micron>=1.1:
            yy = inv_lmbd_micron-1.82
            ax = 1 + 0.17699*yy - 0.50447*yy*yy - 0.02427*yy*yy*yy + 0.72085*yy*yy*yy*yy + 0.01979*yy*yy*yy*yy*yy - 0.77530*yy*yy*yy*yy*yy*yy + 0.32999*yy*yy*yy*yy*yy*yy*yy
            bx = 1.41338*yy + 2.28305*yy*yy + 1.07233*yy*yy*yy - 5.38434*yy*yy*yy*yy - 0.62251*yy*yy*yy*yy*yy + 5.30260*yy*yy*yy*yy*yy*yy - 2.09002*yy*yy*yy*yy*yy*yy*yy
            k = 3.1*ax + bx
        elif inv_lmbd_micron<1.1:
            ax = 0.574*pow(inv_lmbd_micron,1.61)
            bx = -0.527*pow(inv_lmbd_micron,1.61)
            k = 3.1*ax + bx
    return k

def EBV_foreground_dust(coord):
    """Function for estimating E(B-V) dust attenuation due to the foreground Galactic dust attenuation at a given coordinate on the sky.

    :param ra:
        Right ascension coordinate in degree.

    :param dec:
        Declination coordinate in degree.

    :returns ebv:
        E(B-V) value.
    """

    from astroquery.irsa_dust import IrsaDust
    table = IrsaDust.get_extinction_table(coord)

    Alambda_SDSS = np.zeros(5)
    Alambda_SDSS[0] = np.array(table['A_SandF'][table['Filter_name']=='SDSS u'])[0]
    Alambda_SDSS[1] = np.array(table['A_SandF'][table['Filter_name']=='SDSS g'])[0]
    Alambda_SDSS[2] = np.array(table['A_SandF'][table['Filter_name']=='SDSS r'])[0]
    Alambda_SDSS[3] = np.array(table['A_SandF'][table['Filter_name']=='SDSS i'])[0]
    Alambda_SDSS[4] = np.array(table['A_SandF'][table['Filter_name']=='SDSS z'])[0]

    # central wavelengths of the SDSS 5 bands: 
    filters = ['sdss_u', 'sdss_g', 'sdss_r', 'sdss_i', 'sdss_z']
    wave_SDSS = cwave_filters(filters)

    # calculate average E(B-V):
    ebv_SDSS = Alambda_SDSS/k_lmbd_Fitz1986_LMC(wave_SDSS)
    ebv = np.mean(ebv_SDSS)
    return ebv