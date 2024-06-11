from __future__ import print_function,  division,  absolute_import

import os, sys
import numpy as np
from pathlib import Path
home = str(Path.home())
CosmosPath = home+'/Dropbox/Work_Stuff/GalaxyFormation/JWSTCosmosWeb/'

""" InRainbowsWeb configuration variables. """

paths_dict = {'current': home+'/scaloni/sed_fitting/scripts/',
              #'kernels':  CosmosPath+'kernels/',
              'kernels': home+'/scaloni/sed_fitting/kernels/',
              'maps':  home+'/scaloni/sed_fitting/maps/',
              'catalogs':  home+'/scaloni/sed_fitting/catalogs/'}

maps_paths_dict = {'f814w': 'ACS/mosaic_cosmos_primer_60mas_hst_acs_wfc_f814w',
                   'f090w': 'NIRcam/mosaic_nircam_f090w_PRIMER-COSMOS_epoch2_60mas_v0_3',
                   'f115w': 'NIRcam/mosaic_nircam_f115w_PRIMER-COSMOS_epoch2_60mas_v0_3',
                   'f150w': 'NIRcam/mosaic_nircam_f150w_PRIMER-COSMOS_epoch2_60mas_v0_3',
                   'f200w': 'NIRcam/mosaic_nircam_f200w_PRIMER-COSMOS_epoch2_60mas_v0_3',
                   'f277w': 'NIRcam/mosaic_nircam_f277w_PRIMER-COSMOS_epoch2_60mas_v0_3',
                   'f356w': 'NIRcam/mosaic_nircam_f356w_PRIMER-COSMOS_epoch2_60mas_v0_3',
                   'f410m': 'NIRcam/mosaic_nircam_f410m_PRIMER-COSMOS_epoch2_60mas_v0_3',
                   'f444w': 'NIRcam/mosaic_nircam_f444w_PRIMER-COSMOS_epoch2_60mas_v0_3'}

maps_appendix_dict = {'sci': '_sci.fits.gz',
                      'drz': '_drz.fits.gz',
                      'err': '_err.fits.gz',
                      'wht': '_wht.fits.gz'}

band_wvl_association = {'f814w': 0.814, 'f090w': 0.900, 'f115w': 1.150,
                        'f150w': 1.500, 'f200w': 2.000, 'f277w': 2.770,
                        'f356w': 3.560, 'f410m': 4.100, 'f444w': 4.440}

def get_map_path(band, kind):
    return paths_dict['maps']+maps_paths_dict[band]+maps_appendix_dict[kind]
