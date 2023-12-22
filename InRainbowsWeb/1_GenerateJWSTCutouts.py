import argparse
import os, sys, subprocess, multiprocessing
from pathlib import Path
max_cores = multiprocessing.cpu_count()
import warnings
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.nddata.utils import Cutout2D
from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import config

class FitsUtils:
    '''
    This is a specific version of FitsUtils, working ONLY for PRIMER maps.
    The fact is that PRIMER error maps, buttana a cu 'i fici e a cu 'un ciu rici,
    DO NOT have a valid header; hell, valid, do not have a header altogether.
    So, the cutouts generation will fail miserably. The workaround is to
    take the header from the corresponding _sci map, and stick it to the
    generated cutouts. In this way, the _sci and _err cutouts will share the same
    WCS, and perform spatially resolved photometry gets much easier.
    THIS MUST BE CHANGED WHEN DEALING WITH BRAMMER GRIZLI RESULTS, SINCE THE _WHT
    MAPS THAT GABE GIVES HAVE A VALID, DECENT HEADER.
    '''
    def __init__(self, signal_path, filtername, kind = 'sci'):
        import numpy as np
        from astropy.io import fits
        from astropy.wcs import WCS
        self.fits_path = signal_path
        self.fits = fits.open(signal_path)
        if kind == 'sci':
            self.signal_with_nans = self.fits[0].data
            self.signal = np.nan_to_num(self.signal_with_nans)
            self.hdr = self.fits[0].header
        elif kind == 'err':
            self.signal_with_nans = self.fits[0].data
            self.signal = np.nan_to_num(self.signal_with_nans)
            scipath = signal_path[:-12]+'_sci.fits.gz'
            self.hdr = fits.open(scipath)[0].header
        elif kind == 'wht':
            self.signal_with_nans = np.sqrt(np.absolute(1/fits.getdata(self.fits_path)))
            self.signal = np.nan_to_num(self.signal_with_nans)
            scipath = signal_path[:-12]+'_sci.fits.gz'
            self.hdr = fits.open(scipath)[0].header
        self.wcs = WCS(self.hdr)
        self.filtername = filtername
        self.kind = kind

working_bands = ['f090w', 'f150w', 'f200w', 'f277w', 'f356w', 'f410m', 'f444w']

def split_per_map(band, map_kind, df, cutout_size, master_outpath):
    print()
    print('Loading {}...'.format(band))
    map_bundle = FitsUtils(config.get_map_path(band, map_kind), band, map_kind)
    # Generate cutouts. !!! Not pickleable, so I'm stuck with for loop !!!
    for _, row in tqdm(df.iterrows(), total = len(df), desc='Generating cutouts for band {0}'.format(map_bundle.filtername), \
                   bar_format='{l_bar}{bar}| Elapsed: {elapsed} ETA: {remaining}'): generate_cutout(row, map_bundle, cutout_size, master_outpath)
    del map_bundle # MEMORY SAVER
    return

def generate_cutout(row, fits_utils, cutout_size, master_outpath):
    if os.path.exists(master_outpath+'{0:.0f}/{1}_{2}.fits'.format(row['COSMOS_ID'], fits_utils.filtername, fits_utils.kind)): return
    coord = SkyCoord(row['ALPHA_J2000']*u.deg, row['DELTA_J2000']*u.deg, frame = 'icrs')
    cutout = Cutout2D(fits_utils.signal, coord, cutout_size, fits_utils.wcs, fill_value = 'nan')
    if cutout.data.max() == 0: return
    source_pos = np.array(cutout.to_cutout_position((fits_utils.wcs.wcs_world2pix(row['ALPHA_J2000'], row['DELTA_J2000'], 0)))).astype('int')
    if cutout.data[source_pos[0], source_pos[1]] == 0: return

    # Create folder
    if os.path.exists(master_outpath+'{:.0f}'.format(row['COSMOS_ID'])): pass
    else: subprocess.call('mkdir '+master_outpath+'{:.0f}'.format(row['COSMOS_ID']), shell = True)

    # Save the .fits cutout
    hdu = fits.ImageHDU()
    hdu.data = cutout.data
    hdu.header = cutout.wcs.to_header()
    hdu.verify('fix') # DIO CRASTO
    hdu.writeto(master_outpath+'{0:.0f}/{1}_{2}.fits'.format(row['COSMOS_ID'], fits_utils.filtername, fits_utils.kind), overwrite=True)
    return

parser = argparse.ArgumentParser(description = 'Generate cutouts from JWST maps, given a set of coordinates.')
parser.add_argument('--map_kind', type = str, help = 'Science map (sci), weight map (wht) or error map (err).', required = True)
parser.add_argument('--catalog_path', type = str, help = 'Path to the input catalog (from COSMOS2020).', required = True)
parser.add_argument('--cutout_size', type = float, help = 'Cutout size, in arcsec.', required = True)
parser.add_argument('--cutouts_path', type = str, default = 'Galaxies/', help = 'Master path for the cutouts. Default is Galaxies/')
parser.add_argument('--cores', type = int, default = 0, choices = range(1, max_cores+1), help = 'The maximum number of cores to use. Default is the maximum number of available cores minus 2.')
args = parser.parse_args()

if __name__ == '__main__':
    # Number of cores to use to run everything in parallel.
    if args.cores == 0: max_workers = max_cores - 2
    else: max_workers = args.cores
    with ProcessPoolExecutor(max_workers = max_workers) as executor: 
        executor.map(partial(split_per_map, map_kind = args.map_kind, df = pd.read_csv(args.catalog_path), \
                cutout_size = args.cutout_size*u.arcsec, master_outpath = args.cutouts_path), working_bands)