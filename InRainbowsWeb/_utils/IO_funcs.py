import numpy as np
from astropy.io import fits, ascii
from astropy import units as u
from copy import deepcopy

class FitsUtils:
    def __init__(self, signal_path, filtername, telescope, kind = 'sci'):
        from astropy.wcs import WCS
        #import config
        self.fits_path = signal_path
        self.fits = fits.open(signal_path)
        self.signal = fits.getdata(self.fits_path)
        try: self.hdr = self.fits[1].header
        except: self.hdr = fits.getheader(signal_path)
        self.wcs = WCS(self.hdr)
        self.filtername = filtername
        self.kind = kind
        self.telescope = telescope
        
    def remove_nans(self):
        self.signal_with_nans = deepcopy(self.signal)
        self.signal = np.nan_to_num(self.signal_with_nans)

    def convert_10nJy_to_mJy(self):
        # 10 nJy = 1E-5 mJy 
        self.signal = ((self.signal*10)*u.nJy).to('mJy').value
        self.signal_with_nans = ((self.signal_with_nans*10)*u.nJy).to('mJy').value

    def convert_10nJy_to_uJy(self):
        # 10 nJy = 1E-5 mJy 
        self.signal = ((self.signal*10)*u.nJy).to('uJy').value
        self.signal_with_nans = ((self.signal_with_nans*10)*u.nJy).to('uJy').value

    def convert_MJysr_to_uJy(self, conversion_factor):
        # 1 MJy sr-1 = 1E-6 Jy sr-1 
        # pixel area in sr is in hdr['PIXAR_SR'].
        self.signal *= (1E6*conversion_factor)*u.Jy.to('uJy')
        self.signal_with_nans *= (1E6*conversion_factor)*u.Jy.to('uJy')
    
    def get_pixel_scale(self):
        import numpy as np
        try: pixel_scale = np.abs((self.hdr['CD1_1'])*u.deg.to('arcsec'))
        except: pixel_scale = np.abs((self.hdr['CDELT1'])*u.deg.to('arcsec'))
        return pixel_scale
    
    def degrade_to_worst(self, kernel_basepath, resize_kernel = False, save_fits = True):
        from astropy.convolution import convolve
        from scipy.ndimage import zoom
        # Do not convolve if already f444w
        if self.filtername == 'f444w':
            self.signal_convolved_with_nans = deepcopy(self.signal)
            self.signal_convolved = np.nan_to_num(self.signal_convolved_with_nans)
            return
        # Load the kernel
        sb, eb = '{}_{}'.format(self.telescope, self.filtername.upper()), 'JWST_F444W'
        self.kernel_path = kernel_basepath+'kernel_{0}_to_{1}.fits'.format(sb, eb)
        self.kernel = fits.getdata(self.kernel_path)
        # Resize image to match kernel pixel scale
        kernel_pixel_scale, image_pixel_scale = fits.getheader(self.kernel_path)['CD2_2'], self.hdr['CD2_2']
        self.regrid_factor = image_pixel_scale/kernel_pixel_scale
        self.signal_zoomed = zoom(self.signal, self.regrid_factor)
        # Square kernel for err (also square map) or variance map
        if self.kind == 'sci': signal_2conv, kernel_2conv = deepcopy(self.signal_zoomed), deepcopy(self.kernel)        
        elif self.kind == 'err': signal_2conv, kernel_2conv = self.signal_zoomed**2, self.kernel**2
        elif self.kind == 'wht': signal_2conv, kernel_2conv = deepcopy(self.signal_zoomed), self.kernel**2
        # Convolve
        sign_conv = convolve(signal_2conv, kernel = kernel_2conv, boundary = 'fill', normalize_kernel = True, preserve_nan = True)
        # Square root signal if error map
        if self.kind == 'err': sign_conv = np.sqrt(sign_conv)
        # Bring back everything to original pixel scale
        self.signal_convolved = zoom(sign_conv, 1/self.regrid_factor)        
        # Store also the nan version of the signal
        if hasattr(self, 'signal_with_nans'):
            ok_nan = np.where(np.nan_to_num(self.signal_with_nans-1) == 0) # I know, can't do anything 'bout it
            sign_conv[ok_nan] = np.nan
            self.signal_convolved_with_nans = sign_conv
        # Eventually save .fits
        if save_fits:
            hdu = fits.PrimaryHDU(data=self.signal_convolved, header = self.hdr)
            newpath = '/'.join(self.fits_path.split('/')[:-1])
            hdu.writeto('{0}/{1}_{2}_convolved.fits'.format(newpath, self.telescope, self.filtername), overwrite=True)
        
    def convert_wht_to_err(self):
        self.signal = np.sqrt(np.absolute(1/self.signal))
        if hasattr(self, 'signal_with_nans'): self.signal_with_nans = np.sqrt(np.absolute(1/self.signal_with_nans))
        if hasattr(self, 'signal_convolved'): self.signal_convolved = np.sqrt(np.absolute(1/self.signal_convolved))
        if hasattr(self, 'signal_convolved_with_nans'): self.signal_convolved_with_nans = np.sqrt(np.absolute(1/self.signal_convolved_with_nans))