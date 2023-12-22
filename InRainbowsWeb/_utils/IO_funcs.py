import numpy as np
from astropy.io import fits, ascii
from astropy import units as u

class FitsUtils:
    def __init__(self, signal_path, filtername, kind = 'sci'):
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
        
    def remove_nans(self):
        self.signal_with_nans = self.signal.copy()
        self.signal = np.nan_to_num(self.signal_with_nans)

    def convert_MJysr_to_uJy(self, conversion_factor):
        # 1 MJy sr-1 = 1E-6 Jy sr-1 
        # pixel area in sr is in hdr['PIXAR_SR'].
        self.signal *= (1E6*conversion_factor)*u.Jy.to('uJy')
        self.signal_with_nans *= (1E6*conversion_factor)*u.Jy.to('uJy')
    
    def get_pixel_scale(self):
        import numpy as np
        if ('CDELT1' in self.hdr) & ('CDELT2' in self.hdr):
            pixel_scale_x=abs(self.hdr['CDELT1'])
            pixel_scale_y=abs(self.hdr['CDELT2'])
        elif ('CD1_1' in self.hdr) & ('CD1_2' in self.hdr) & \
                ('CD2_1' in self.hdr) & ('CD2_2' in self.hdr):
            _ = np.arctan(self.hdr['CD2_1']/self.hdr['CD1_1'])
            pixel_scale_x = abs(self.hdr['CD1_1']/np.cos(_))
            pixel_scale_y = abs(self.hdr['CD2_2']/np.cos(_))
        else:
            raise ValueError
        pixel_scale = np.sqrt(pixel_scale_x*pixel_scale_y)
        return pixel_scale
    
    def degrade_to_worst(self, kernel_basepath, resize_kernel = True):
        from astropy.convolution import convolve
        ## Do not convolve if already f444w, if blabla
        #if self.filtername == 'f444w':
        #    self.signal_convolved = self.signal.copy()
        #    return
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
            kernel_2conv = resize(self.kernel, (601, 601), preserve_range = True)
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
        if hasattr(self, 'signal_with_nans'):
            ok_nan = np.where(np.nan_to_num(self.signal_with_nans-1) == 0) # I know, can't do anything 'bout it
            sign_conv[ok_nan] = np.nan
            self.signal_convolved_with_nans = sign_conv
        
    def convert_wht_to_err(self):
        self.signal = np.sqrt(np.absolute(1/self.signal))
        if hasattr(self, 'signal_with_nans'): self.signal_with_nans = np.sqrt(np.absolute(1/self.signal_with_nans))
        if hasattr(self, 'signal_convolved'): self.signal_convolved = np.sqrt(np.absolute(1/self.signal_convolved))
        if hasattr(self, 'signal_convolved_with_nans'): self.signal_convolved_with_nans = np.sqrt(np.absolute(1/self.signal_convolved_with_nans))