'''
	specfit.py - Definition of class for fitting linear combination of spectra.
'''

######################################################################

import os
import numpy as np
from astropy.io import fits as pyfits
from astropysics import spec
import scipy.ndimage.filters
import scipy.interpolate
import logging
import scipy.constants

_c_kms = scipy.constants.c / 1.e3  # Speed of light in km s^-1
DF = -8.0


class SpecFit():
    ##################################################################

    def __init__(self, nspec):

        '''
Initialize class.
    Input:
        nspec = Number of spectra that composes the observed spectra
        '''

        # number of componentes
        self.nspec = nspec

        # Store template spectra and scale factor
        self.template = [[]] * nspec
        self.templateNames = [[]] * nspec
        self.templateScale = [[]] * nspec
        self.specgrid = [[]] * nspec

        # velocity for each component
        self.vel = np.zeros(nspec)

        # scale factor for each component
        self.scale = [[]] * nspec
        # self.mcscale = pm.Uniform("scale", 0, 1, size=nspec)

        # template selected for each component
        self.ntemp = np.zeros(nspec, dtype=int)

        # template grid dimensions for each component
        self.grid_ndim = np.zeros(nspec, dtype=int)

        # Grids
        self.Grid = [[]] * nspec

        # store the observed spectra
        self.ospec = None

        self._autoprop = False

    ##################################################################

    def setAutoProp(self, value):
        self._autoprop = value

    ##################################################################

    def loadNextGenTemplate(self, ncomp, filename):
        '''
Loads template spectra from a list of files (in filename), for
component ncomp.
        '''

        splist = np.loadtxt(filename, unpack=True, usecols=(0,),
                            dtype='S', ndmin=1)

        self.template[ncomp] = [0] * len(splist)
        self.templateScale[ncomp] = [1] * len(splist)

        logging.debug('Loading template spectra for component %i from %s[%i]' % (ncomp, filename, len(splist)))

        for i in range(len(splist)):
            logging.debug('Reading %s' % (splist[i]))
            sp = np.loadtxt(splist[i], unpack=True, usecols=(0, 1),
                            converters={0: lambda s: float(s.replace('D', 'e')),
                                        1: lambda s: float(s.replace('D', 'e'))})
            asort = sp[0].argsort()
            self.template[ncomp][i] = spec.Spectrum(sp[0][asort],
                                                    10 ** (sp[1][asort]) + 8.0)

        return 0

    ##################################################################

    def loadPickleTemplate(self, ncomp, filename):
        '''
Loads template spectra from a list of files (in filename), for
component ncomp.
        '''

        splist = np.loadtxt(filename, unpack=True,
                            dtype='S', ndmin=2)
        if splist.shape[0] < self.grid_ndim[ncomp]:
            raise IOError('Grid dimensions is not consistent with expected. Expecting %i got %i.' % (
                self.grid_ndim[ncomp], splist.shape[0]))

        self.template[ncomp] = [0] * len(splist[0])
        self.templateNames[ncomp] = [0] * len(splist[0])
        self.templateScale[ncomp] = [1] * len(splist[0])  # np.zeros(len(splist))+1.0

        if self.grid_ndim[ncomp] > 0:
            grid = splist[1:self.grid_ndim[ncomp] + 1]
            gdim = np.zeros(self.grid_ndim[ncomp])
            for i in range(len(grid)):
                gdim[i] = len(np.unique(grid[i]))
            index = np.arange(len(splist[0])).reshape(gdim)
            self.Grid[ncomp] = index

        logging.debug('Loading template spectra for component %i from %s[%i]' % (ncomp, filename, len(splist)))

        for i in range(len(splist[0])):
            logging.debug('Reading %s' % (splist[0][i]))
            sp = np.load(splist[0][i])
            self.template[ncomp][i] = spec.Spectrum(sp[0], sp[1])
            self.templateNames[ncomp][i] = splist[0][i]

        return 0

    ##################################################################

    def loadCoelhoTemplate(self, ncomp, filename):
        '''
        Loads template spectra from a list of files (in filename), for
        component ncomp.
        '''

        splist = np.loadtxt(filename, unpack=True,
                            dtype='S', ndmin=2)
        if splist.shape[0] < self.grid_ndim[ncomp]:
            raise IOError('Grid dimensions is not consistent with expected. Expecting %i got %i.' % (
                self.grid_ndim[ncomp], splist.shape[0]))

        self.template[ncomp] = [0] * len(splist[0])
        self.templateNames[ncomp] = [0] * len(splist[0])
        self.templateScale[ncomp] = [1] * len(splist[0])

        if self.grid_ndim[ncomp] > 0:
            grid = splist[1:self.grid_ndim[ncomp] + 1]
            index = np.arange(len(splist[0])).reshape((len(np.unique(grid[0])), len(np.unique(grid[1]))))
            self.Grid[ncomp] = index

        logging.debug('Loading template spectra for component %i from %s[%i]' % (ncomp, filename, len(splist)))

        notFound = 0
        for i in range(len(splist[0])):

            logging.debug('Reading %s' % (splist[0][i]))
            if os.path.exists(splist[0][i]):
                hdu = pyfits.open(splist[0][i])
                wave = hdu[0].header['CRVAL1'] + np.arange(len(hdu[0].data)) * hdu[0].header['CDELT1']
                self.template[ncomp][i] = spec.Spectrum(wave, hdu[0].data)
                self.templateNames[ncomp][i] = splist[0][i]
            else:
                logging.warning('Could not find template %s. %i/%i' % (splist[0][i], notFound, len(splist[0])))
                notFound += 1
                self.template[ncomp][i] = self.template[ncomp][i - 1]
                self.templateNames[ncomp][i] = splist[0][i] + "NOTFOUND"

                # sp = np.load(splist[0][i])
        if notFound > len(splist[0]) / 2:
            raise IOError('More than 50% of template spectra could not be loaded')

        return 0

    ##################################################################

    def loadPickle(self, filename, linearize=True):
        '''
Loads observed spectra from numpy pickle file.
        '''

        logging.debug('Loading observed spectra for from %s' % (filename))

        sp = np.load(filename)

        self.ospec = spec.Spectrum(sp[0], sp[1])

        if linearize and not self.ospec.isLinear():
            logging.debug('Linearizing observed spectra')
            self.ospec.linearize()
            logging.debug('Done')

        return 0

    ##################################################################

    def loadtxtSpec(self, filename):
        '''
Load the observed spectra.
        '''

        logging.debug('Loading observed spectra for from %s' % (filename))

        sp = np.loadtxt(filename, unpack=True, usecols=(0, 1),
                        converters={0: lambda s: float(s.replace('D', 'e')),
                                    1: lambda s: float(s.replace('D', 'e'))})

        self.ospec = spec.Spectrum(sp[0], sp[1])

        return 0

    ##################################################################

    def loadSDSSFits(self, filename, linearize=False):
        '''
Load the observed spectra.
        '''

        logging.debug('Loading observed spectra for from %s' % (filename))

        sp = pyfits.open(filename)

        mask = np.bitwise_and(sp[1].data['and_mask'] == 0,
                              sp[1].data['or_mask'] == 0)

        self.ospec = spec.Spectrum(x=10 ** (sp[1].data['loglam'][mask]),
                                   flux=sp[1].data['flux'][mask],
                                   ivar=sp[1].data['ivar'][mask])

        '''
        if linearize and not self.ospec.isLinear():
            logging.debug('Linearizing observed spectra')
            self.ospec.linearize()
            logging.debug('Done')
        '''

        return 0

    ##################################################################

    def gridSpec(self, ncomp=0):
        '''
        Resample and grid template spectrum.
        :return:
        '''

        # Use first spectrum as reference
        refspec = self.template[ncomp][0]

        specgrid = np.zeros((len(self.template[ncomp]), len(refspec.flux)))

        for i in range(len(specgrid)):
            specgrid[i] += self.template[ncomp][i].resample(refspec.x, replace=False)[1] * \
                           self.templateScale[ncomp][i]

        self.specgrid[ncomp] = specgrid
        self.scale[ncomp] = np.zeros(len(specgrid)).reshape(len(specgrid), -1) + 1. / len(specgrid)

    ##################################################################

    def chi2(self, p):
        '''
Calculate chi-square of the data against model.
        '''

        for i in range(self.nspec):
            logging.debug('%f / %f' % (p[i], p[i + 1]))
            self.scale[i] = p[i * 2]
            self.vel[i] = p[i * 2 + 1]

        model = self.modelSpec()

        # c2 = np.mean( (self.ospec.flux - model.flux )**2.0 / self.ospec.flux)
        c2 = self.ospec.flux - model.flux
        return c2

    ##################################################################

    def modelSpec(self):
        '''
Calculate model spectra.
        '''

        # _model = self.template[0][self.ntemp[0]]

        logging.debug('Building model spectra')

        dopCor = np.sqrt((1.0 + self.vel[0] / _c_kms) / (1. - self.vel[0] / _c_kms))
        scale = self.scale[0] * self.templateScale[0][self.ntemp[0]]

        _model = MySpectrum(self.template[0][self.ntemp[0]].x * dopCor,
                            self.template[0][self.ntemp[0]].flux * scale)

        # logging.debug('Applying instrument signature')

        # kernel = self.obsRes()/np.mean(_model.x[1:]-_model.x[:-1])

        # _model.flux = scipy.ndimage.filters.gaussian_filter(_model.flux,kernel)


        for i in range(1, self.nspec):
            dopCor = np.sqrt((1.0 + self.vel[i] / _c_kms) / (1. - self.vel[i] / _c_kms))
            scale = self.scale[i] * self.templateScale[i][self.ntemp[i]]

            tmp = MySpectrum(self.template[i][self.ntemp[i]].x * dopCor,
                             self.template[i][self.ntemp[i]].flux * scale)

            # logging.debug('Applying instrument signature')

            # kernel = self.obsRes()/np.mean(tmp.x[1:]-tmp.x[:-1])

            # tmp.flux = scipy.ndimage.filters.gaussian_filter(tmp.flux,kernel)

            tmp = MySpectrum(*tmp.resample(_model.x, replace=False))

            _model.flux += tmp.flux

        '''
        if not _model.isLinear():
            logging.warning('Data must be linearized...')

        '''
        # kernel = self.obsRes()/tmp.getDx()/2./np.pi

        # _model.flux = scipy.ndimage.filters.gaussian_filter(_model.flux,kernel)

        logging.debug('Resampling model spectra')
        _model = MySpectrum(*_model.myResample(self.ospec.x, replace=False))
        if self._autoprop:
            mflux = np.mean(_model.flux)
            oflux = np.mean(self.ospec.flux)
            _model.flux *= (oflux / mflux)
        return _model

    ##################################################################

    def modelSpecThreadSafe(self, vel, scale, ntemp):
        '''
Calculate model spectra.
        '''

        # _model = self.template[0][self.ntemp[0]]

        logging.debug('Building model spectra')

        dopCor = np.sqrt((1.0 + vel[0] / _c_kms) / (1. - vel[0] / _c_kms))
        scale = scale[0] * self.templateScale[0][ntemp[0]]

        _model = MySpectrum(self.template[0][ntemp[0]].x * dopCor,
                            self.template[0][ntemp[0]].flux * scale)

        # logging.debug('Applying instrument signature')

        # kernel = self.obsRes()/np.mean(_model.x[1:]-_model.x[:-1])

        # _model.flux = scipy.ndimage.filters.gaussian_filter(_model.flux,kernel)


        for i in range(1, self.nspec):
            dopCor = np.sqrt((1.0 + vel[i] / _c_kms) / (1. - vel[i] / _c_kms))
            scale = scale[i] * self.templateScale[i][ntemp[i]]

            tmp = MySpectrum(self.template[i][ntemp[i]].x * dopCor,
                             self.template[i][ntemp[i]].flux * scale)

            # logging.debug('Applying instrument signature')

            # kernel = self.obsRes()/np.mean(tmp.x[1:]-tmp.x[:-1])

            # tmp.flux = scipy.ndimage.filters.gaussian_filter(tmp.flux,kernel)

            tmp = MySpectrum(*tmp.resample(_model.x, replace=False))

            _model.flux += tmp.flux

        '''
        if not _model.isLinear():
            logging.warning('Data must be linearized...')

        '''
        # kernel = self.obsRes()/tmp.getDx()/2./np.pi

        # _model.flux = scipy.ndimage.filters.gaussian_filter(_model.flux,kernel)

        logging.debug('Resampling model spectra')
        _model = MySpectrum(*_model.myResample(self.ospec.x, replace=False))
        return _model

    ##################################################################

    def normTemplate(self, ncomp, w0, w1):
        '''
Normalize spectra against data in the wavelenght regions
        '''

        for i in range(len(self.template[ncomp])):
            maskt = np.bitwise_and(self.template[ncomp][i].x > w0,
                                   self.template[ncomp][i].x < w1)
            mask0 = np.bitwise_and(self.ospec.x > w0,
                                   self.ospec.x < w1)

            scale = np.mean(self.ospec.flux[mask0]) / np.mean(self.template[ncomp][i].flux[maskt])

            self.templateScale[ncomp][i] = scale
            # self.template[ncomp][i].flux *= scale

    ##################################################################

    def gaussian_filter(self, ncomp, kernel):

        for i in range(len(self.template[ncomp])):
            if not self.template[ncomp][i].isLinear():
                logging.warning('Spectra must be linearized for gaussian filter...')

            self.template[ncomp][i].flux = scipy.ndimage.filters.gaussian_filter(self.template[ncomp][i].flux, kernel)

    ##################################################################

    def obsRes(self):
        return self.ospec.getDx()

    ##################################################################

    def preprocTemplate(self):
        '''
Pre-process all template spectra to have aproximate coordinates as
those of the observed spectrum and linearize the spectrum.
        '''

        logging.debug('Preprocessing all template spectra. Spectra will be trimmed and linearized')

        ores = self.obsRes()

        xmin = np.max([self.template[0][0].x[0], self.ospec.x[0] - 100.0 * ores])
        xmax = np.min([self.template[0][0].x[-1], self.ospec.x[-1] + 100.0 * ores])

        for i in range(self.nspec):
            for j in range(len(self.template[i])):
                # t_res = np.mean(self.template[i][j].x[1:]-self.template[i][j].x[:-1])
                # newx = np.arange(xmin,xmax,t_res)
                # self.template[i][j] = spec.Spectrum(*self.template[i][j].resample(newx,replace=False))

                self.template[i][j].linearize(lower=xmin, upper=xmax)

                tmp_spres = np.mean(self.template[i][j].x[1:] - self.template[i][j].x[:-1])
                logging.debug('Template spres = %f' % (tmp_spres))
                logging.debug('Data spres = %f' % (ores))

                if tmp_spres < ores / 10.:
                    logging.debug('Template spectroscopic resolution too high! Resampling...')
                    newx = np.arange(xmin, xmax, ores / 10.)
                    self.template[i][j] = spec.Spectrum(*self.template[i][j].resample(newx, replace=False))

    ##################################################################

    def saveTemplates2Pickle(self, ncomp, filename):

        splist = np.loadtxt(filename, unpack=True, usecols=(0,),
                            dtype='S', ndmin=1)

        logging.debug('Saving template spectra to pickle file...')

        for ntemp in range(len(self.template[ncomp])):
            logging.debug(splist[ntemp])
            sp = np.array([self.template[ncomp][ntemp].x,
                           self.template[ncomp][ntemp].flux])
            np.save(splist[ntemp], sp)

    ##################################################################

    def suitableScale(self):
        '''
Find a suitable scale values for all spectra.
        '''

        logging.debug('Looking for suitable scale in all spectra. Will choose the larger value.')

        obsmean = np.mean(self.ospec.flux)
        maxscale = 0.
        minscale = obsmean

        for i in range(len(self.template)):
            for j in range(len(self.template[i])):
                maskt = np.bitwise_and(self.template[i][j].x > self.ospec.x[0],
                                       self.template[i][j].x < self.ospec.x[-1])

                nscale = obsmean / np.mean(self.template[i][j].flux[maskt]) / self.templateScale[i][j]

                if maxscale < nscale:
                    maxscale = nscale
                if minscale > nscale:
                    minscale = nscale

        return maxscale, minscale


######################################################################

class MySpectrum(spec.Spectrum):
    def __init__(self, x, flux, err=None, ivar=None,
                 unit='wl', name='', copy=True, sort=True):
        spec.Spectrum.__init__(self, x=x, flux=flux, err=err, ivar=ivar,
                               unit=unit, name=name, copy=copy, sort=sort)

    ##################################################################

    def myResample(self, newx, replace=False):
        '''
        kernel = np.mean(newx[1:]-newx[:-1])/np.mean(self.x[1:]-self.x[:-1])

        dx = self.x[1:]-self.x[:-1]

        newy = scipy.ndimage.filters.gaussian_filter(self.flux,np.float(kernel))
        tck = scipy.interpolate.splrep(self.x,newy)
        newy2 =scipy.interpolate.splev(newx,tck)
        '''

        kernel = np.median(newx[1:] - newx[:-1]) / np.median(self.x[1:] - self.x[:-1])  # *2.0 #/2./np.pi

        newflux = scipy.ndimage.filters.gaussian_filter1d(self.flux, kernel)

        tck = scipy.interpolate.splrep(self.x, newflux)

        return newx, scipy.interpolate.splev(newx, tck)

        '''
        newy = np.zeros(len(newx))

        for i in range(len(newx)):
            xini = 0
            xend = 0

            if i == 0:
                xini = newx[i]-(newx[i+1]-newx[i])/2.
            else:
                xini = newx[i]-(newx[i]-newx[i-1])/2.

            if i == len(newx)-1:
                xend = newx[i]+(newx[i]-newx[i-1])/2.
            else:
                xend = newx[i]+(newx[i+1]-newx[i])/2.

            mask = np.bitwise_and(self.x > xini, self.x < xend)

            #newy[i] = np.sum( dx[mask[:-1]] * self.flux[mask] )
            newy[i] = np.mean(self.flux[mask])
            #print newx[i],newy[i],newy2[i],xini,xend, (xend-xini) , np.mean(self.flux[mask]),(xend-xini) * np.mean(self.flux[mask])
            #print self.x[mask],self.flux[mask],dx[mask[:-1]]



        return newx,newy #scipy.interpolate.splev(newx,tck)
'''

        ##################################################################

######################################################################
