
'''
PyMCMC model factory.

'''

import numpy as np
import pymc
from StringIO import StringIO
import logging
import scipy.constants
from astropysics import spec
from specfit.lib.specfit import SpecFit

_c_kms = scipy.constants.c / 1.e3  # Speed of light in km s^-1

__ncomp__ = 2

def ModelFactory(ofile,templist,temptype):
    '''
A model factory to do the spectroscopic fit with PyMC.

Input:
        ofile		- The observed spectra.
        templist	- The template list.
        temptype    - Type of template spectra. (Chooses read algorithm)
                      1 - Pickle
                      2 - SDSSFits
                      3 - Coelho 2014 model spectra
C - Tiago Ribeiro
    '''

    spMod = SpecFit(__ncomp__)


    # Load observed spectra

    if ofile.rfind('.fits') > 0:
        spMod.loadSDSSFits(ofile,True)
    elif ofile.rfind('.npy') > 0:
        spMod.loadPickle(ofile)
    else:
        raise IOError('Cannot read %s data type. Only SDSS type fits and numpy (pickle) \
                      files are suported.'%(ofile))

    # Load template spectra, for each component

    for i in range(__ncomp__):
        fp = open(templist[i],'r')
        line1 = StringIO(fp.readline())
        line1 = np.loadtxt(line1,dtype='S')
        spMod.grid_ndim[i] = len(line1)-1 # Set grid dimension for 1st component

        logging.debug('Component %s[%i] with %i dimentions.'%(templist[i],i,spMod.grid_ndim[i]))

        if temptype == 1:
            spMod.loadPickleTemplate(i,templist[i])
        elif temptype == 2:
            spMod.loadSDSSFitsTemplate(i,templist[i])
        elif temptype == 3:
            spMod.loadCoelhoTemplate(i,templist[i])

    # Pre-process template files

    # spMod.normTemplate(0,5500.,5520.) # TODO: Is this a suitable range?
    # spMod.normTemplate(1,5500.,5520.)
    spMod.normTemplate(0,5500.,9000.) # TODO: Is this a suitable range?
    spMod.normTemplate(1,5500.,9000.)

    spMod.setAutoProp(True)

    spMod.preprocTemplate()

    for n in range(__ncomp__):
        spMod.gridSpec(n)

    # Prepare PyMC stochastic variables

    scale1 = [ pymc.Uniform('scale1_%06i'%i, 0., 1., value=0.5/len(spMod.specgrid[0]))
                       for i in range(len(spMod.specgrid[0])) ]

    scale2 = [ pymc.Uniform('scale2_%06i'%i, 0., 1., value=0.5/len(spMod.specgrid[1]))
                       for i in range(len(spMod.specgrid[1])) ]

    vmin = -100.
    vmax = +100.
    velocity1 = pymc.Uniform('velocity1', vmin, vmax , 0.)
    velocity2 = pymc.Uniform('velocity2', vmin, vmax , 0.)

    @pymc.deterministic
    def spModel(scale1=scale1,velocity1=velocity1,scale2=scale2,velocity2=velocity2):

        flux1 = np.sum(np.multiply(spMod.specgrid[0],
                                   np.array(scale1).reshape(len(scale1),-1)),
                       axis=0)
        flux2 = np.sum(np.multiply(spMod.specgrid[1],
                                   np.array(scale2).reshape(len(scale2),-1)),
                       axis=0)

        dopCor1 = np.sqrt((1.0 + velocity1 / _c_kms) / (1. - velocity1 / _c_kms))
        dopCor2 = np.sqrt((1.0 + velocity2 / _c_kms) / (1. - velocity2 / _c_kms))

        model1 = spec.Spectrum(spMod.template[0][0].x * dopCor1,
                               flux1)
        model2 = spec.Spectrum(spMod.template[1][0].x * dopCor2,
                               flux2)

        flux = model1.resample(spMod.ospec.x,replace=False)[1] + \
               model2.resample(spMod.ospec.x,replace=False)[1]

        logging.debug('spmodel: [%05i]:%10.6f %+14.3f [%05i]:%10.6f %+14.3f' % (np.argmax(scale1),
                                                          np.max(scale1),
                                                          velocity1,
                                                          np.argmax(scale2),
                                                          np.max(scale2),
                                                          velocity2))

        return flux

    mflux = np.mean(spMod.ospec.flux)
    sig = pymc.Uniform('sig',
                       np.median(spMod.ospec.err),
                       np.median(spMod.ospec.flux),
                       value=np.median(spMod.ospec.flux)/2.)

    #sig = np.zeros_like(spMod.ospec.err)+np.median(spMod.ospec.flux)/2.
    #mask = spMod.ospec.x < 5000.
    #err[mask] = spMod.ospec.err[mask]/10.
    y = pymc.Normal('y', mu=spModel,
                    tau=1./sig**2.,
                    value=spMod.ospec.flux,
                    observed=True)

    return locals()
