'''
PyMCMC model factory.

'''

import os
import numpy as np
import pymc
from StringIO import StringIO
import logging
import scipy.constants
from astropysics import spec
from specfit.lib.specfit import SpecFit
import logging

_c_kms = scipy.constants.c / 1.e3  # Speed of light in km s^-1

__ncomp__ = 2

fileHandler = logging.handlers.RotatingFileHandler("singlecomb.log",
                                                       maxBytes=100 *
                                                       1024 * 1024,
                                                       backupCount=10)

# _log_handler = logging.FileHandler(fileHandler)
fileHandler.setFormatter(logging.Formatter(fmt='%(asctime)s[%(levelname)s:%(threadName)s]-%(name)s-'
                                               '(%(filename)s:%(lineno)d):: %(message)s'))
fileHandler.setLevel(logging.DEBUG)

log = logging.getLogger('singlecomb')
log.addHandler(fileHandler)

def SingleComb(ncomp, ofile, templist, temptype=1):
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

    spMod = SpecFit(ncomp)

    # Load observed spectra

    if ofile.rfind('.fits') > 0:
        spMod.loadSDSSFits(ofile, True)
    elif ofile.rfind('.npy') > 0:
        spMod.loadPickle(ofile)
    else:
        raise IOError('Cannot read %s data type. Only SDSS type fits and numpy (pickle) \
                      files are suported.' % (ofile))

    # Load template spectra, for each component

    for i in range(ncomp):
        fp = open(templist[i], 'r')
        line1 = StringIO(fp.readline())
        line1 = np.loadtxt(line1, dtype='S')
        spMod.grid_ndim[i] = len(line1) - 1  # Set grid dimension for 1st component

        logging.debug('Component %s[%i] with %i dimentions.' % (templist[i], i, spMod.grid_ndim[i]))

        if temptype == 1:
            spMod.loadPickleTemplate(i, templist[i])
        elif temptype == 2:
            spMod.loadSDSSFitsTemplate(i, templist[i])
        elif temptype == 3:
            spMod.loadCoelhoTemplate(i, templist[i])

    # Pre-process template files

    # spMod.normTemplate(0,5500.,5520.) # TODO: Is this a suitable range?
    # spMod.normTemplate(1,5500.,5520.)
    spMod.normTemplate(0, 5500., 9000.)  # TODO: Is this a suitable range?
    spMod.normTemplate(1, 5500., 9000.)

    spMod.setAutoProp(False)

    spMod.preprocTemplate()

    for n in range(__ncomp__):
        spMod.gridSpec(n)

    gridmin = np.zeros(np.sum(spMod.grid_ndim), dtype=int)
    gridmax = np.zeros(np.sum(spMod.grid_ndim), dtype=int)

    for i in range(ncomp):
        logging.debug('gridmax: %i %i %i %i' % (i * 2, i * 2 + 1,
                                                len(spMod.Grid[i]) - 1,
                                                len(spMod.Grid[i][0]) - 1))
        for j in range(spMod.grid_ndim[i]):
            gridmax[i * 2 + j] = spMod.Grid[i].shape[j] - 1

    # template = pymc.Container([pymc.DiscreteUniform('template_%02i' % (i + 1),
    #                                  lower=gridmin[i],
    #                                  upper=gridmax[i],
    #                                  value=int((gridmin[i]+gridmax[i])/2)) for i in range(np.sum(spMod.grid_ndim))])

    # Todo: Find a way to generalize the number of template components to spMod.nspec
    template1 = pymc.DiscreteUniform('template1',
                                     lower=gridmin[:spMod.grid_ndim[0]],
                                     upper=gridmax[:spMod.grid_ndim[0]],
                                     value=(gridmin[:spMod.grid_ndim[0]]+gridmax[:spMod.grid_ndim[0]])/2,
                                     size=spMod.grid_ndim[0])

    template2 = pymc.DiscreteUniform('template2',
                                     lower=gridmin[spMod.grid_ndim[0]:],
                                     upper=gridmax[spMod.grid_ndim[0]:],
                                     value=(gridmin[spMod.grid_ndim[0]:]+gridmax[spMod.grid_ndim[0]:])/2,
                                     size=spMod.grid_ndim[1])

    # Prepare PyMC stochastic variables

    @pymc.deterministic
    def spModel(template1=template1, template2=template2):

        log_str = ''
        templates = [template1, template2]
        for idx in range(len(spMod.grid_ndim)):

            index = np.zeros(spMod.grid_ndim[idx], dtype=int)
            for iidx in range(spMod.grid_ndim[idx]):
                index[iidx] = templates[idx][iidx]
                log_str += '%i ' % templates[idx][iidx]
            spMod.ntemp[idx] = spMod.Grid[idx].item(*index)
        log.info(log_str)
        # log.debug(' %04i %04i %04i %04i' % (template1[0], template1[1], template2[0], template2[1]))
        # spMod.ntemp[0] = spMod.Grid[0][template1[0]][template1[1]]
        # spMod.ntemp[1] = spMod.Grid[1][template2[0]][template2[1]]
        pres = spMod.fit()
        # print
        # print '- C1', spMod.ntemp[0], pres[0], pres[1]
        # print '- C2', spMod.ntemp[1], pres[2], pres[3]
        # Todo: Generalize to spMod.nspec
        spMod.scale[0][spMod.ntemp[0]] = pres[0]
        spMod.scale[1][spMod.ntemp[1]] = pres[1]
        spMod.vel[0] = 0. #pres[0]
        spMod.vel[1] = 0. #pres[2]

        modspec = spMod.modelSpec()
        return modspec.flux

    mflux = np.mean(spMod.ospec.flux)
    # sig = pymc.Uniform('sig',
    #                    np.median(spMod.ospec.flux)/100.,
    #                    np.median(spMod.ospec.flux)/10.,
    #                    value=np.zeros_like(spMod.ospec.flux)+np.median(spMod.ospec.flux) / 25.)
    sig = pymc.Uniform('sig',
                       np.min(spMod.ospec.err)*0.9,
                       np.max(spMod.ospec.err)*1.1,
                       value=spMod.ospec.err)

    # sig = spMod.ospec.err #np.zeros_like(spMod.ospec.err)+np.median(spMod.ospec.flux)/25.
    # mask = spMod.ospec.x < 5000.
    # err[mask] = spMod.ospec.err[mask]/10.
    y = pymc.Normal('y', mu=spModel,
                    tau=1. / sig**2.,
                    value=spMod.ospec.flux,
                    observed=True)

    return locals()
