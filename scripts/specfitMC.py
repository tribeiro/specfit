#!/usr/bin/env python

'''
Using PyMCMC

C - Tiago Ribeiro
'''

import sys, os
import numpy as np
import scipy.stats
import matplotlib.pyplot as py
import pymc
import logging
from specfit.lib.modelfactory.singlecomb import SingleComb

logging.basicConfig(format='%(levelname)s:%(asctime)s::%(message)s',
                    level=logging.DEBUG)


def main(argv):
    from optparse import OptionParser

    parser = OptionParser()

    parser.add_option('-f', '--filename',
                      help='Input spectrum to fit.'
                      , type='string')
    parser.add_option('-o', '--output',
                      help='Output root name.'
                      , type='string')
    parser.add_option('-g', '--grid',
                      help='List of grid spectra.'
                      , type='string')
    parser.add_option('-t', '--template-list',
                      help='''List of list of grid spectra. Use when the
number of components to fit is larger than 1.'''
                      , type='string')
    parser.add_option('--niter',
                      help="Number of iterations (default=10000).",
                      type='int', default=10000)
    parser.add_option('--burn',
                      help="Number of burn iterations (default=2500).",
                      type='int', default=2500)
    parser.add_option('--thin',
                      help="Number of sub-iterations for each iteration (default=3).",
                      type='int', default=3)
    parser.add_option('--tune_interval',
                      help="Number of iterations for tuning (default=500).",
                      type='int', default=500)
    parser.add_option('--tune_throughout',
                      help='Run in verbose mode (default = False).', action='store_true',
                      default=False)
    parser.add_option('--templatetype',
                      help='''Type of template file. Choices are: 1 - Pickle,
2 - SDSSFits, 3 - Coelho 2014 model spectra''', type='int', default=1)
    parser.add_option('--no-overwrite',
                      help='Check if output file exists before writing to it. Save to a different file if it exists.',
                      action='store_true',
                      default=False)
    parser.add_option('-v', '--verbose',
                      help='Run in verbose mode.', action='store_true',
                      default=False)
    parser.add_option('--savefig',
                      help='Save png with output.', action='store_true',
                      default=False)
    parser.add_option('--show',
                      help='Show figure with output.', action='store_true',
                      default=False)
    opt, args = parser.parse_args(argv)

    threadId = os.getpid()

    dfile = opt.filename
    outfile = opt.output
    outroot = opt.output[:opt.output.rfind('.')]
    # tlist = [opt.grid]

    outfile = outroot + '_%010i.npy' % threadId
    dbname = outroot + '_%010i.pickle' % threadId
    spname = outroot + '_%010i.spres.npy' % threadId
    logname = outroot + '_%010i.log' % threadId
    logging.basicConfig(format='%(levelname)s:%(asctime)s::%(message)s',
                        level=logging.INFO,
                        filename=logname)

    if opt.no_overwrite and os.path.exists(dbname):
        logging.debug('File %s exists (running on "no overwrite" mode).' % dbname)
        index = 0
        dbname = outroot + '_%010i.%04i.pickle' % (threadId, index)
        while os.path.exists(dbname):
            dbname = outroot + '_%010i.%04i.pickle' % (threadId, index)
            index += 1
            logging.debug('%s' % dbname)
        fp = open(dbname, 'w')
        fp.close()

        logging.info('dbname: %s' % dbname)
    if opt.no_overwrite and os.path.exists(spname):
        logging.debug('File %s exists (running on "no overwrite" mode).' % spname)
        index = 0
        spname = outroot + '_%010i.spres.%04i.npy' % (threadId, index)
        while os.path.exists(spname):
            spname = outroot + '_%010i.spres.%04i.npy' % (threadId, index)
            index += 1
            logging.debug('%s' % spname)
        fp = open(spname, 'w')
        fp.close()

        logging.info('spname: %s' % spname)

    if opt.no_overwrite and os.path.exists(outfile):
        logging.debug('File %s exists (running on "no overwrite" mode).' % outfile)
        index = 0
        outroot = outfile[:opt.filename.rfind('.')]
        outfile = outroot + '_%010i.%04i.npy' % (threadId, index)
        while os.path.exists(outfile):
            outfile = outroot + '_%010i.%04i.npy' % (threadId, index)
            index += 1
            logging.debug('%s' % outfile)

        fp = open(outfile, 'w')
        fp.close()

    logging.info('outfile: %s' % outfile)

    csvname = outroot + '_%010i.csv' % (threadId)
    plotname = outroot + '_%010i' % (threadId)
    if opt.no_overwrite and os.path.exists(csvname):
        logging.debug('File %s exists (running on "no overwrite" mode).' % csvname)
        index = 0
        csvname = outroot + '_%010i.%04i.csv' % (threadId, index)
        while os.path.exists(csvname):
            csvname = outroot + '_%010i.%04i.csv' % (threadId, index)
            index += 1
            logging.debug('%s' % csvname)

        fp = open(csvname, 'w')
        fp.close()

        logging.info('csvname: %s' % csvname)

    logging.info('Preparing model...')

    tlist = np.loadtxt(opt.template_list, dtype='S')

    M = pymc.MCMC(SingleComb(2,
                             dfile,
                             tlist,
                             opt.templatetype), db='ram')#,
                  #db='pickle', dbname=dbname)

    # FIXME: This should not be hard coded!
    sd_ = [2, 2, 1, 1, 1   ]

    # for i in range(4):
    # Todo: Generalize to spMod.nspec
    M.use_step_method(pymc.DiscreteMetropolis, M.template1, proposal_sd=sd_[0], proposal_distribution='Normal')
    # M.use_step_method(pymc.DiscreteMetropolis, M.template1[1], proposal_sd=sd_[1], proposal_distribution='Normal')
    M.use_step_method(pymc.DiscreteMetropolis, M.template2, proposal_sd=sd_[2], proposal_distribution='Normal')
    # M.use_step_method(pymc.DiscreteMetropolis, M.template2[1], proposal_sd=sd_[3], proposal_distribution='Normal')

    logging.info('Model done...')

    logging.info('Starting sampler...')

    NITER = opt.niter * opt.thin
    NBURN = opt.burn * opt.thin
    if NBURN > NITER:
        NITER += NBURN
    M.sample(iter=NITER, burn=NBURN, thin=opt.thin,
             tune_interval=opt.tune_interval, tune_throughout=opt.tune_throughout) #, dbname=dbname)  # ,
    # verbose=-1)

    logging.info('Sampler done. Saving results...')

    M.db.close()

    # print M.trace('template1')[:]
    dtype = [('sig', '<f8'), ('tempscale1', '<f8'), ('tempscale2', '<f8')]

    for i in range(M.ncomp):
        for j in range(M.spMod.grid_ndim[i]):
            dtype.append(('template%i_%i' % (i+1, j), '<i4'))

    oarray = np.zeros(len(M.trace('template1')[:]), dtype=dtype)

    for i in range(M.ncomp):
        # index = np.zeros(M.spMod.grid_ndim[i], dtype=np.int)
        temp_ =  M.trace('template%i' % (i+1))[:].T
        for j in range(M.spMod.grid_ndim[i]):
            oarray['template%i_%i' % (i+1, j)] = temp_[j]
            # index[j] = np.int(scipy.stats.mode(temp_[j])[0][0])
        # M.spMod.ntemp[i] = M.spMod.Grid[i].item(*index)

    for idx in range(2):
        for icomp in range(len(oarray['template1_0'])):
            index = np.zeros(M.spMod.grid_ndim[idx], dtype=int)
            for iidx in range(M.spMod.grid_ndim[idx]):
                index[iidx] = oarray['template%i_%i' % (idx+1, iidx)][icomp]
            ntemp = M.spMod.Grid[idx].item(*index)
            oarray['tempscale%i' % (idx + 1)][icomp] = M.spMod.templateScale[idx][ntemp]

    np.save(outfile, oarray)

    M.write_csv(csvname)

    logging.info('Done')

    pymc.Matplot.plot(M)
    pymc.Matplot.autocorrelation(M)
    py.show()

    return 0


################################################################################

if __name__ == '__main__':
    main(sys.argv)

################################################################################
