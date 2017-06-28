#!/usr/bin/env python

'''
Take the trace results from specfitMC and calculates the proper scaling factor for each different solution.

C - Tiago Ribeiro
'''

import sys, os
import numpy as np
import pymc
import logging
from astropy.io import ascii
from StringIO import StringIO
import numpy as np
from specfit.lib.specfit import SpecFit

logging.basicConfig(format='%(levelname)s:%(asctime)s::%(message)s',
                    level=logging.DEBUG)


def main(argv):
    from optparse import OptionParser

    parser = OptionParser()

    parser.add_option('-f', '--filename',
                      help='Input list of traces to consolidate.'
                      , type='string')
    parser.add_option('-t', '--template-list',
                      help='''List of list of grid spectra. Use when the
                      number of components to fit is larger than 1. This must be the same list used in the fit
                      and must be accessible to the script.'''
                      , type='string')

    opt, args = parser.parse_args(argv)

    trace_list = ascii.read(opt.filename)

    templist = np.loadtxt(opt.template_list, dtype='S')

    spMod = SpecFit(2)
    spMod.loadSDSSFits(trace_list['DATA'][0], True)
    temptype = 1

    for itemp in range(len(templist)):
        fp = open(templist[itemp], 'r')
        line1 = np.loadtxt(StringIO(fp.readline()), dtype='S')
        spMod.grid_ndim[itemp] = len(line1) - 1  # Set grid dimension for 1st component

        logging.debug('Component %s[%i] with %i dimentions.' % (templist[itemp], itemp, spMod.grid_ndim[itemp]))

        if temptype == 1:
            spMod.loadPickleTemplate(itemp, templist[itemp])
        elif temptype == 2:
            spMod.loadSDSSFitsTemplate(itemp, templist[itemp])
        elif temptype == 3:
            spMod.loadCoelhoTemplate(itemp, templist[itemp])


    spMod.normTemplate(0, 5500., 9000.)
    spMod.normTemplate(1, 5500., 9000.)
    spMod.setAutoProp(False)
    spMod.preprocTemplate()

    for n in range(2):
        spMod.gridSpec(n)

    for i in range(len(trace_list['DATA'])):
        spMod.loadSDSSFits(trace_list['DATA'][i], True)

        # Reading trace
        trace = np.load(trace_list['TRACE'][i])
        new_dtype = trace.dtype.descr
        new_dtype.append(('theta1', np.float))
        new_dtype.append(('theta2', np.float))

        new_trace = np.zeros(len(trace), dtype=new_dtype)

        grid_dict = {}
        for itrace in range(len(trace)):
            trace_str = '%i.%i_%i.%i.%i' % (trace['template1_0'][itrace], trace['template1_1'][itrace],
                                            trace['template2_0'][itrace], trace['template2_1'][itrace],
                                            trace['template2_2'][itrace])

            if trace_str not in grid_dict.keys():
                print 'Fitting %s' % trace_str
                ii, ij, ik = trace['template2_0'][itrace], trace['template2_1'][itrace], trace['template2_2'][itrace]
                spMod.ntemp[0] = spMod.Grid[0][trace['template1_0'][itrace]][trace['template1_1'][itrace]]
                spMod.ntemp[1] = spMod.Grid[1][ii][ij][ik]
                p = spMod.fit()

                new_trace[itrace] = tuple(trace[itrace]) + tuple(p)
                grid_dict[trace_str] = tuple(p)
            else:
                new_trace[itrace] = tuple(trace[itrace]) + grid_dict[trace_str]

        np.save(trace_list['TRACE'][i].replace('.npy','_c.npy'), new_trace)


    return 0


################################################################################

if __name__ == '__main__':
    main(sys.argv)

################################################################################
