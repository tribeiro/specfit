#! /usr/bin/env

'''
Convert Coelho files to numpy.
'''

import sys, os
import numpy as np
import logging as log


def main(argv):

    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option('-v', '--verbose',
                      help='Run in verbose mode.', action='store_true', default=False)
    parser.add_option('--wmin',
                      help='Minimum wavelenght.',
                      type=float)
    parser.add_option('--wmax',
                      help='Maximum wavelenght.',
                      type=float)

    opt, args = parser.parse_args(argv)

    if opt.verbose:
        log.basicConfig(level=log.INFO)

    for fname in args[1:]:
        sp = np.loadtxt(fname, unpack=True, usecols=(0, 1))
        # asort = sp[0].argsort()

        # rsp = np.array([sp[0][asort], 10 ** (sp[1][asort]) + 8.0])

        # mask = np.zeros(len(rsp[0]))
        # if opt.wmin:
        #     mask = np.bitwise_and(mask, mask < opt.wmin)
        # if opt.wmax:
        #     mask = np.bitwise_and(mask, mask > opt.wmax)

        # rsp = np.array([rsp[0], sp[1]])

        ofile = os.path.splitext(fname)[0] + '.npy'
        log.info('%s -> %s' % (fname, ofile))
        np.save(ofile, sp)

    return 0


if __name__ == '__main__':
    main(sys.argv)
