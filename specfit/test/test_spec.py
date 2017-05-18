from matplotlib import pyplot as plt
from StringIO import StringIO
import numpy as np
from specfit.lib.specfit import SpecFit
import logging

logging.basicConfig(format='%(levelname)s:%(asctime)s::%(message)s',
                    level=logging.DEBUG)

sdss_test_spec = 'spec-2826-54389-0590.fits'
templist = ['data/gridWD_foo.lis', 'data/GridNextGen-3.0a+0.4.lis']
templist = ['gridWD.lis', 'GridNextGen_3D_001.lis']

spMod = SpecFit(2)
spMod.loadSDSSFits(sdss_test_spec, True)
temptype = 1

for i in range(len(templist)):
    fp = open(templist[i], 'r')
    line1 = np.loadtxt(StringIO(fp.readline()), dtype='S')
    spMod.grid_ndim[i] = len(line1) - 1  # Set grid dimension for 1st component

    logging.debug('Component %s[%i] with %i dimentions.' % (templist[i], i, spMod.grid_ndim[i]))

    if temptype == 1:
        spMod.loadPickleTemplate(i, templist[i])
    elif temptype == 2:
        spMod.loadSDSSFitsTemplate(i, templist[i])
    elif temptype == 3:
        spMod.loadCoelhoTemplate(i, templist[i])


spMod.normTemplate(0, 5500., 9000.)
spMod.normTemplate(1, 5500., 9000.)
spMod.setAutoProp(False)
spMod.preprocTemplate()

for n in range(2):
    spMod.gridSpec(n)

# spMod.ntemp[0] = spMod.Grid[0][5][4]
# spMod.ntemp[1] = spMod.Grid[1][2][0]
#
# spMod.ntemp[0] = spMod.Grid[0][28][12]
# spMod.ntemp[1] = spMod.Grid[1][3][0]
#
# spMod.ntemp[0] = spMod.Grid[0][8][6]
# spMod.ntemp[1] = spMod.Grid[1][1][1]
#
# spMod.ntemp[0] = spMod.Grid[0][53][14]
# spMod.ntemp[1] = spMod.Grid[1][2][1]

plt.fill_between(spMod.ospec.x, spMod.ospec.flux+spMod.ospec.err, spMod.ospec.flux-spMod.ospec.err, color='0.8')

spMod.ntemp[0] = spMod.Grid[0][48][12]
spMod.ntemp[1] = spMod.Grid[1][8][10][2]
p = spMod.fit()
for i in range(len(p)):
    print '%f ' % p[i],
print
# spMod.scale[0][spMod.ntemp[0]] = 0.0
modsp = spMod.modelSpec()
plt.plot(modsp.x, modsp.flux, 'b')


# spMod.ntemp[0] = spMod.Grid[0][51][11]
# spMod.ntemp[1] = spMod.Grid[1][7][9]
# p = spMod.fit()
# for i in range(len(p)):
#     print '%f ' % p[i],
# print
# #
# # # spMod.scale[0][spMod.ntemp[0]] = 0.0
# modsp = spMod.modelSpec()
# plt.plot(modsp.x, modsp.flux, 'g')
#
# spMod.ntemp[0] = spMod.Grid[0][8][9]
# spMod.ntemp[1] = spMod.Grid[1][1][1]
# p = spMod.fit()
# for i in range(len(p)):
#     print '%f ' % p[i],
# print
# # # spMod.scale[0][spMod.ntemp[0]] = 0.0
# modsp = spMod.modelSpec()
# plt.plot(modsp.x, modsp.flux,'r')

# spMod.scale[0][0] = 1.0
# spMod.scale[1][0] = 1.0
# spMod.vel[0] = 0.0
# spMod.vel[1] = 0.0

# plt.plot(spMod.ospec.x, spMod.ospec.flux)

# plt.plot(modsp2.x, modsp2.flux)

plt.show()
