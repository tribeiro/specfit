from distutils.core import setup

setup(
    name='specfit',
    version='0.0.1',
    packages=['specfit', 'specfit.lib', 'specfit.lib.modelfactory'],
    scripts=['scripts/specfitMC.py', 'scripts/consolidate_res.py'],
    url='http://github.com/astroufsc/specfit',
    license='GPL v2',
    author='Tiago Ribeiro',
    author_email='tribeiro@ufs.br',
    description='Spectral fitting tools.'
)
