import setuptools
from Cython.Build import cythonize
import numpy

setuptools.setup(
    name='CRISPR_HMM',
    version='1.0.0',
    description='Sequence alignment of CRISPR mutation data with hidden Markov model.',
    url='https://github.com/JasperYH/CRISPR-HMM',
    author='Jingyuan Hu',
    install_requires=['numpy', 'cython', 'pathos'],
    author_email='jingyuan@ds.dfci.harvard.edu',
    packages=setuptools.find_packages(),
    zip_safe=False,
    ext_modules=cythonize("crispr_hmm/fast_viterbi.pyx"),
    setup_requires=['Cython', 'numpy'],  # Add setup_requires section
    include_dirs=[numpy.get_include()]
)
