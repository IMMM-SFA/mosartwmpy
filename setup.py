import re
from setuptools import setup, find_packages


def readme():
    """Return the contents of the project README file."""
    with open('README.md') as f:
        return f.read()


version = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", open('mosartwmpy/_version.py').read(), re.M).group(1)

setup(
    name='mosartwmpy',
    version=version,
    packages=find_packages(),
    url='https://github.com/IMMM-SFA/mosartwmpy',
    license='BSD2-Simplified',
    author='Travis Thurber',
    author_email='travis.thurber@pnnl.gov',
    description='Python implementation of MOSART-WM: A water routing and management model',
    long_description=readme(),
    long_description_content_type="text/markdown",
    python_requires='>=3.7.*, <4',
    include_package_data=True,
    install_requires=[
        'bmipy==2.0',
        'dask[complete]==2021.6.2',
        'h5netcdf==0.11.0',
        'matplotlib==3.4.2',
        'nc-time-axis==1.3.1',
        'netCDF4==1.5.7',
        'numba==0.53.1',
        'numpy==1.21.0',
        'pandas==1.3.0',
        'pathvalidate==2.4.1',
        'psutil==5.8.0',
        'pyarrow==4.0.1',
        'python-benedict==0.24.0',
        'regex==2021.7.6',
        'requests==2.25.1',
        'rioxarray==0.5.0',
        'tqdm==4.61.2',
        'xarray==0.18.2'
    ],
    extras_require={
        'dev': [
            'build~=0.5.1',
            'nbsphinx~=0.8.6',
            'recommonmark~=0.7.1',
            'setuptools~=57.0.0',
            'sphinx~=4.0.2',
            'sphinx-panels~=0.6.0',
            'sphinx-rtd-theme~=0.5.2',
            'twine~=3.4.1'
        ]
    }
)
