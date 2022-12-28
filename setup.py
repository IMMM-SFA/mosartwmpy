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
    entry_points={
        'console_scripts': [
            'create_grand_parameters = mosartwmpy.utilities.create_grand_parameters:create_grand_parameters',
            'bil_to_parquet = mosartwmpy.utilities.bil_to_parquet:bil_to_parquet',
        ]
    },
    install_requires=[
        'bmipy~=2.0',
        'click~=8.0.1',
        'contextily~=1.2.0',
        'dask[complete]~=2021.10.0',
        'geopandas~=0.10.2',
        'h5netcdf~=0.11.0',
        'hvplot~=0.7.3',
        'matplotlib~=3.4.3',
        'nc-time-axis~=1.4.0',
        'netCDF4~=1.5.7',
        'numba~=0.53.1',
        'numpy~=1.20.3',
        'pandas~=1.3.4',
        'pathvalidate~=2.5.0',
        'psutil~=5.8.0',
        'pyarrow~=6.0.0',
        'pyomo~=6.2',
        'python-benedict~=0.24.3',
        'regex~=2021.10.23',
        'requests~=2.26.0',
        'rioxarray~=0.8.0',
        'tqdm~=4.62.3',
        'xarray~=0.19.0'
    ],
    extras_require={
        'dev': [
            'build~=0.7.0',
            'nbsphinx~=0.8.7',
            'recommonmark~=0.7.1',
            'setuptools~=58.3.0',
            'sphinx~=4.2.0',
            'sphinx-panels~=0.6.0',
            'sphinx-rtd-theme~=1.0.0',
            'twine~=3.4.2'
        ]
    }
)
