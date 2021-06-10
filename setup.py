from setuptools import setup, find_packages


def readme():
    """Return the contents of the project README file."""
    with open('README.md') as f:
        return f.read()


setup(
    name='mosartwmpy',
    version='0.0.6',
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
        'dask[complete]==2021.5.1',
        'matplotlib==3.4.2',
        'nc-time-axis==1.2.0',
        'netCDF4==1.5.6',
        'numexpr==2.7.3',
        'numpy==1.20.3',
        'pandas==1.2.4',
        'pathvalidate==2.3.0',
        'psutil==5.8.0',
        'pyarrow==4.0.1',
        'python-benedict==0.24.0',
        'regex==2021.4.4',
        'requests==2.25.1',
        'rioxarray==0.4.2',
        'tqdm==4.60.0',
        'xarray==0.18.2'
    ],
    extras_require={
        'dev': [
            'build==0.4.0',
            'recommonmark==0.7.1',
            'setuptools==54.1.2',
            'sphinx==3.5.2',
            'sphinx-rtd-theme==0.5.1',
            'twine==3.4.1'
        ]
    }
)
