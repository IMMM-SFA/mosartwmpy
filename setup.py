from setuptools import setup, find_packages


def readme():
    """Return the contents of the project README file."""
    with open('README.md') as f:
        return f.read()


def get_requirements():
    """Return a list of package requirements from the requirements.txt file."""
    with open('requirements.txt') as f:
        return f.read().split()


setup(
    name='mosartwmpy',
    version='0.0.5',
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
    install_requires=get_requirements(),
    extras_require={
        'dev': ['build==0.4.0', 'recommonmark==0.7.1', 'setuptools==54.1.2', 'sphinx==3.5.2', 'sphinx-rtd-theme==0.5.1', 'twine==3.4.1']
    }
)
