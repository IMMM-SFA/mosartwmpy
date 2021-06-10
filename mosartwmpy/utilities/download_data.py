import io
import logging
import os
from pathlib import Path
import pkg_resources
import requests
import sys
from tqdm import tqdm
import zipfile

from benedict import benedict


def download_data(dataset: str, destination: str = None, manifest: str = pkg_resources.resource_filename('mosartwmpy', 'data_manifest.yaml')) -> None:
    """Convenience wrapper for the InstallSupplement class.
    
    Download and unpack example data supplement from Zenodo that matches the current installed
    distribution.
    
    Args:
        dataset (str): name of the dataset to download, as found in the data_manifest.yaml
        destination (str): full path to the directory in which to unpack the downloaded files; must be write enabled; defaults to the directory listed in the manifest
        manifest (str): full path to the manifest yaml file describing the available downloads; defaults to the bundled data_manifest.yaml
    """

    data_dictionary = benedict(manifest, format='yaml')
    
    if not data_dictionary.get(dataset, None):
        raise Exception(f'Dataset "{dataset}" not found in the manifest ({manifest}).')
    
    get = InstallSupplement(
        url=data_dictionary.get(f'{dataset}.url'),
        destination=destination if destination is not None else Path(data_dictionary.get(f'{dataset}.destination', './'))
    )
    get.fetch_zenodo()


class InstallSupplement:
    """Download and unpack example data supplement from Zenodo that matches the current installed
    distribution.

    :param example_data_directory:              Full path to the directory you wish to install
                                                the example data to.  Must be write-enabled
                                                for the user.

    """

    def __init__(self, url, destination):

        self.initialize_logger()
        self.destination = self.valid_directory(destination)
        self.url = url

    def initialize_logger(self):
        """Initialize logger to stdout."""

        # initialize logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # logger console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(""))
        logger.addHandler(console_handler)

    @staticmethod
    def close_logger():
        """Shutdown logger."""

        # Remove logging handlers
        logger = logging.getLogger()

        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        logging.shutdown()

    def valid_directory(self, directory):
        """Ensure the provided directory exists."""

        if os.path.isdir(directory):
            return directory
        else:
            msg = f"The write directory provided by the user does not exist: {directory}"
            logging.exception(msg)
            self.close_logger()
            raise NotADirectoryError(msg)

    def fetch_zenodo(self):
        """Download and unpack the Zenodo example data supplement for the
        current distribution."""

        # retrieve content from URL
        try:
            logging.info(f"Downloading example data from {self.url}")
            r = requests.get(self.url, stream=True)
            with io.BytesIO() as stream:
                with tqdm.wrapattr(
                    stream,
                    'write',
                    file=sys.stdout,
                    miniters=1,
                    desc=self.url,
                    total=int(r.headers.get('content-length', 0))
                ) as file:
                    for chunk in r.iter_content(chunk_size=4096):
                        file.write(chunk)
                with zipfile.ZipFile(stream) as zipped:
                    # extract each file in the zipped dir to the project
                    for f in zipped.namelist():
                        logging.info("Unzipped: {}".format(os.path.join(self.destination, f)))
                        zipped.extract(f, self.destination)

            logging.info("Download and install complete.")

            self.close_logger()

        except requests.exceptions.MissingSchema:
            msg = f"Unable to download data from {self.url}"
            logging.exception(msg)
            self.close_logger()
            raise
