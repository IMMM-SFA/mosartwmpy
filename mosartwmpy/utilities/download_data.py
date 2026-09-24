import io
import logging
import os
from pathlib import Path
import importlib.resources
import requests
import sys
from tqdm import tqdm
import zipfile

from benedict import benedict

from mosartwmpy.utilities.msdlive import (
    find_archive,
    is_msdlive_record_url,
    get_anonymous_credentials,
    open_record_file_stream,
    record_id_from_url,
)


def download_data(dataset: str, destination: str = None, manifest: str = str(importlib.resources.files('mosartwmpy').joinpath('data_manifest.yaml'))) -> None:
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
        destination=destination if destination is not None else Path(data_dictionary.get(f'{dataset}.destination', './')),
        filename=data_dictionary.get(f'{dataset}.filename'),
    )
    get.fetch()


class InstallSupplement:
    """Download and unpack example data supplement from Zenodo that matches the current installed
    distribution.

    :param example_data_directory:              Full path to the directory you wish to install
                                                the example data to.  Must be write-enabled
                                                for the user.

    """

    def __init__(self, url, destination, filename=None):

        self.initialize_logger()
        self.destination = self.valid_directory(destination)
        self.url = url
        self.filename = filename

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

    def fetch(self):
        """Download and unpack the example data supplement for the current distribution.

        Datasets hosted on MSD-LIVE record urls are fetched through a signed S3 access
        point, since those files are not served over plain HTTP; everything else is
        downloaded directly.
        """

        if is_msdlive_record_url(self.url):
            self.fetch_msdlive()
        else:
            self.fetch_zenodo()

    def fetch_zenodo(self):
        """Download and unpack an example data supplement served over plain HTTP."""

        # retrieve content from URL
        try:
            logging.info(f"Downloading example data from {self.url}")
            r = requests.get(self.url, stream=True)
            self._unpack(
                chunks=r.iter_content(chunk_size=4096),
                total=int(r.headers.get('content-length', 0)),
                description=self.url,
            )

        except requests.exceptions.MissingSchema:
            msg = f"Unable to download data from {self.url}"
            logging.exception(msg)
            self.close_logger()
            raise

    def fetch_msdlive(self):
        """Download and unpack an example data supplement from a MSD-LIVE record.

        Files on newer MSD-LIVE records are not reachable over plain HTTP, so this
        requests anonymous read only credentials and signs a request against the
        record's S3 access point. No MSD-LIVE account is required.
        """

        try:
            record_id = record_id_from_url(self.url)
            logging.info(f"Downloading example data from MSD-LIVE record {record_id}")

            credentials = get_anonymous_credentials()
            archive = find_archive(record_id, filename=self.filename, credentials=credentials)
            response = open_record_file_stream(record_id, archive['key'], credentials=credentials)

            self._unpack(
                chunks=response.iter_content(chunk_size=1024 * 1024),
                total=archive['size'],
                description=archive['key'].rsplit('/', 1)[-1],
            )

        except (requests.exceptions.RequestException, FileNotFoundError, ValueError, KeyError):
            msg = f"Unable to download data from {self.url}"
            logging.exception(msg)
            self.close_logger()
            raise

    def _unpack(self, chunks, total, description):
        """Buffer a zip archive from an iterator of chunks and extract it to the destination.

        Args:
            chunks (iterable): iterator yielding bytes of the archive
            total (int): expected size in bytes, for the progress bar
            description (str): label for the progress bar
        """

        with io.BytesIO() as stream:
            with tqdm.wrapattr(
                stream,
                'write',
                file=sys.stdout,
                miniters=1,
                desc=description,
                total=total,
            ) as file:
                for chunk in chunks:
                    file.write(chunk)
            with zipfile.ZipFile(stream) as zipped:
                # extract each file in the zipped dir to the project
                for f in zipped.namelist():
                    logging.info("Unzipped: {}".format(os.path.join(self.destination, f)))
                    zipped.extract(f, self.destination)

        logging.info("Download and install complete.")

        self.close_logger()
