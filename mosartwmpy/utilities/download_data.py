import logging
import os
from pathlib import Path
import importlib.resources
import requests
import sys
import tempfile
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

# bytes read from the network per chunk on either transport; 1 MiB keeps the progress bar to a few
# hundred updates for the 700 MB sample archive rather than the 175,000 that 4 KiB chunks produced
CHUNK_SIZE = 1024 * 1024

# seconds to wait for the server to respond on the plain HTTP transport (the MSD-LIVE transport sets
# its own in msdlive.py); without one a stalled connection hangs the download indefinitely
REQUEST_TIMEOUT = 120

# this module logs through its own logger and never adds to or removes from the root logger: the
# download runs inside notebooks and coupled drivers whose root handlers (a model's log file, for
# instance) must survive it. when nothing has configured logging at all, as in the standalone
# `mosartwmpy.download` CLI, InstallSupplement attaches a stdout handler to this logger for the
# duration of the download so progress is still printed.
logger = logging.getLogger(__name__)


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

        self.console_handler = None
        self.initialize_logger()
        self.destination = self.valid_directory(destination)
        self.url = url
        self.filename = filename

    def initialize_logger(self):
        """Print progress to stdout when nothing else has configured logging.

        If the process already has logging configured (a root handler exists, as it does once a
        model is running or a notebook has set up logging), messages simply propagate to those
        handlers and nothing is added. Otherwise a stdout handler is attached to this module's
        logger, not to the root logger, and removed again by `close_logger`.
        """

        if logging.getLogger().hasHandlers():
            return
        logger.setLevel(logging.INFO)
        self.console_handler = logging.StreamHandler(sys.stdout)
        self.console_handler.setLevel(logging.INFO)
        self.console_handler.setFormatter(logging.Formatter(""))
        logger.addHandler(self.console_handler)

    def close_logger(self):
        """Remove the stdout handler this instance attached, if any; other handlers are left alone."""

        if self.console_handler is not None:
            logger.removeHandler(self.console_handler)
            self.console_handler.close()
            self.console_handler = None

    def valid_directory(self, directory):
        """Ensure the provided directory exists."""

        if os.path.isdir(directory):
            return directory
        else:
            msg = f"The write directory provided by the user does not exist: {directory}"
            logger.error(msg)
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
        msg = f"Unable to download data from {self.url}"
        try:
            logger.info(f"Downloading example data from {self.url}")
            # a timeout so a stalled server cannot hang the download forever, and raise_for_status so
            # an HTTP error is reported as such instead of its HTML body failing later as a bad zip
            r = requests.get(self.url, stream=True, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            self._unpack(
                chunks=r.iter_content(chunk_size=CHUNK_SIZE),
                total=int(r.headers.get('content-length', 0)),
                description=self.url,
            )

        except requests.exceptions.RequestException:
            logger.exception(msg)
            self.close_logger()
            raise
        except zipfile.BadZipFile as e:
            # a 200 response that is not a zip archive, i.e. an error page served with a success status
            logger.exception(msg)
            self.close_logger()
            raise zipfile.BadZipFile(f"{msg}: the response is not a zip archive ({e})") from e

    def fetch_msdlive(self):
        """Download and unpack an example data supplement from a MSD-LIVE record.

        Files on newer MSD-LIVE records are not reachable over plain HTTP, so this
        requests anonymous read only credentials and signs a request against the
        record's S3 access point. No MSD-LIVE account is required.
        """

        msg = f"Unable to download data from {self.url}"
        try:
            record_id = record_id_from_url(self.url)
            logger.info(f"Downloading example data from MSD-LIVE record {record_id}")

            credentials = get_anonymous_credentials()
            archive = find_archive(record_id, filename=self.filename, credentials=credentials)
            response = open_record_file_stream(record_id, archive['key'], credentials=credentials)

            self._unpack(
                chunks=response.iter_content(chunk_size=CHUNK_SIZE),
                total=archive['size'],
                description=archive['key'].rsplit('/', 1)[-1],
            )

        except (requests.exceptions.RequestException, FileNotFoundError, ValueError, KeyError):
            logger.exception(msg)
            self.close_logger()
            raise
        except zipfile.BadZipFile as e:
            logger.exception(msg)
            self.close_logger()
            raise zipfile.BadZipFile(f"{msg}: the response is not a zip archive ({e})") from e

    def _unpack(self, chunks, total, description):
        """Stream a zip archive from an iterator of chunks to a temporary file and extract it to the destination.

        The archive is spooled to disk rather than held in memory: the sample data archive is
        about 700 MB, and buffering it in memory needed heap roughly equal to the archive plus
        its largest member.

        Args:
            chunks (iterable): iterator yielding bytes of the archive
            total (int): expected size in bytes, for the progress bar
            description (str): label for the progress bar
        """

        try:
            # spool into the destination directory rather than the system temporary directory:
            # TMPDIR on compute nodes is often a small scratch or RAM-backed tmpfs, while the
            # destination is already known to hold room for the extracted data
            with tempfile.TemporaryFile(dir=self.destination) as stream:
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
                stream.seek(0)
                with zipfile.ZipFile(stream) as zipped:
                    # extract each file in the zipped dir to the project
                    for f in zipped.namelist():
                        logger.info("Unzipped: {}".format(os.path.join(self.destination, f)))
                        zipped.extract(f, self.destination)

            logger.info("Download and install complete.")

        finally:
            # the temporary file is removed by its context manager; this releases the stdout
            # handler, if one was attached, whether or not the download and extraction succeeded
            self.close_logger()
