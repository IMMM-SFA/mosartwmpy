import io
import logging
import sys
import tracemalloc
import unittest
import zipfile

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from mosartwmpy.utilities import download_data, msdlive
from mosartwmpy.utilities.download_data import InstallSupplement


# the placeholder is listed first, as S3 returns keys in lexical order, so picking
# the first entry rather than the archive would silently download 12 bytes
LIST_RESPONSE = """<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Name>msdlive-project-im3</Name>
  <Prefix>abcde-12345/</Prefix>
  <IsTruncated>false</IsTruncated>
  <Contents>
    <Key>abcde-12345/dummy.txt</Key>
    <Size>12</Size>
  </Contents>
  <Contents>
    <Key>abcde-12345/sample_data.zip</Key>
    <Size>702662166</Size>
  </Contents>
</ListBucketResult>
"""

CREDENTIALS = {
    'region': 'us-west-2',
    'accountId': '889772541283',
    'credentials': {
        'accessKeyId': 'ASIAEXAMPLE',
        'secretKey': 'secret',
        'sessionToken': 'token',
        'expiration': 0,
    },
}


class FakeResponse:
    """Stand-in for a requests.Response."""

    def __init__(self, text='', content=b'', status_code=200):
        self.text = text
        self._content = content
        self.status_code = status_code
        self.headers = {'content-length': str(len(content))}
        self.chunk_sizes = []

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f'status {self.status_code}')

    def iter_content(self, chunk_size=1):
        self.chunk_sizes.append(chunk_size)
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i:i + chunk_size]

    @property
    def content(self):
        return self._content or self.text.encode('utf-8')


class RecordUrlTest(unittest.TestCase):
    """Test recognizing MSD-LIVE record urls."""

    MSDLIVE_URLS = [
        'https://data.msdlive.org/records/m28qs-54544',
        'https://data.msdlive.org/records/m28qs-54544/files/a.zip?download=1',
        'https://data.msdlive.org/api/records/m6pp5-7xt54/files/a.zip/content',
    ]

    OTHER_URLS = [
        'https://zenodo.org/record/6959736/files/mosartwmpy_tutorial_1981_05.zip?download=1',
        'https://zenodo.org/record/5609702/files/mosartwmpy_validation.zip?download=1',
        'https://example.com/data.zip',
    ]

    def test_recognizes_record_urls(self):
        for url in self.MSDLIVE_URLS:
            self.assertTrue(msdlive.is_msdlive_record_url(url), url)

    def test_ignores_other_urls(self):
        for url in self.OTHER_URLS:
            self.assertFalse(msdlive.is_msdlive_record_url(url), url)

    def test_ignores_none(self):
        self.assertFalse(msdlive.is_msdlive_record_url(None))

    def test_extracts_record_id(self):
        self.assertEqual(msdlive.record_id_from_url(self.MSDLIVE_URLS[0]), 'm28qs-54544')
        self.assertEqual(msdlive.record_id_from_url(self.MSDLIVE_URLS[1]), 'm28qs-54544')
        self.assertEqual(msdlive.record_id_from_url(self.MSDLIVE_URLS[2]), 'm6pp5-7xt54')

    def test_raises_without_record_id(self):
        with self.assertRaises(ValueError):
            msdlive.record_id_from_url('https://zenodo.org/record/6959736/files/a.zip')


class SignedRequestTest(unittest.TestCase):
    """Test the SigV4 signing, without contacting AWS."""

    def signed(self, **kwargs):
        with patch.object(msdlive.requests, 'request', return_value=FakeResponse()) as mock:
            msdlive._signed_request(CREDENTIALS, 'abcde-12345', **kwargs)
        return mock.call_args

    def test_targets_the_record_access_point(self):
        args, kwargs = self.signed(key='abcde-12345/sample_data.zip')
        self.assertEqual(
            args[1],
            'https://abcde-12345-889772541283.s3-accesspoint.us-west-2.amazonaws.com'
            '/abcde-12345/sample_data.zip',
        )

    def test_sends_required_headers(self):
        _, kwargs = self.signed(key='abcde-12345/sample_data.zip')
        headers = kwargs['headers']
        self.assertEqual(headers['x-amz-security-token'], 'token')
        self.assertEqual(headers['x-amz-content-sha256'], msdlive.UNSIGNED_PAYLOAD)
        self.assertIn('x-amz-date', headers)
        self.assertTrue(headers['Authorization'].startswith(msdlive.ALGORITHM))
        self.assertIn('Credential=ASIAEXAMPLE/', headers['Authorization'])
        # every signed header must be declared, and only signed headers may be listed
        self.assertIn(
            'SignedHeaders=host;x-amz-content-sha256;x-amz-date;x-amz-security-token',
            headers['Authorization'],
        )

    def test_extra_headers_are_passed_through(self):
        _, kwargs = self.signed(key='a/b.zip', headers={'Range': 'bytes=0-3'})
        self.assertEqual(kwargs['headers']['Range'], 'bytes=0-3')

    def test_query_is_sorted_into_the_url(self):
        args, _ = self.signed(query={'prefix': 'abcde-12345/', 'list-type': '2'})
        # canonical query must be sorted, so list-type precedes prefix
        self.assertIn('?list-type=2&prefix=abcde-12345%2F', args[1])

    def test_signature_changes_with_the_key(self):
        _, one = self.signed(key='a/one.zip')
        _, two = self.signed(key='a/two.zip')
        self.assertNotEqual(one['headers']['Authorization'], two['headers']['Authorization'])


class FindArchiveTest(unittest.TestCase):
    """Test choosing which file in a record to download."""

    def setUp(self):
        patcher = patch.object(msdlive, '_signed_request', return_value=FakeResponse(text=LIST_RESPONSE))
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_lists_files_largest_first(self):
        files = msdlive.list_record_files('abcde-12345', credentials=CREDENTIALS)
        self.assertEqual([f['size'] for f in files], [702662166, 12])

    def test_prefers_the_zip_over_the_placeholder(self):
        archive = msdlive.find_archive('abcde-12345', credentials=CREDENTIALS)
        self.assertEqual(archive['key'], 'abcde-12345/sample_data.zip')
        self.assertEqual(archive['size'], 702662166)

    def test_selects_by_extension_not_by_size(self):
        """A large non-archive must not be chosen over a smaller zip.

        Guards the case that matters: these records pair the real payload with a
        placeholder file, and picking on size alone would break if the placeholder
        ever grew.
        """

        listing = LIST_RESPONSE.replace('<Size>12</Size>', '<Size>999999999999</Size>')
        with patch.object(msdlive, '_signed_request', return_value=FakeResponse(text=listing)):
            archive = msdlive.find_archive('abcde-12345', credentials=CREDENTIALS)
        self.assertEqual(archive['key'], 'abcde-12345/sample_data.zip')

    def test_honors_an_explicit_filename(self):
        archive = msdlive.find_archive('abcde-12345', filename='dummy.txt', credentials=CREDENTIALS)
        self.assertEqual(archive['key'], 'abcde-12345/dummy.txt')

    def test_raises_for_a_missing_filename(self):
        with self.assertRaises(FileNotFoundError):
            msdlive.find_archive('abcde-12345', filename='absent.zip', credentials=CREDENTIALS)


class FindArchiveWithoutZipTest(unittest.TestCase):
    """A record holding no archive should raise rather than return the placeholder."""

    NO_ZIP = LIST_RESPONSE.replace('sample_data.zip', 'sample_data.tar')

    def test_raises_without_a_zip(self):
        with patch.object(msdlive, '_signed_request', return_value=FakeResponse(text=self.NO_ZIP)):
            with self.assertRaises(FileNotFoundError):
                msdlive.find_archive('abcde-12345', credentials=CREDENTIALS)


class DownloadRoutingTest(unittest.TestCase):
    """Test that InstallSupplement routes each url to the right transport."""

    @staticmethod
    def zip_bytes():
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as zipped:
            zipped.writestr('input/hello.txt', 'hello')
        return stream.getvalue()

    def test_msdlive_url_uses_the_signed_path(self):
        with TemporaryDirectory() as tmpdir:
            supplement = InstallSupplement(
                url='https://data.msdlive.org/records/abcde-12345',
                destination=tmpdir,
                filename='sample_data.zip',
            )
            with patch.object(msdlive, 'get_anonymous_credentials', return_value=CREDENTIALS), \
                 patch.object(msdlive, '_signed_request') as signed:
                signed.side_effect = [
                    FakeResponse(text=LIST_RESPONSE),          # list
                    FakeResponse(content=self.zip_bytes()),    # download
                ]
                supplement.fetch()
            self.assertTrue((Path(tmpdir) / 'input' / 'hello.txt').is_file())

    def test_plain_url_uses_requests(self):
        with TemporaryDirectory() as tmpdir:
            supplement = InstallSupplement(
                url='https://zenodo.org/record/1/files/a.zip?download=1',
                destination=tmpdir,
            )
            with patch(
                'mosartwmpy.utilities.download_data.requests.get',
                return_value=FakeResponse(content=self.zip_bytes()),
            ) as get:
                supplement.fetch()
            get.assert_called_once()
            self.assertTrue((Path(tmpdir) / 'input' / 'hello.txt').is_file())


class UnpackTest(unittest.TestCase):
    """The archive is streamed to disk in large chunks, not buffered in memory."""

    MEMBER_SIZE = 32 * 1024 * 1024

    @classmethod
    def large_zip_bytes(cls):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_STORED) as zipped:
            zipped.writestr('input/payload.bin', bytes(cls.MEMBER_SIZE))
        return stream.getvalue()

    def fetch_plain(self, response, tmpdir):
        supplement = InstallSupplement(url='https://zenodo.org/record/1/files/a.zip?download=1', destination=tmpdir)
        with patch('mosartwmpy.utilities.download_data.requests.get', return_value=response):
            supplement.fetch()

    def test_peak_memory_stays_far_below_the_archive_size(self):
        response = FakeResponse(content=self.large_zip_bytes())
        with TemporaryDirectory() as tmpdir:
            tracemalloc.start()
            try:
                self.fetch_plain(response, tmpdir)
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            self.assertEqual((Path(tmpdir) / 'input' / 'payload.bin').stat().st_size, self.MEMBER_SIZE)
        # buffering the archive in memory peaks at about the archive size; streaming peaks at a few chunks
        self.assertLess(peak, self.MEMBER_SIZE / 4, f'peak traced memory {peak / 2**20:.1f} MB for a {self.MEMBER_SIZE / 2**20:.0f} MB archive')

    def test_plain_download_reads_one_mebibyte_chunks(self):
        response = FakeResponse(content=DownloadRoutingTest.zip_bytes())
        with TemporaryDirectory() as tmpdir:
            self.fetch_plain(response, tmpdir)
        self.assertEqual(response.chunk_sizes, [download_data.CHUNK_SIZE])
        self.assertEqual(download_data.CHUNK_SIZE, 1024 * 1024)

    def test_corrupt_archive_raises_and_releases_the_logger(self):
        response = FakeResponse(content=b'<html>not a zip</html>')
        with TemporaryDirectory() as tmpdir:
            with self.assertRaises(zipfile.BadZipFile):
                self.fetch_plain(response, tmpdir)
        # no stdout handler InstallSupplement attaches may survive a failed download, on any logger
        for logger in (logging.getLogger(), download_data.logger):
            stdout_handlers = [h for h in logger.handlers if getattr(h, 'stream', None) is sys.stdout]
            self.assertEqual(stdout_handlers, [], 'the download handler is removed on failure')

    def test_spools_the_archive_into_the_destination_directory(self):
        """TMPDIR on compute nodes is often small or RAM-backed; the destination is known to have room."""

        response = FakeResponse(content=DownloadRoutingTest.zip_bytes())
        calls = []
        original = download_data.tempfile.TemporaryFile

        def recording(*args, **kwargs):
            calls.append(kwargs.get('dir'))
            return original(*args, **kwargs)

        with TemporaryDirectory() as tmpdir:
            with patch.object(download_data.tempfile, 'TemporaryFile', side_effect=recording):
                self.fetch_plain(response, tmpdir)
            self.assertEqual(calls, [tmpdir])
            self.assertTrue((Path(tmpdir) / 'input' / 'hello.txt').is_file())


class LoggerTest(unittest.TestCase):
    """InstallSupplement never adds to or removes from the root logger.

    A notebook or coupled driver that downloads data in the same process as a running model
    must keep the model's log handlers; a failed download must not leak a handler per attempt.
    """

    URL = 'https://zenodo.org/record/1/files/a.zip?download=1'

    def setUp(self):
        self.root = logging.getLogger()
        self.module_logger = download_data.logger
        self.root_handlers_before = list(self.root.handlers)
        self.module_handlers_before = list(self.module_logger.handlers)
        # stand in for a running model's log file handler
        self.model_handler = logging.StreamHandler(io.StringIO())
        self.root.addHandler(self.model_handler)
        self.addCleanup(self.root.removeHandler, self.model_handler)

    def fetch(self, response):
        with TemporaryDirectory() as tmpdir:
            supplement = InstallSupplement(url=self.URL, destination=tmpdir)
            with patch('mosartwmpy.utilities.download_data.requests.get', return_value=response):
                supplement.fetch()

    def test_root_handler_survives_a_successful_download(self):
        self.fetch(FakeResponse(content=DownloadRoutingTest.zip_bytes()))
        self.assertIn(self.model_handler, self.root.handlers)
        self.assertEqual(self.root.handlers, self.root_handlers_before + [self.model_handler])
        self.assertEqual(self.module_logger.handlers, self.module_handlers_before)

    def test_failed_downloads_add_no_handlers(self):
        for _ in range(2):
            with self.assertRaises(zipfile.BadZipFile):
                self.fetch(FakeResponse(content=b'<html>not a zip</html>'))
        self.assertEqual(self.root.handlers, self.root_handlers_before + [self.model_handler])
        self.assertEqual(self.module_logger.handlers, self.module_handlers_before)

    def test_messages_reach_the_existing_root_handler(self):
        """With logging already configured, progress propagates to the host's handlers at the host's level."""

        # a running model configures the root logger at INFO; the module logger must not override
        # the host's level either way, so set it here as the model would
        level_before = self.root.level
        self.root.setLevel(logging.INFO)
        self.addCleanup(self.root.setLevel, level_before)
        self.fetch(FakeResponse(content=DownloadRoutingTest.zip_bytes()))
        self.assertIn('Download and install complete.', self.model_handler.stream.getvalue())

    def test_unconfigured_logging_prints_to_stdout_for_the_duration(self):
        """Standalone use (the download CLI) still prints progress, through a handler on the module logger."""

        # make the process look unconfigured: no root handlers at all, restored afterwards
        self.root.removeHandler(self.model_handler)
        saved = list(self.root.handlers)
        for handler in saved:
            self.root.removeHandler(handler)
        self.addCleanup(lambda: [self.root.addHandler(h) for h in saved])

        with TemporaryDirectory() as tmpdir:
            supplement = InstallSupplement(url=self.URL, destination=tmpdir)
            attached = [h for h in self.module_logger.handlers if h not in self.module_handlers_before]
            self.assertEqual(len(attached), 1, 'one stdout handler is attached to the module logger')
            self.assertIs(attached[0].stream, sys.stdout)
            self.assertEqual(self.root.handlers, [], 'the root logger is still untouched')
            with patch('mosartwmpy.utilities.download_data.requests.get', return_value=FakeResponse(content=DownloadRoutingTest.zip_bytes())):
                supplement.fetch()
        self.assertEqual(self.module_logger.handlers, self.module_handlers_before, 'the handler is removed after the download')
        self.assertEqual(self.root.handlers, [])


if __name__ == '__main__':
    unittest.main()
