import io
import unittest
import zipfile

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from mosartwmpy.utilities import msdlive
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

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f'status {self.status_code}')

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i:i + chunk_size]


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


if __name__ == '__main__':
    unittest.main()
