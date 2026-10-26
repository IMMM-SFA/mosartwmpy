import json
import unittest
from unittest.mock import patch

from mosartwmpy.utilities import msdlive
from mosartwmpy.tests.test_msdlive import CREDENTIALS, LIST_RESPONSE, FakeResponse


class JsonResponse(FakeResponse):
    """A fake credentials response."""

    def json(self):
        return json.loads(self.text)


def listing(payload, truncated='false', token=None):
    """Build an S3 listing with the given <Contents> payload after the placeholder entry."""
    next_token = f'<NextContinuationToken>{token}</NextContinuationToken>' if token else ''
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Name>msdlive-project-im3</Name>
  <Prefix>abcde-12345/</Prefix>
  <IsTruncated>{truncated}</IsTruncated>
  {next_token}
  <Contents>
    <Key>abcde-12345/dummy.txt</Key>
    <Size>12</Size>
  </Contents>
  {payload}
</ListBucketResult>
"""


ARCHIVE = '<Contents><Key>abcde-12345/sample_data.zip</Key><Size>702662166</Size></Contents>'


class CredentialsValidationTest(unittest.TestCase):
    """The credentials response must carry the nested keys the request signer dereferences."""

    def fetch(self, payload):
        with patch.object(msdlive.requests, 'get', return_value=JsonResponse(text=json.dumps(payload))):
            return msdlive.get_anonymous_credentials()

    def test_complete_credentials_are_returned(self):
        self.assertEqual(self.fetch(CREDENTIALS), CREDENTIALS)

    def test_missing_nested_key_is_named(self):
        for key in ('accessKeyId', 'secretKey', 'sessionToken'):
            payload = json.loads(json.dumps(CREDENTIALS))
            del payload['credentials'][key]
            with self.assertRaises(KeyError) as context:
                self.fetch(payload)
            self.assertIn(f'credentials.{key}', str(context.exception))

    def test_credentials_that_are_not_an_object_are_rejected(self):
        payload = dict(CREDENTIALS, credentials='ASIAEXAMPLE')
        with self.assertRaises(KeyError):
            self.fetch(payload)


class ListingValidationTest(unittest.TestCase):
    """Malformed or incomplete listings raise a ValueError that names the problem."""

    def list_files(self, *responses):
        with patch.object(msdlive, '_signed_request', side_effect=[FakeResponse(text=r) for r in responses]):
            return msdlive.list_record_files('abcde-12345', credentials=CREDENTIALS)

    def test_well_formed_listing_is_parsed(self):
        files = self.list_files(LIST_RESPONSE)
        self.assertEqual([f['key'] for f in files], ['abcde-12345/sample_data.zip', 'abcde-12345/dummy.txt'])

    def test_entry_without_size_raises_with_the_entry(self):
        with self.assertRaises(ValueError) as context:
            self.list_files(listing('<Contents><Key>abcde-12345/sample_data.zip</Key></Contents>'))
        self.assertIn('sample_data.zip', str(context.exception))
        self.assertIn('Size', str(context.exception))

    def test_entry_without_key_raises(self):
        with self.assertRaises(ValueError):
            self.list_files(listing('<Contents><Size>702662166</Size></Contents>'))

    def test_non_numeric_size_raises(self):
        with self.assertRaises(ValueError):
            self.list_files(listing('<Contents><Key>abcde-12345/sample_data.zip</Key><Size>big</Size></Contents>'))

    def test_malformed_xml_raises_value_error(self):
        with self.assertRaises(ValueError) as context:
            self.list_files('<ListBucketResult><Contents><Key>x</Key>')
        self.assertIn('Unexpected listing response', str(context.exception))

    def test_truncated_listing_without_token_raises(self):
        with self.assertRaises(ValueError) as context:
            self.list_files(listing('', truncated='true'))
        self.assertIn('truncated', str(context.exception))

    def test_truncated_listing_with_token_fetches_the_next_page(self):
        files = self.list_files(listing('', truncated='true', token='page-2'), listing(ARCHIVE))
        self.assertEqual([f['size'] for f in files], [702662166, 12, 12])


if __name__ == '__main__':
    unittest.main()
