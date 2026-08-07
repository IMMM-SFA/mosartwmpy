"""Fetch public MSD-LIVE datasets that are not served over plain HTTP.

MSD-LIVE records created from July 2023 onward store their files in a
project-owned S3 bucket reached through a per-record access point, rather than in
InvenioRDM's managed storage. For those records the Invenio file API reports only
a small placeholder file and ``/api/records/<id>/files/<name>/content`` returns
404, so there is no URL a plain HTTP client can fetch. Older records are
unaffected and are still downloaded directly by ``download_data``.

The data is public: a credentials endpoint hands out short lived, read only AWS
credentials to anonymous callers, which are then used to sign requests against
the record's access point. No account or login is involved.

Requests are signed with AWS Signature Version 4 using only ``requests`` and the
standard library, to avoid adding a ``boto3`` dependency for this one code path.
"""

import datetime
import hashlib
import hmac
import logging
import re
import urllib.parse
import xml.etree.ElementTree as ElementTree

import requests

# hands out anonymous, read only credentials scoped to public records
CREDENTIALS_URL = 'https://data.msdlive.org/api/get-aws-credentials'

# recognize a MSD-LIVE record page or api url and pull out the record id
RECORD_ID_PATTERN = re.compile(r'data\.msdlive\.org/(?:api/)?records/([a-z0-9]+-[a-z0-9]+)', re.IGNORECASE)

S3_NAMESPACE = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}

ALGORITHM = 'AWS4-HMAC-SHA256'
# the access point rejects a payload hash for streamed reads; this is the documented sentinel
UNSIGNED_PAYLOAD = 'UNSIGNED-PAYLOAD'


def is_msdlive_record_url(url: str) -> bool:
    """Whether a url points at a MSD-LIVE record.

    Args:
        url (str): the url to inspect

    Returns:
        bool: True if a record id can be read out of the url
    """

    return url is not None and RECORD_ID_PATTERN.search(url) is not None


def record_id_from_url(url: str) -> str:
    """Extract the record id from a MSD-LIVE record url.

    Args:
        url (str): a record page or api url, optionally including a file path

    Returns:
        str: the record id, i.e. the ``m28qs-54544`` in a record url

    Raises:
        ValueError: if no record id is present in the url
    """

    match = RECORD_ID_PATTERN.search(url or '')
    if not match:
        raise ValueError(f'Unable to find a MSD-LIVE record id in "{url}".')
    return match.group(1)


def get_anonymous_credentials(credentials_url: str = CREDENTIALS_URL) -> dict:
    """Request short lived, read only AWS credentials for public records.

    Args:
        credentials_url (str): endpoint handing out the credentials

    Returns:
        dict: the parsed response, with ``region``, ``accountId`` and ``credentials`` keys
    """

    response = requests.get(credentials_url, timeout=60)
    response.raise_for_status()
    payload = response.json()
    for key in ('region', 'accountId', 'credentials'):
        if key not in payload:
            raise KeyError(f'MSD-LIVE credentials response is missing "{key}".')
    return payload


def _sign(key: bytes, message: str) -> bytes:
    """HMAC-SHA256 one step of the SigV4 key derivation chain."""

    return hmac.new(key, message.encode('utf-8'), hashlib.sha256).digest()


def _signing_key(secret_key: str, datestamp: str, region: str) -> bytes:
    """Derive the SigV4 signing key for S3 in a region on a date."""

    key = _sign(f'AWS4{secret_key}'.encode('utf-8'), datestamp)
    key = _sign(key, region)
    key = _sign(key, 's3')
    return _sign(key, 'aws4_request')


def _signed_request(
    credentials: dict,
    record_id: str,
    method: str = 'GET',
    key: str = '',
    query: dict = None,
    headers: dict = None,
    stream: bool = False,
) -> requests.Response:
    """Send a SigV4 signed request to a record's S3 access point.

    Args:
        credentials (dict): as returned by ``get_anonymous_credentials``
        record_id (str): the MSD-LIVE record id, which names the access point
        method (str): HTTP method
        key (str): object key within the access point, or empty to address the root
        query (dict): query parameters, used for listing
        headers (dict): extra request headers, i.e. a Range header
        stream (bool): whether to stream the response body

    Returns:
        requests.Response: the response, with status left for the caller to check
    """

    region = credentials['region']
    account_id = credentials['accountId']
    session = credentials['credentials']

    host = f'{record_id}-{account_id}.s3-accesspoint.{region}.amazonaws.com'
    canonical_uri = '/' + urllib.parse.quote(key, safe='/~')

    # S3 requires the canonical query string to be sorted and rfc3986 encoded
    query = query or {}
    canonical_query = '&'.join(
        f'{urllib.parse.quote(k, safe="~")}={urllib.parse.quote(str(v), safe="~")}'
        for k, v in sorted(query.items())
    )

    now = datetime.datetime.now(datetime.timezone.utc)
    amzdate = now.strftime('%Y%m%dT%H%M%SZ')
    datestamp = now.strftime('%Y%m%d')

    # only these headers are signed; anything else the caller adds is not
    signed = {
        'host': host,
        'x-amz-content-sha256': UNSIGNED_PAYLOAD,
        'x-amz-date': amzdate,
        'x-amz-security-token': session['sessionToken'],
    }
    signed_header_names = ';'.join(sorted(signed))
    canonical_headers = ''.join(f'{name}:{signed[name]}\n' for name in sorted(signed))

    canonical_request = '\n'.join([
        method,
        canonical_uri,
        canonical_query,
        canonical_headers,
        signed_header_names,
        UNSIGNED_PAYLOAD,
    ])

    scope = f'{datestamp}/{region}/s3/aws4_request'
    string_to_sign = '\n'.join([
        ALGORITHM,
        amzdate,
        scope,
        hashlib.sha256(canonical_request.encode('utf-8')).hexdigest(),
    ])

    signature = hmac.new(
        _signing_key(session['secretKey'], datestamp, region),
        string_to_sign.encode('utf-8'),
        hashlib.sha256,
    ).hexdigest()

    request_headers = dict(signed)
    request_headers['Authorization'] = (
        f'{ALGORITHM} Credential={session["accessKeyId"]}/{scope}, '
        f'SignedHeaders={signed_header_names}, Signature={signature}'
    )
    if headers:
        request_headers.update(headers)

    url = f'https://{host}{canonical_uri}'
    if canonical_query:
        url = f'{url}?{canonical_query}'

    return requests.request(method, url, headers=request_headers, stream=stream, timeout=120)


def list_record_files(record_id: str, credentials: dict = None) -> list:
    """List the files available in a public MSD-LIVE record.

    Args:
        record_id (str): the MSD-LIVE record id
        credentials (dict): reuse existing credentials, otherwise fetched

    Returns:
        list: dicts with ``key`` and ``size``, sorted largest first
    """

    credentials = credentials or get_anonymous_credentials()

    files = []
    continuation_token = None
    while True:
        query = {'list-type': '2', 'prefix': f'{record_id}/'}
        if continuation_token:
            query['continuation-token'] = continuation_token
        response = _signed_request(credentials, record_id, query=query)
        response.raise_for_status()

        tree = ElementTree.fromstring(response.text)
        for contents in tree.findall('s3:Contents', S3_NAMESPACE):
            key = contents.find('s3:Key', S3_NAMESPACE).text
            size = int(contents.find('s3:Size', S3_NAMESPACE).text)
            files.append({'key': key, 'size': size})

        truncated = tree.find('s3:IsTruncated', S3_NAMESPACE)
        if truncated is not None and truncated.text == 'true':
            token = tree.find('s3:NextContinuationToken', S3_NAMESPACE)
            continuation_token = token.text if token is not None else None
            if continuation_token:
                continue
        break

    return sorted(files, key=lambda f: f['size'], reverse=True)


def find_archive(record_id: str, filename: str = None, credentials: dict = None) -> dict:
    """Choose which file in a record to download.

    Args:
        record_id (str): the MSD-LIVE record id
        filename (str): a specific file to look for, otherwise the largest zip is used
        credentials (dict): reuse existing credentials, otherwise fetched

    Returns:
        dict: the chosen file, with ``key`` and ``size``

    Raises:
        FileNotFoundError: if the record has no matching file
    """

    credentials = credentials or get_anonymous_credentials()
    files = list_record_files(record_id, credentials=credentials)

    if filename:
        for entry in files:
            if entry['key'].rsplit('/', 1)[-1] == filename:
                return entry
        raise FileNotFoundError(f'MSD-LIVE record {record_id} has no file named "{filename}".')

    # records carry a small placeholder file alongside the real payload, so prefer
    # the largest zip rather than simply the first entry
    archives = [entry for entry in files if entry['key'].lower().endswith('.zip')]
    if not archives:
        raise FileNotFoundError(
            f'MSD-LIVE record {record_id} contains no zip archive; found '
            f'{[entry["key"] for entry in files]}.'
        )
    return archives[0]


def open_record_file_stream(record_id: str, key: str, credentials: dict = None) -> requests.Response:
    """Open a streaming response for a file in a public MSD-LIVE record.

    Args:
        record_id (str): the MSD-LIVE record id
        key (str): the full object key within the access point
        credentials (dict): reuse existing credentials, otherwise fetched

    Returns:
        requests.Response: a streaming response positioned at the start of the file
    """

    credentials = credentials or get_anonymous_credentials()
    logging.debug(f'Requesting {key} from MSD-LIVE record {record_id}.')
    response = _signed_request(credentials, record_id, key=key, stream=True)
    response.raise_for_status()
    return response
