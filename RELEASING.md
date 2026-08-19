# Releasing mosartwmpy

## Sample input data on MSD-LIVE

The `sample_input` entry in `mosartwmpy/data_manifest.yaml` points at MSD-LIVE record
[`m28qs-54544`](https://data.msdlive.org/records/m28qs-54544) (v0.0.8, doi
`10.57931/3398687`), whose reservoir parameters include the `CAP_MIN` minimum storage
column and ship as netCDF, Parquet, and CSV.

MSD-LIVE stores files in two different backends, which affects how they can be fetched:

- Records up to v0.0.6 keep their files in InvenioRDM's managed storage, so
  `https://data.msdlive.org/api/records/<id>/files/<name>/content` serves a signed
  redirect that any HTTP client can follow.
- Records from v0.0.7 (July 2023) onward keep their files in a project owned S3 bucket
  reached through a per record access point. For these, the Invenio file API lists only a
  small placeholder file, `/content` returns 404, and **no plain URL exists**.

`mosartwmpy/utilities/msdlive.py` handles the second case: it requests anonymous, read
only AWS credentials and signs requests against the record's access point using
Signature Version 4. No MSD-LIVE account is needed, and no `boto3` dependency was added,
since `requests` plus `hmac`/`hashlib` is enough. `download_data` picks the transport
from the manifest URL, so Zenodo entries are unaffected.

Two consequences worth knowing:

- A manifest entry for one of these records is the record URL, not a file URL, plus an
  optional `filename`. Without `filename` the largest `.zip` in the record is used.
- The credentials endpoint is undocumented. If MSD-LIVE later registers project bucket
  files with Invenio, or changes that endpoint, this code path can be dropped in favor of
  a plain URL. The behavior was reported to the MSD-LIVE team, along with a request that
  project bucket files be registered with InvenioRDM so plain-URL downloads work again.

Publishing a new data version is also a chance to give
[`syt0j-x0203`](https://data.msdlive.org/records/syt0j-x0203) (v0.0.7) a DOI-visible fix
or retract it; it holds real data but has never been reachable through the download
utility.

### After publishing a new data version

1. Update `sample_input.url` in `mosartwmpy/data_manifest.yaml` to the new record URL.
2. Run `python -m mosartwmpy.download`, select `sample_input`, and confirm the reservoir
   parameter file has the expected columns.

## Version

The version lives in `mosartwmpy/_version.py` and is read by `setup.py`. Bump it
in a release branch, update `CHANGELOG.md`, and open a PR.

The release date is written in three places that must agree: the `## [x.y.z] - DATE`
heading in `CHANGELOG.md`, `date-released` in `CITATION.cff`, and `publication_date` in
`.zenodo.json`. Restamp all three to the day the tag actually lands — if review slips,
this drifts, and Zenodo will mint a DOI carrying whatever date is in the file.

## Pre-release dry-run (TestPyPI)

Every PR runs `.github/workflows/testpypi-dryrun.yml`, which builds the sdist and
wheel, runs `twine check`, and uploads the built distribution as a workflow
artifact. It does not publish anywhere. You can also trigger it manually from the
Actions tab ("TestPyPI dry-run" -> "Run workflow").

To exercise the full upload -> install -> import round-trip against TestPyPI
without touching real PyPI, do a local upload. Because the `mosartwmpy` project on
TestPyPI is owned by the original maintainers, upload under a personal project name
instead (the import name is unaffected):

```bash
# from a clean checkout
python -m build                      # builds dist/ as "mosartwmpy"

# temporarily rename the DISTRIBUTION only (do NOT commit this change)
sed -i.bak 's/name="mosartwmpy"/name="mosartwmpy-dev"/' setup.py
rm -rf dist build *.egg-info
python -m build
mv setup.py.bak setup.py             # revert immediately

python -m twine check dist/*
python -m twine upload --repository testpypi dist/*   # needs ~/.pypirc [testpypi] token

# verify in a fresh environment (deps come from real PyPI)
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  mosartwmpy-dev==<version>
python -c "import mosartwmpy; print(mosartwmpy.__version__)"
```

TestPyPI rejects re-uploading an existing version; bump to a `.devN` or `rcN`
suffix if you need another round.

## Publishing to PyPI

> **Known issue (fix before the next tagged release):** `publish-to-pypi.yml`
> triggers only on `push` to `main`, but its publish jobs are gated on
> `startsWith(github.ref, 'refs/tags/')`. Those conditions never both hold, so a
> tag push currently publishes nothing. Add a tag trigger, e.g.:
>
> ```yaml
> on:
>   push:
>     branches: [main]
>     tags: ['v*']
> ```
>
> Publishing also requires trusted publishing (OIDC) to be configured for
> `IMMM-SFA/mosartwmpy` on both PyPI and TestPyPI (a project-owner action), or the
> publish step will fail.

Once the workflow trigger is fixed and trusted publishing is configured, publish
by pushing a version tag to `main`:

```bash
git tag v<version>
git push origin v<version>
```

CI then builds, publishes to PyPI and TestPyPI via trusted publishing,
Sigstore-signs the artifacts, and creates a GitHub Release.
