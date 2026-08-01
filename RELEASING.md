# Releasing mosartwmpy

## Blocking for v1.0.0: publish the sample input data

The reservoir parameter files bundled in the published sample input data predate the
`CAP_MIN` minimum storage column, so that feature cannot be exercised against the
downloadable inputs until a new data package is published.

Before tagging v1.0.0:

1. Upload the new sample input package (including reservoir parameter files with `CAP_MIN`)
   as a new version of the MSD-LIVE record.
2. Update the `sample_input` URL in `mosartwmpy/data_manifest.yaml` to the new record.
3. Re-run `python -m mosartwmpy.download` and confirm a fresh run picks up `CAP_MIN`.

Without `CAP_MIN` present the model still runs, falling back to 10% of storage capacity,
so this blocks feature verification rather than the release itself.

### Note on the existing records

The manifest points at [`m6pp5-7xt54`](https://data.msdlive.org/records/m6pp5-7xt54) (v0.0.6),
which is correct: it holds the real 702 MB `mosartwmpy_sample_input_data_1980_1985.zip`.
The later [`syt0j-x0203`](https://data.msdlive.org/records/syt0j-x0203) (v0.0.7) contains only a
12-byte `dummy.txt` placeholder and no data, so it must not be linked from the manifest.
Publishing the new version is also a chance to give v0.0.7 real content or retract it.

Both records share concept id `kehjf-ap948`. Note that the browser-facing
`/records/<id>/files/<name>?download=1` URL is hotlink protected and returns 403 to
non-browser clients; the API path used by the download utility works:

```
https://data.msdlive.org/api/records/<id>/files/<name>/content
```

## Version

The version lives in `mosartwmpy/_version.py` and is read by `setup.py`. Bump it
in a release branch, update `CHANGELOG.md`, and open a PR.

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
