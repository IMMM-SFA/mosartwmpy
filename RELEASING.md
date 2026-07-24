# Releasing mosartwmpy

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
