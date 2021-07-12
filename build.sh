rm -f dist/*
python -m build
python -m twine upload dist/*