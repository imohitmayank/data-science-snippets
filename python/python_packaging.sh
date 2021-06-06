# clean the previous build files
python setup.py clean --all
# build the new distribution files
python setup.py sdist bdist_wheel
# upload the latest version to pypi
twine upload --skip-existing dist/*
