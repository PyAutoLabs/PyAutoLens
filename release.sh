#!/usr/bin/env bash

export PACKAGE_NAME=autolens

rm -rf $p/dist
rm -rf $p/build

set -e

export VERSION=$1

cat $PACKAGE_NAME/__init__.py | grep -v __version__ > temp

cat temp > $PACKAGE_NAME/__init__.py
rm temp
echo "__version__ = '"$VERSION"'" >> $PACKAGE_NAME/__init__.py

git add $PACKAGE_NAME/__init__.py

set +e
git commit -m "Incremented version number"
set -e

python3 setup.py sdist bdist_wheel
twine upload dist/* --skip-existing --username $PYPI_USERNAME --password $PYPI_PASSWORD


git push --tags

rm -rf $p/dist
rm -rf $p/build
