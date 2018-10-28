#!/usr/bin/env bash
cd ..
python3 -m pip install virtualenv
VENVPATH='./venv'
if ! [[ -d  $VENVPATH ]]; then
    virtualenv -p python3 $VENVPATH
fi
source $VENVPATH/bin/activate
python -m pip install --upgrade pip
python -m pip --version
python -m pip install -r requirements.txt

if  [[ -d bigartm ]]; then
    rm -r bigartm
fi

git clone --branch=stable https://github.com/bigartm/bigartm.git
cd bigartm
mkdir build && cd build
cmake ..
make -j4

python -m pip install python/bigartm*.whl

cd ../..
rm -r bigartm