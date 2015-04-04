#!/bin/bash

GREEN='\033[0;32m'
NC='\033[0m'

function PRINT_INFO {
  echo -e "${GREEN}$1${NC}"
}

if [[ -n $(command -v axel 2> /dev/null) ]]; then
  DOWNLOAD="axel -n 64 -a"
else
  DOWNLOAD="wget"
fi
GIT_CLONE="git clone"

rm -rf deps > /dev/null 2>&1
mkdir deps
cd deps
INSTALL_DIR=$(pwd)
mkdir build
mkdir include
mkdir lib
cd build

PRINT_INFO "Resolving Boost.NumPy"
${GIT_CLONE} https://github.com/ndarray/Boost.NumPy.git
pushd Boost.NumPy > /dev/null
cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON .
make install
popd > /dev/null

