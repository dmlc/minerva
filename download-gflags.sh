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

mkdir -p deps
cd deps
INSTALL_DIR=$(pwd)
mkdir -p build
mkdir -p include
mkdir -p lib
cd build

PRINT_INFO "Resolving gflags"
${GIT_CLONE} https://github.com/schuhschuh/gflags.git
pushd gflags > /dev/null
git checkout v2.1.1
cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} .
make install
popd > /dev/null

