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

PRINT_INFO "Resolving Google Test"
${DOWNLOAD} https://googletest.googlecode.com/files/gtest-1.7.0.zip
unzip gtest-1.7.0.zip
pushd gtest-1.7.0 > /dev/null
cmake -DBUILD_SHARED_LIBS=ON .
make
cp -f libgtest.so ${INSTALL_DIR}/lib
cp -f libgtest_main.so ${INSTALL_DIR}/lib
cp -rf include ${INSTALL_DIR}
popd > /dev/null

