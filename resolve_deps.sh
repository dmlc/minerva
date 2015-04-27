#!/bin/bash

GREEN='\033[0;32m'
NC='\033[0m'

function PRINT_INFO {
  echo -e "${GREEN}$1${NC}"
}

cd third_party
INSTALL_DIR=$(pwd)
rm -rf include lib &> /dev/null
mkdir include
mkdir lib

PRINT_INFO "Resolving Google Test"
pushd gtest-1.7.0 > /dev/null
cmake -DBUILD_SHARED_LIBS=ON .
make
cp -f libgtest.so ${INSTALL_DIR}/lib
cp -f libgtest_main.so ${INSTALL_DIR}/lib
cp -rf include ${INSTALL_DIR}
popd > /dev/null

PRINT_INFO "Resolving gflags"
pushd gflags-2.1.2 > /dev/null
cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} .
make install
popd > /dev/null

PRINT_INFO "Resolving glog"
pushd glog-0.3.3 > /dev/null
./configure --prefix=${INSTALL_DIR}
make install
popd > /dev/null

