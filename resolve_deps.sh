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

PRINT_INFO "Resolving Boost"
${DOWNLOAD} http://downloads.sourceforge.net/project/boost/boost/1.57.0/boost_1_57_0.tar.gz
tar -xf boost_1_57_0.tar.gz
pushd boost_1_57_0 > /dev/null
./bootstrap.sh --prefix=${INSTALL_DIR} -with-libraries=python,thread
./b2 install -d0 -j4
popd > /dev/null

PRINT_INFO "Resolving Boost.NumPy"
${GIT_CLONE} https://github.com/ndarray/Boost.NumPy.git
pushd Boost.NumPy > /dev/null
cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} -DBoost_LIBRARY_DIR=${INSTALL_DIR}/lib -DBoost_INCLUDE_DIR=${INSTALL_DIR}/include -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON .
make install
popd > /dev/null

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

PRINT_INFO "Resolving gflags"
${GIT_CLONE} https://github.com/schuhschuh/gflags.git
pushd gflags > /dev/null
git checkout v2.1.1
cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} .
make install
popd > /dev/null

PRINT_INFO "Resolving glog"
${DOWNLOAD} https://google-glog.googlecode.com/files/glog-0.3.3.tar.gz
tar -xf glog-0.3.3.tar.gz
pushd glog-0.3.3 > /dev/null
./configure --prefix=${INSTALL_DIR}
make install
popd > /dev/null

PRINT_INFO "Resolving protobuf"
wget https://github.com/google/protobuf/releases/download/v3.0.0-alpha-1/protobuf-cpp-3.0.0-alpha-1.tar.gz
tar -xf protobuf-cpp-3.0.0-alpha-1.tar.gz
mv protobuf-3.0.0-alpha-1 protobuf-cpp-3.0.0-alpha-1
pushd  protobuf-cpp-3.0.0-alpha-1 > /dev/null
./configure --prefix=${INSTALL_DIR}
make install
popd > /dev/null

PRINT_INFO "Resolving python protobuf"
wget https://pypi.python.org/packages/source/p/protobuf/protobuf-3.0.0-alpha-1.tar.gz
tar -xf protobuf-3.0.0-alpha-1.tar.gz
pushd  protobuf-3.0.0-alpha-1 > /dev/null
./setup.py install --user
popd > /dev/null

