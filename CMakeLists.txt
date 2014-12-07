cmake_minimum_required(VERSION 2.8)

project(Minerva)

message(STATUS "cmake generator: ${CMAKE_GENERATOR}")
message(STATUS "cmake build tool: ${CMAKE_BUILD_TOOL}")
message(STATUS "cmake build type: ${CMAKE_BUILD_TYPE}")

set(COMPILER_FLAGS "-Wall -g -pthread -m64 -fPIC")
set(GENERAL_FLAGS_DEBUG "${COMPILER_FLAGS} -O0 -march=native")
set(GENERAL_FLAGS_RELEASE "${COMPILER_FLAGS} -O2 -march=native -mtune=native -mssse3 -ftree-vectorize -funswitch-loops -ftree-vectorizer-verbose=0 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${GENERAL_FLAGS_DEBUG} -std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "${GENERAL_FLAGS_RELEASE} -std=c++11")

string(REPLACE "," ";" EXTERNAL_LIB_PATH "${EXTERN_LIB_PATH}")
string(REPLACE "," ";" EXTERNAL_INCLUDE_PATH "${EXTERN_INCLUDE_PATH}")
set(DEPS_INSTALL_DIR ${CMAKE_BINARY_DIR}/deps)

find_package(CUDA)
if (CUDA_FOUND)
  if (CUDA_VERSION_MAJOR LESS 6)
    message(FATAL_ERROR "Need CUDA 6 for uniform addressing")
  endif (CUDA_VERSION_MAJOR LESS 6)
  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
  add_definitions(-DHAS_CUDA)
  string(REPLACE " " "\\\",\\\"" CUDA_CONCAT_DEBUG "${GENERAL_FLAGS_DEBUG}")
  string(REPLACE " " "\\\",\\\"" CUDA_CONCAT_RELEASE "${GENERAL_FLAGS_RELEASE}")
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
  set(CUDA_NVCC_FLAGS_DEBUG "-G -Xcompiler \\\"${CUDA_CONCAT_DEBUG}\\\"")
  set(CUDA_NVCC_FLAGS_RELEASE "-Xcompiler \\\"${CUDA_CONCAT_RELEASE}\\\"")
  set(CUDA_NVCC_FLAGS "-arch sm_35")
endif (CUDA_FOUND)

include_directories(
  SYSTEM ${EXTERNAL_INCLUDE_PATH}
  SYSTEM ${DEPS_INSTALL_DIR}/include
)

link_directories(
  ${LIBRARY_OUTPUT_PATH}
  ${EXTERN_LIB_PATH}
  ${DEPS_INSTALL_DIR}/lib
)

include(ExternalProject)

ExternalProject_Add(Boost
  URL http://downloads.sourceforge.net/project/boost/boost/1.57.0/boost_1_57_0.tar.gz
  URL_HASH MD5=25f9a8ac28beeb5ab84aa98510305299
  CONFIGURE_COMMAND ./bootstrap.sh --prefix=${DEPS_INSTALL_DIR} --with-libraries=python,thread
  BUILD_IN_SOURCE 1
  BUILD_COMMAND ./b2 -j4 -d0
  INSTALL_COMMAND ./b2 install -j4 -d0
)

ExternalProject_Add(BoostNumpy
  DEPENDS Boost
  GIT_REPOSITORY https://github.com/ndarray/Boost.NumPy.git
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${DEPS_INSTALL_DIR} -DBoost_LIBRARY_DIR=${DEPS_INSTALL_DIR}/lib
)

ExternalProject_Add(GoogleTest
  URL http://googletest.googlecode.com/files/gtest-1.7.0.zip
  URL_HASH SHA1=f85f6d2481e2c6c4a18539e391aa4ea8ab0394af
  CMAKE_ARGS -DBUILD_SHARED_LIBS=ON
  INSTALL_DIR ${DEPS_INSTALL_DIR}
  INSTALL_COMMAND
    COMMAND mkdir -p ${DEPS_INSTALL_DIR}/lib
    COMMAND mkdir -p ${DEPS_INSTALL_DIR}/include
    COMMAND ln -sf <BINARY_DIR>/libgtest.so ${DEPS_INSTALL_DIR}/lib
    COMMAND ln -sf <BINARY_DIR>/libgtest_main.so ${DEPS_INSTALL_DIR}/lib
    COMMAND ln -sf <SOURCE_DIR>/include/gtest ${DEPS_INSTALL_DIR}/include
)

ExternalProject_Add(Gflags
  GIT_REPOSITORY https://github.com/schuhschuh/gflags.git
  GIT_TAG v2.1.1
  CMAKE_ARGS -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=${DEPS_INSTALL_DIR}
)

ExternalProject_Add(Glog
  URL http://google-glog.googlecode.com/files/glog-0.3.3.tar.gz
  URL_HASH SHA1=ed40c26ecffc5ad47c618684415799ebaaa30d65
  CONFIGURE_COMMAND ./configure --prefix=${DEPS_INSTALL_DIR}
  BUILD_IN_SOURCE 1
  BUILD_COMMAND make -j
  INSTALL_COMMAND make install
)

add_subdirectory(minerva)
add_subdirectory(apps)
add_subdirectory(owl)
add_subdirectory(tests)
