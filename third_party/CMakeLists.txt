project(Minerva_third_party)

include(ExternalProject)

set(THIRD_SRC ${Minerva_third_party_SOURCE_DIR})
set(THIRD_PREFIX ${Minerva_third_party_BINARY_DIR})

ExternalProject_Add(gflags
  SOURCE_DIR ${THIRD_SRC}/gflags-2.1.2
  PREFIX ${THIRD_PREFIX}
  CMAKE_ARGS -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=${THIRD_PREFIX}
)

ExternalProject_Add(gtest
  SOURCE_DIR ${THIRD_SRC}/gtest-1.7.0
  PREFIX ${THIRD_PREFIX}
  CMAKE_ARGS -DBUILD_SHARED_LIBS=ON
  INSTALL_COMMAND ${CMAKE_COMMAND} -E copy <BINARY_DIR>/libgtest.so <INSTALL_DIR>/lib/ &&
                  ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/include/gtest <INSTALL_DIR>/include/gtest
)

ExternalProject_Add(glog
  SOURCE_DIR ${THIRD_SRC}/glog-0.3.3
  PREFIX ${THIRD_PREFIX}
  CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=${THIRD_PREFIX}
)

# dmlc-core
#set(DMLC_CORE_CONFIGURE_CMD
#  ${CMAKE_COMMAND} -E copy <SOURCE_DIR>/make/config.mk <SOURCE_DIR>)
#set(DMLC_CORE_INSTALL_CMD
#  ${CMAKE_COMMAND} -E copy <SOURCE_DIR>/libdmlc.a <INSTALL_DIR>/lib/ &&
#  ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/include/dmlc <INSTALL_DIR>/include/)
#
#ExternalProject_Add(dmlc-core
#  PREFIX ${THIRD_PARTY_ROOT}/dmlc
#  GIT_REPOSITORY "https://github.com/dmlc/dmlc-core.git"
#  BUILD_IN_SOURCE 1
#  CONFIGURE_COMMAND ${DMLC_CORE_CONFIGURE_CMD}
#  INSTALL_DIR ${THIRD_PARTY_ROOT}
#  INSTALL_COMMAND ${DMLC_CORE_INSTALL_CMD}
#)
#

set(LIBRARY_OUTPUT_PATH ${THIRD_PREFIX}/lib)
