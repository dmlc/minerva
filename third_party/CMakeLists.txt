project(Minerva_third_party)

include(ExternalProject)

set(THIRD_SRC ${Minerva_third_party_SOURCE_DIR})
set(THIRD_PREFIX ${Minerva_third_party_BINARY_DIR})
set(THIRD_LIB_PATH ${THIRD_PREFIX}/lib)
set(THIRD_INCLUDE_PATH ${THIRD_PREFIX}/include PARENT_SCOPE)
set(THIRD_LIBS "" PARENT_SCOPE)
add_custom_target(third-party)

ExternalProject_Add(gflags
  SOURCE_DIR ${THIRD_SRC}/gflags-2.1.2
  PREFIX ${THIRD_PREFIX}
  CMAKE_ARGS -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=${THIRD_PREFIX}
)
set(THIRD_LIBS ${THIRD_LIBS} ${THIRD_LIB_PATH}/libgflags.so)
add_dependencies(third-party gflags)

ExternalProject_Add(gtest
  SOURCE_DIR ${THIRD_SRC}/gtest-1.7.0
  PREFIX ${THIRD_PREFIX}
  CMAKE_ARGS -DBUILD_SHARED_LIBS=ON
  INSTALL_COMMAND ${CMAKE_COMMAND} -E copy <BINARY_DIR>/libgtest.so <INSTALL_DIR>/lib/ &&
                  ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/include/gtest <INSTALL_DIR>/include/gtest
)
set(THIRD_LIBS ${THIRD_LIBS} ${THIRD_LIB_PATH}/libgtest.so)
add_dependencies(third-party gtest)

ExternalProject_Add(glog
  SOURCE_DIR ${THIRD_SRC}/glog-0.3.3
  PREFIX ${THIRD_PREFIX}
  CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=${THIRD_PREFIX}
)
set(THIRD_LIBS ${THIRD_LIBS} ${THIRD_LIB_PATH}/libglog.so)
add_dependencies(third-party glog)

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

include_directories(SYSTEM ${THIRD_INCLUDE_INCLUDE})
