project(Minerva_third_party)

include(ExternalProject)

set(THIRD_SRC ${Minerva_third_party_SOURCE_DIR})
set(THIRD_PREFIX ${Minerva_third_party_BINARY_DIR})
set(THIRD_LIB_PATH ${THIRD_PREFIX}/lib)
set(THIRD_INCLUDE_PATH ${THIRD_PREFIX}/include PARENT_SCOPE)
set(THIRD_LIBS "")
add_custom_target(third-party)

ExternalProject_Add(gflags
  SOURCE_DIR ${THIRD_SRC}/gflags-2.1.1
  PREFIX ${THIRD_PREFIX}
  #CMAKE_ARGS -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=${THIRD_PREFIX}
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${THIRD_PREFIX} -DCMAKE_CXX_FLAGS=-fPIC
)
set(THIRD_LIBS ${THIRD_LIBS} ${THIRD_LIB_PATH}/libgflags.a) # XXX use shared lib since the static lib is not compiled with -fPIC
add_dependencies(third-party gflags)

#ExternalProject_Add(glog
#  SOURCE_DIR ${THIRD_SRC}/glog-0.3.3
#  PREFIX ${THIRD_PREFIX}
#  CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=${THIRD_PREFIX}
#)
#set(THIRD_LIBS ${THIRD_LIBS} ${THIRD_LIB_PATH}/libglog.a)
#add_dependencies(third-party glog)

# dmlc-core
set(DMLC_CORE_CONFIGURE_CMD
  ${CMAKE_COMMAND} -E copy <SOURCE_DIR>/make/config.mk <SOURCE_DIR>)
set(DMLC_CORE_INSTALL_CMD
  ${CMAKE_COMMAND} -E make_directory <INSTALL_DIR>/lib/ &&
  ${CMAKE_COMMAND} -E make_directory <INSTALL_DIR>/include/ &&
  ${CMAKE_COMMAND} -E copy <SOURCE_DIR>/libdmlc.a <INSTALL_DIR>/lib/ &&
  ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/include/dmlc <INSTALL_DIR>/include/dmlc)
ExternalProject_Add(dmlc-core
  PREFIX ${THIRD_PREFIX}
  GIT_REPOSITORY "https://github.com/dmlc/dmlc-core.git"
  UPDATE_COMMAND ""
  BUILD_IN_SOURCE 1
  CONFIGURE_COMMAND ${DMLC_CORE_CONFIGURE_CMD}
  INSTALL_COMMAND ${DMLC_CORE_INSTALL_CMD}
)
set(THIRD_LIBS ${THIRD_LIBS} ${THIRD_LIB_PATH}/libdmlc.a)
add_dependencies(third-party dmlc-core)

# mshadow
set(MSHADOW_INSTALL_CMD
  ${CMAKE_COMMAND} -E make_directory <INSTALL_DIR>/lib/ &&
  ${CMAKE_COMMAND} -E make_directory <INSTALL_DIR>/include/ &&
  ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/mshadow <INSTALL_DIR>/mshadow)
ExternalProject_Add(mshadow
  PREFIX ${THIRD_PREFIX}
  GIT_REPOSITORY "https://github.com/dmlc/mshadow.git"
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ${MSHADOW_INSTALL_CMD}
)
add_dependencies(third-party mshadow)

if(BUILD_TESTS)
  ExternalProject_Add(gtest
    SOURCE_DIR ${THIRD_SRC}/gtest-1.7.0
    PREFIX ${THIRD_PREFIX}
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${THIRD_PREFIX}
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy <BINARY_DIR>/libgtest.a <INSTALL_DIR>/lib/ &&
                    ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/include/gtest <INSTALL_DIR>/include/gtest
  )
  set(THIRD_PARTY_GTEST_LIB ${THIRD_LIB_PATH}/libgtest.a PARENT_SCOPE)
  add_dependencies(third-party gtest)
endif(BUILD_TESTS)

set(THIRD_LIBS ${THIRD_LIBS} PARENT_SCOPE) # make it visable to paraent scope
