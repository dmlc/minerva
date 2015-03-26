cmake_minimum_required(VERSION 2.8)

function(find_cudnn)
  set(CUDNN_ROOT "" CACHE PATH "CUDNN root path")
  find_path(CUDNN_INCLUDE_DIRS cudnn.h
    PATHS ${CUDNN_ROOT} 
          ${CUDNN_ROOT}/include
    DOC "CUDNN include path")
  find_library(CUDNN_LIBRARIES NAMES libcudnn.so
    PATHS ${CUDNN_ROOT} 
          ${CUDNN_ROOT}/lib
          ${CUDNN_ROOT}/lib64
    DOC "CUDNN library path")
    message(STATUS "Found CUDNN (include: ${CUDNN_INCLUDE_DIRS}, library: ${CUDNN_LIBRARIES})")
  if(CUDNN_INCLUDE_DIRS AND CUDNN_LIBRARIES)
    set(CUDNN_FOUND TRUE PARENT_SCOPE)
    message(STATUS "Found CUDNN (include: ${CUDNN_INCLUDE_DIRS}, library: ${CUDNN_LIBRARIES})")
    mark_as_advanced(CUDNN_INCLUDE_DIRS CUDNN_LIBRARIES)
  else()
    MESSAGE(FATAL_ERROR "Failed to find CUDNN in path: ${CUDNN_ROOT}")
  endif()
endfunction()

project(Minerva)

option(BUILD_CXX_APPS "build cxx applications"                OFF)
option(BUILD_TESTS    "build all unittests"                   OFF)
option(BUILD_OWL      "build python wrapper -- owl package"   ON )
option(ENABLE_CUDA    "enable cuda"                           ON )
option(ENABLE_PS      "enable parameter server"               OFF)

message(STATUS "cmake generator: ${CMAKE_GENERATOR}")
message(STATUS "cmake build tool: ${CMAKE_BUILD_TOOL}")
message(STATUS "cmake build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11" SUPPORTS_CXX11)
check_cxx_compiler_flag("-flto"      SUPPORTS_LTO)

set(COMPILER_FLAGS "-Wall -m64 -fPIC -march=native")
set(GENERAL_FLAGS_DEBUG "${COMPILER_FLAGS} -O0 -g")
set(GENERAL_FLAGS_RELEASE "${COMPILER_FLAGS} -O2 -mtune=native -mssse3 -ftree-vectorize -funswitch-loops -ftree-vectorizer-verbose=0 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${GENERAL_FLAGS_DEBUG} -std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "${GENERAL_FLAGS_RELEASE} -std=c++11 -flto")

set(Minerva_DEP_PATH "${Minerva_SOURCE_DIR}/deps")
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${Minerva_DEP_PATH})
message(STATUS "Search dependencies in path: ${CMAKE_PREFIX_PATH}")
find_package(Threads REQUIRED)
find_package(GLog REQUIRED)
find_package(GFlags REQUIRED)
find_package(Boost COMPONENTS thread system REQUIRED)

message(STATUS "Found GLog libraries: ${GLOG_LIBRARIES}")
message(STATUS "Found GFlags libraries: ${GFLAGS_LIBRARIES}")
message(STATUS "Found Boost libraries: ${Boost_LIBRARIES}")

include_directories(SYSTEM ${GLOG_INCLUDE_DIR})
include_directories(SYSTEM ${GFLAGS_INCLUDE_DIRS})
include_directories(SYSTEM ${Boost_INCLUDE_DIR})

if (ENABLE_CUDA)
  set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_ROOT})
  find_package(CUDA)
  if (CUDA_FOUND)
    message(STATUS "Compiling with CUDA support")
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
    find_cudnn()
    include_directories(SYSTEM ${CUDNN_INCLUDE_DIRS})
  else ()
    message(FATAL_ERROR "CUDA enabled but not found")
  endif ()
else ()
  message(STATUS "CUDA disabled by user")
endif()

if(ENABLE_PS)
  message(STATUS "Compiling with PS support (Owl package will not be built)")
  add_definitions(-DHAS_PS)
  set(BUILD_OWL OFF)
else()
  message(STATUS "PS support disabled")
endif()

# TODO should get rid of these two
include_directories(SYSTEM ${EXTERN_INCLUDE_PATH})
link_directories(${EXTERN_LIB_PATH})

add_subdirectory(minerva)

message(STATUS "Building cxx applications             -- ${BUILD_CXX_APPS}")
message(STATUS "Building owl package (python binding) -- ${BUILD_OWL}")
message(STATUS "Building unit tests                   -- ${BUILD_TESTS}")

if(BUILD_CXX_APPS)
  add_subdirectory(apps)
endif()

if(BUILD_OWL)
  add_subdirectory(owl)
endif()

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()
