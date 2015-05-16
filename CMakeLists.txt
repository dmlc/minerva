cmake_minimum_required(VERSION 2.8)

function(find_cudnn)
  set(CUDNN_ROOT "" CACHE PATH "cuDNN root path")
  find_path(CUDNN_INCLUDE_DIRS cudnn.h
    PATHS ${CUDNN_ROOT}
          ${CUDNN_ROOT}/include
    DOC "cuDNN include path")
  find_library(CUDNN_LIBRARIES NAMES libcudnn.so
    PATHS ${CUDNN_ROOT}
          ${CUDNN_ROOT}/lib
          ${CUDNN_ROOT}/lib64
    DOC "cuDNN library path")
  if(CUDNN_INCLUDE_DIRS AND CUDNN_LIBRARIES)
    set(CUDNN_FOUND TRUE PARENT_SCOPE)
    message(STATUS "Found cuDNN (include: ${CUDNN_INCLUDE_DIRS}, library: ${CUDNN_LIBRARIES})")
    mark_as_advanced(CUDNN_INCLUDE_DIRS CUDNN_LIBRARIES)
  else()
    MESSAGE(FATAL_ERROR "Failed to find cuDNN in path: ${CUDNN_ROOT}")
  endif()
endfunction()

project(Minerva)

option(BUILD_CXX_APPS "build C++ applications"                OFF)
option(BUILD_TESTS    "build all unittests"                   OFF)
option(ENABLE_CUDA    "enable CUDA"                           ON )
option(ENABLE_PS      "enable Parameter Server"               OFF)
option(ENABLE_BLAS    "use BLAS library for CPU operations"   OFF)

message(STATUS "cmake generator: ${CMAKE_GENERATOR}")
message(STATUS "cmake build tool: ${CMAKE_BUILD_TOOL}")
message(STATUS "cmake build type: ${CMAKE_BUILD_TYPE}")

############################################# compiler flags
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11"       SUPPORTS_CXX11)
check_cxx_compiler_flag("-flto"            SUPPORTS_LTO)
check_cxx_compiler_flag("-mssse3"          SUPPORTS_MSSSE3)
check_cxx_compiler_flag("-ftree-vectorize" SUPPORTS_VECTORIZE)

set(COMPILER_FLAGS "-Wall -fPIC -march=native")
set(GENERAL_FLAGS_RELEASE "${COMPILER_FLAGS} -O2 -mtune=native -mssse3 -ftree-vectorize -funswitch-loops -ftree-vectorizer-verbose=0")
set(CMAKE_CXX_FLAGS_DEBUG "${GENERAL_FLAGS_DEBUG} -std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "${GENERAL_FLAGS_RELEASE} -std=c++11 -flto -DNDEBUG")

############################################# find packages
find_package(Threads REQUIRED)
if (ENABLE_CUDA)
  set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_ROOT})
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
    find_cudnn()
    include_directories(SYSTEM ${CUDNN_INCLUDE_DIRS})
  else ()
    message(FATAL_ERROR "CUDA enabled but not found")
  endif ()
endif()

############################################# third party libraries
add_subdirectory(third_party)
include_directories(SYSTEM ${THIRD_INCLUDE_PATH})
############################################# macros

if(ENABLE_PS)
  add_definitions(-DHAS_PS)
endif()

if(ENABLE_BLAS)
  add_definitions(-DHAS_CBLAS)
endif()

message(STATUS "building C++ applications             -- ${BUILD_CXX_APPS}")
message(STATUS "building unit tests                   -- ${BUILD_TESTS}")
message(STATUS "enable CUDA                           -- ${ENABLE_CUDA}")
message(STATUS "enable Parameter Server               -- ${ENABLE_PS}")
message(STATUS "use BLAS library for CPU operations   -- ${ENABLE_BLAS}")

add_subdirectory(minerva)

if(BUILD_CXX_APPS)
  add_subdirectory(apps)
endif()

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

