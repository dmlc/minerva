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

function(find_blas)
  set(BLAS_ROOT "" CACHE PATH "BLAS root path")
  find_path(CBLAS_INCLUDE_DIRS cblas.h
    PATHS ${BLAS_ROOT}
          ${BLAS_ROOT}/include
    DOC "BLAS include path")
  find_library(CBLAS_LIBRARIES NAMES libcblas.so
    PATHS ${BLAS_ROOT}
          ${BLAS_ROOT}/lib
          ${BLAS_ROOT}/lib64
    DOC "BLAS library path")
  if(CBLAS_INCLUDE_DIRS AND CBLAS_LIBRARIES)
    set(CBLAS_FOUND TRUE PARENT_SCOPE)
    message(STATUS "Found BLAS (include: ${CBLAS_INCLUDE_DIRS}, library: ${CBLAS_LIBRARIES})")
    mark_as_advanced(CBLAS_INCLUDE_DIRS CBLAS_LIBRARIES)
  else()
    MESSAGE(FATAL_ERROR "Failed to find BLAS in path: ${BLAS_ROOT}. If you are using other BLAS than cBLAS, please link your .so into libcblas.so.")
  endif()
endfunction()

function(find_ps)
  set(PS_ROOT "" CACHE PATH "minervaps root path")
  find_library(PS_LIBRARIES NAMES libminervaps.a
    PATHS ${PS_ROOT}/build
    DOC "PS library path")
  if(PS_LIBRARIES)
    set(PS_FOUND TRUE PARENT_SCOPE)
    message(STATUS "Found PS (library: ${PS_LIBRARIES})")
    mark_as_advanced(PS_LIBRARIES)
  else()
    MESSAGE(FATAL_ERROR "Failed to find PS in path: ${PS_ROOT}. Make sure you have built ps with 'make minerva'.")
  endif()
endfunction()

project(Minerva)

option(BUILD_CXX_APPS    "build C++ applications"                OFF)
option(BUILD_TESTS       "build all unittests"                   OFF)
option(BUILD_CPU_ONLY    "build cpu-only version"                OFF)
option(BUILD_WITH_PS     "build with parameter server support"   OFF)
option(BUILD_WITH_BLAS   "build with BLAS library for CPU"       OFF)

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
set(GENERAL_FLAGS_DEBUG "${COMPILER_FLAGS} -O0 -g")
set(GENERAL_FLAGS_RELEASE "${COMPILER_FLAGS} -O2 -mtune=native -mssse3")
set(CMAKE_CXX_FLAGS_DEBUG "${GENERAL_FLAGS_DEBUG} -std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "${GENERAL_FLAGS_RELEASE} -std=c++11 -DNDEBUG")
#MSHADOW and CXXNET Flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fopenmp -DMSHADOW_STAND_ALONE=1 -DCXXNET_USE_OPENCV=1 -DCXXNET_USE_OPENCV_DECODER=1")

############################################# find packages
set(CORE_DEPS "")

find_package(Threads REQUIRED)
set(CORE_DEPS ${CORE_DEPS} ${CMAKE_THREAD_LIBS_INIT})

if (BUILD_CPU_ONLY)
  message(STATUS "build cpu-only version; cuda is not enabled!")
else(BUILD_CPU_ONLY)
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
    #set(CUDA_NVCC_FLAGS "-arch sm_35")
    set(CUDA_NVCC_FLAGS "-gencode=arch=compute_20,code=sm_20
                         -gencode=arch=compute_30,code=sm_30
                         -gencode=arch=compute_35,code=sm_35
                         -gencode=arch=compute_50,code=sm_50
                         -gencode=arch=compute_52,code=sm_52
                         -gencode=arch=compute_52,code=compute_52")
    find_cudnn()
    include_directories(SYSTEM ${CUDNN_INCLUDE_DIRS})
  else ()
    message(FATAL_ERROR "CUDA enabled but not found")
  endif ()
endif(BUILD_CPU_ONLY)

############################################# opencv
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV libraries: ${OpenCV_LIBRARIES}")
message(STATUS "OpenCV includes: ${OpenCV_INCLUDE_DIRS}")
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})

############################################# third party libraries
add_subdirectory(third_party)
include_directories(SYSTEM ${THIRD_INCLUDE_PATH})
set(CORE_DEPS ${CORE_DEPS} ${THIRD_LIBS})

############################################# macros

if(BUILD_WITH_PS)
  add_definitions(-DHAS_PS)
  find_ps()
  message(STATUS "PS enabled, forcing BUILD_TESTS to 0")
  set(BUILD_TESTS 0)
endif()

if(BUILD_WITH_BLAS)
  add_definitions(-DHAS_CBLAS)
  find_blas()
  include_directories(SYSTEM ${CBLAS_INCLUDE_DIRS})
endif()

message(STATUS "build C++ applications              -- ${BUILD_CXX_APPS}")
message(STATUS "build unit tests                    -- ${BUILD_TESTS}")
message(STATUS "build cpu-only version              -- ${BUILD_CPU_ONLY}")
message(STATUS "build with parameter server support -- ${BUILD_WITH_PS}")
message(STATUS "build with BLAS library for CPU     -- ${BUILD_WITH_BLAS}")

add_subdirectory(minerva)

if(BUILD_CXX_APPS)
  add_subdirectory(apps)
endif()

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()
