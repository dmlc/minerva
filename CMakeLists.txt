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

string(REPLACE "," ";" external_lib_list "${EXTERNAL_LIBS}")
string(REPLACE "," ";" external_lib_path_list "${EXTERNAL_LIB_PATH}")
string(REPLACE "," ";" external_include_path_list "${EXTERNAL_INCLUDE_PATH}")

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
  SYSTEM ${external_include_path_list}
)

link_directories(
  ${LIBRARY_OUTPUT_PATH}
  ${external_lib_path_list}
)

add_subdirectory(minerva)
add_subdirectory(apps)
add_subdirectory(owl)
add_subdirectory(tests)
