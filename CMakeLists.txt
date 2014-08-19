#set(CMAKE_CXX_COMPILER "g++-4.8")
cmake_minimum_required(VERSION 2.8.11)

project(Minerva)

message(STATUS "cmake generator: ${CMAKE_GENERATOR}")
message(STATUS "cmake build tool: ${CMAKE_BUILD_TOOL}")
message(STATUS "cmake build type: ${CMAKE_BUILD_TYPE}")

set(COMPILER_FLAGS "-std=c++11 -Wall -g -pthread -m64 -fPIC")
set(CMAKE_CXX_FLAGS_DEBUG "${COMPILER_FLAGS} -O0 -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${COMPILER_FLAGS} -O2 -march=native -mtune=native -mssse3 -ftree-vectorize -funswitch-loops -ftree-vectorizer-verbose=0 -DNDEBUG")

string(REPLACE "," ";" external_lib_list "${EXTERNAL_LIBS}")
string(REPLACE "," ";" external_lib_path_list "${EXTERNAL_LIB_PATH}")
string(REPLACE "," ";" external_include_path_list "${EXTERNAL_INCLUDE_PATH}")

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
