set(CMAKE_CXX_COMPILER "g++-4.8")
cmake_minimum_required(VERSION 2.8.11)

project(Minerva)

message(STATUS "cmake generator: ${CMAKE_GENERATOR}")
message(STATUS "cmake build tool: ${CMAKE_BUILD_TOOL}")
message(STATUS "cmake build type: ${CMAKE_BUILD_TYPE}")

set(COMPILER_FLAGS "-std=c++11 -Wall -g -pthread -m64")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -march=native ${COMPILER_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -march=native -mtune=native -mssse3 -ftree-vectorize -funswitch-loops -ftree-vectorizer-verbose=0 ${COMPILER_FLAGS}")

#set(DEPS_INCLUDE_DIR ${Minerva_SOURCE_DIR}/deps/local/include)
#set(DEPS_LIBS_DIR ${Minerva_SOURCE_DIR}/deps/local/lib)
#set(LIBS_INSTALL_DIR ${Minerva_SOURCE_DIR}/deps/local/)

#include(ExternalProject)

#macro(requires_core_deps NAME)
#target_link_libraries(${NAME} ${DEP_LIBS})
#add_dependencies(${NAME} minerva)
#endmacro(requires_core_deps)

add_subdirectory(minerva)
add_subdirectory(tests)
