project(MinervaOwl)

find_package(PythonLibs)
include_directories(
  ${MinervaLib_SOURCE_DIR}
  ${PYTHON_INCLUDE_DIRS}
)

file(GLOB_RECURSE src_list "*.cpp")
add_library(owl SHARED ${src_list})
set(LIBRARY_OUTPUT_PATH ${Minerva_BINARY_DIR}/owl)
target_link_libraries(owl boost_python boost_numpy python2.7 minerva)

if (CUDA_FOUND)
  CUDA_ADD_CUBLAS_TO_TARGET(owl)
endif (CUDA_FOUND)
