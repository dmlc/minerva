project(MinervaLib)

include_directories(
  ${MinervaLib_SOURCE_DIR}
)

file(GLOB_RECURSE src_file_list "*.cpp")
file(GLOB_RECURSE cuda_src_file_list "*.cu")
if (CUDA_FOUND)
  CUDA_ADD_LIBRARY(minerva ${src_file_list} ${cuda_src_file_list} STATIC)
else (CUDA_FOUND)
  add_library(minerva ${src_file_list} STATIC)
endif (CUDA_FOUND)
target_link_libraries(minerva glog gflags)
set(LIBRARY_OUTPUT_PATH ${Minerva_BINARY_DIR}/lib)
