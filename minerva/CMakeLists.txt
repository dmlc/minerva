project(MinervaLib)

include_directories(
  ${MinervaLib_SOURCE_DIR}
)

file(GLOB_RECURSE src_file_list "*.cpp")
file(GLOB_RECURSE cuda_src_file_list "*.cu")
if (CUDA_FOUND)
  CUDA_ADD_LIBRARY(minerva STATIC ${src_file_list} ${cuda_src_file_list})
else (CUDA_FOUND)
  add_library(minerva STATIC ${src_file_list})
endif (CUDA_FOUND)
target_link_libraries(minerva glog gflags boost_thread boost_system)
set(LIBRARY_OUTPUT_PATH ${Minerva_BINARY_DIR}/lib)
