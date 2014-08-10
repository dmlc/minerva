project(MinervaLib)

include_directories(
  ${MinervaLib_SOURCE_DIR}
)

file(GLOB_RECURSE src_file_list "*.cpp")
add_library(minerva STATIC ${src_file_list})
target_link_libraries(minerva glog gflags)
set(LIBRARY_OUTPUT_PATH ${Minerva_BINARY_DIR}/lib)
