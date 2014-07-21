project(MinervaOwl)

include_directories(
  SYSTEM
  ${MinervaLib_SOURCE_DIR}
  /usr/include/python2.7 # Hacking for Python headers
)

file(GLOB_RECURSE src_list "*.cpp")
add_library(owl SHARED ${src_list})
set(LIBRARY_OUTPUT_PATH ${Minerva_BINARY_DIR}/owl)
target_link_libraries(owl boost_python python2.7 minerva)
