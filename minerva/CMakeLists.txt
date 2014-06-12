project(MinervaLib)

include_directories(
  SYSTEM
  ${MinervaLib_SOURCE_DIR}
)

file(GLOB_RECURSE SRC_LIST "*.cpp")
add_library(minerva STATIC ${SRC_LIST})
set(LIBRARY_OUTPUT_PATH ${Minerva_BINARY_DIR}/lib)
