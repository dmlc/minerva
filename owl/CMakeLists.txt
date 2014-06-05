project(MinervaOwl)

include_directories(
  SYSTEM
  ${MinervaLib_SOURCE_DIR}
  /usr/include/python2.7 # Hacking for Python headers
)

file(GLOB_RECURSE SRC_LIST "*.cpp")
file(GLOB_RECURSE MINERVA_SRC_LIST "../minerva/*.cpp")
add_library(owl SHARED ${SRC_LIST} ${MINERVA_SRC_LIST})
set(LIBRARY_OUTPUT_PATH ${Minerva_BINARY_DIR}/owl)
target_link_libraries(owl boost_python python2.7)
