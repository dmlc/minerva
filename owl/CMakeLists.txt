project(MinervaOwl)

find_package(PythonLibs REQUIRED)
message(STATUS "Found python libs, version: ${PYTHONLIBS_VERSION_STRING}")

include_directories(
  ${MinervaLib_SOURCE_DIR}
  ${PYTHON_INCLUDE_DIRS}
)

find_package(Boost COMPONENTS python thread system REQUIRED)
find_package(BoostNumPy REQUIRED)

include_directories(SYSTEM ${BOOSTNUMPY_INCLUDE_DIR})

file(GLOB_RECURSE src_list "*.cpp")
add_library(owl SHARED ${src_list})
set(LIBRARY_OUTPUT_PATH ${Minerva_BINARY_DIR}/owl)
target_link_libraries(owl minerva ${Boost_LIBRARIES} ${BOOSTNUMPY_LIBRARIES} ${PYTHON_LIBRARIES})

if (CUDA_FOUND)
  CUDA_ADD_CUBLAS_TO_TARGET(owl)
endif (CUDA_FOUND)

#set(LIBRARY_OUTPUT_PATH ${Minerva_BINARY_DIR}/lib)
install(TARGETS owl DESTINATION ${MinervaOwl_SOURCE_DIR}/owl)
