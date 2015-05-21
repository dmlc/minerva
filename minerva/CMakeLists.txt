project(MinervaLib)

include_directories(
  ${MinervaLib_SOURCE_DIR}
)

file(GLOB_RECURSE src_file_list "*.cpp")
file(GLOB_RECURSE cuda_src_file_list "*.cu")

if (CUDA_FOUND)
  CUDA_ADD_LIBRARY(minerva ${src_file_list} ${cuda_src_file_list} SHARED)
  CUDA_ADD_CUBLAS_TO_TARGET(minerva)
  target_link_libraries(minerva ${CUDNN_LIBRARIES} ${CUDA_curand_LIBRARY})
else (CUDA_FOUND)
  add_library(minerva SHARED ${src_file_list})
endif (CUDA_FOUND)

add_dependencies(minerva third-party)
target_link_libraries(minerva ${CORE_DEPS})

if (BUILD_WITH_BLAS)
  target_link_libraries(minerva ${CBLAS_LIBRARIES})
endif ()

if (BUILD_WITH_PS)
  target_link_libraries(minerva minervaps)
endif ()

set(LIBRARY_OUTPUT_PATH ${Minerva_BINARY_DIR}/lib)

