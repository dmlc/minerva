project(MinervaLib)

include_directories(
  ${MinervaLib_SOURCE_DIR}
)

file(GLOB_RECURSE src_file_list "*.cpp")
file(GLOB_RECURSE cuda_src_file_list "*.cu")

if (CUDA_FOUND)
  if (ENABLE_PS)
    CUDA_ADD_LIBRARY(minerva STATIC ${src_file_list} ${cuda_src_file_list})
  else (ENABLE_PS)
    CUDA_ADD_LIBRARY(minerva SHARED ${src_file_list} ${cuda_src_file_list})
  endif ()
  CUDA_ADD_CUBLAS_TO_TARGET(minerva)
  target_link_libraries(minerva ${CUDNN_LIBRARIES} ${CUDA_curand_LIBRARY})
else (CUDA_FOUND)
  if (ENABLE_PS)
    add_library(minerva STATIC ${src_file_list})
  else (ENABLE_PS)
    add_library(minerva SHARED ${src_file_list})
  endif ()
endif (CUDA_FOUND)

add_dependencies(minerva third-party)

target_link_libraries(minerva ${THIRD_LIBS})

if (ENABLE_BLAS)
  target_link_libraries(minerva openblas)
endif ()

if (ENABLE_PS)
  target_link_libraries(minerva minervaps)
endif ()

set(LIBRARY_OUTPUT_PATH ${Minerva_BINARY_DIR}/lib)

