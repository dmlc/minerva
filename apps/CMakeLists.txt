project(MinervaApps)

include_directories(
  ${MinervaLib_SOURCE_DIR}
)

set(app_list "")

if (ENABLE_PS)
  file(GLOB_RECURSE app_file_list "ps/*.cpp")
else (ENABLE_PS)
  if (CUDA_FOUND)
    file(GLOB app_file_list "*.cpp")
  else (CUDA_FOUND)
    #For CPU only.
    file(GLOB app_file_list "mnist_mlp.cpp")
  endif (CUDA_FOUND)
endif (ENABLE_PS)

foreach(app_src ${app_file_list})
  get_filename_component(app_name ${app_src} NAME_WE)
  add_executable(${app_name} ${app_src})
  target_link_libraries(${app_name} minerva ${CORE_DEPS})
  if (CUDA_FOUND)
    CUDA_ADD_CUBLAS_TO_TARGET(${app_name})
  endif (CUDA_FOUND)
  list(APPEND app_list ${app_name})
endforeach()

message(STATUS "Build CXX Applications:")
foreach(app ${app_list})
  message(STATUS "  ${app}")
endforeach()
