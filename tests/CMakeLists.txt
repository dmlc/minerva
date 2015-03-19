project(MinervaTests)

include_directories(
  ${MinervaLib_SOURCE_DIR}
)

enable_testing()

if (CUDA_FOUND)
  CUDA_ADD_LIBRARY(unittest_main STATIC unittest_main.cpp)
  CUDA_ADD_CUBLAS_TO_TARGET(unittest_main)
else ()
  add_library(unittest_main STATIC unittest_main.cpp)
endif ()
target_link_libraries(unittest_main minerva gtest)

set(unittest_list "")
set(test_list "")

if (NOT HAS_PS)
  file(GLOB_RECURSE test_file_list "*.cpp")
endif (NOT HAS_PS)

foreach(test_src ${test_file_list})
  get_filename_component(test_name ${test_src} NAME_WE)
  string(REGEX MATCH "^unittest_.*" unittest_name "${test_name}")
  string(LENGTH "${unittest_name}" is_unittest)
  if(is_unittest)
    if(NOT (${test_name} STREQUAL "unittest_main"))
      add_executable(${test_name} ${test_src})
      target_link_libraries(${test_name} unittest_main)
      list(APPEND unittest_list ${test_name})
      add_test(${test_name} ${test_name})
    endif()
  else()
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} minerva gtest)
    list(APPEND test_list ${test_name})
  endif()
endforeach()

message(STATUS "Tests: ${test_list}")
message(STATUS "Unit tests: ${unittest_list}")
