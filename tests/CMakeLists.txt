project(MinervaTests)

include_directories(
  ${MinervaLib_SOURCE_DIR}
)

add_library(main_unittest main_unittest.cpp)
target_link_libraries(main_unittest minerva gtest)

set(test_list "")

file(GLOB_RECURSE test_file_list "*.cpp")
foreach(test_src ${test_file_list})
  get_filename_component(test_name ${test_src} NAME_WE)
  string(REGEX MATCH "^unittest_.*" matched_test_name "${test_name}")
  string(LENGTH "${matched_test_name}" matched_len)
  if(matched_len)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} main_unittest)
    add_dependencies(${test_name} minerva)
    list(APPEND test_list ${test_name})
  endif()
endforeach()

message(STATUS "Find unit tests: ${test_list}")
