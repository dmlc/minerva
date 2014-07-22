project(MinervaTests)

include_directories(
  ${MinervaLib_SOURCE_DIR}
)

set(test_list "")

file(GLOB_RECURSE test_file_list "*.cpp")
foreach(test_src ${test_file_list})
  get_filename_component(test_name ${test_src} NAME_WE)
  add_executable(${test_name} ${test_src})
  target_link_libraries(${test_name} minerva ${external_lib_list})
  add_dependencies(${test_name} minerva)
  list(APPEND test_list ${test_name})
endforeach()

message(STATUS "Find tests: ${test_list}")
