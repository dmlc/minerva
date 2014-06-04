project(MinervaTests)

include_directories(
  SYSTEM
  ${MinervaLib_SOURCE_DIR}
)
link_directories(${Minerva_BINARY_DIR}/lib)

set(test_list "")

file(GLOB_RECURSE SRC_LIST "*.cpp")
foreach(test_src ${SRC_LIST})
  get_filename_component(test_name ${test_src} NAME_WE)
  add_executable(${test_name} ${test_src})
  target_link_libraries(${test_name} minerva)
  add_dependencies(${test_name} minerva)
  list(APPEND test_list ${test_name})
endforeach()

message(STATUS "Find tests: ${test_list}")
