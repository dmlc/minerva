project(MinervaTests)

include_directories(
  ${MinervaLib_SOURCE_DIR}
)

enable_testing()

add_library(utmain OBJECT unittest_main.cpp)

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
      add_executable(${test_name} ${test_src} $<TARGET_OBJECTS:utmain>)
      target_link_libraries(
        ${test_name}
        minerva
        "${THIRD_PARTY_GTEST_LIB}"
      )
      list(APPEND unittest_list ${test_name})
      add_test(${test_name} ${test_name})
    endif()
  else()
    add_executable(${test_name} ${test_src})
    target_link_libraries(
      ${test_name}
      minerva
    )
    list(APPEND test_list ${test_name})
  endif()
endforeach()

message(STATUS "Unit tests:")
foreach(t ${unittest_list})
  message(STATUS "  ${t}")
endforeach()
message(STATUS "Other Tests:")
foreach(t ${test_list})
  message(STATUS "  ${t}")
endforeach()
