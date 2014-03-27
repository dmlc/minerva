project(Minerva)

cmake_minimum_required(VERSION 2.8.11)

message(STATUS "cmake generator: ${CMAKE_GENERATOR}")
message(STATUS "cmake build tool: ${CMAKE_BUILD_TOOL}")
message(STATUS "cmake build type: ${CMAKE_BUILD_TYPE}")
if(${CMAKE_GENERATOR} MATCHES "Unix Makefiles")
	set(COMPILER_FLAGS "-Wall -g -pthread -m64")
	set(CMAKE_C_FLAGS_DEBUG "-O0 -march=native ${COMPILER_FLAGS}")
	set(CMAKE_CXX_FLAGS_DEBUG "-O0 -march=native ${COMPILER_FLAGS}")
	set(CMAKE_C_FLAGS_RELEASE "-O2 -march=native -mtune=native -mssse3 -ftree-vectorize -funswitch-loops -ftree-vectorizer-verbose=0 ${COMPILER_FLAGS}")
	set(CMAKE_CXX_FLAGS_RELEASE "-O2 -march=native -mtune=native -mssse3 -ftree-vectorize -funswitch-loops -ftree-vectorizer-verbose=0 ${COMPILER_FLAGS}")
elseif(MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()

#set(LIBRARY_OUTPUT_PATH ${Minerva_SOURCE_DIR}/lib)
set(DEPS_INCLUDE_DIR ${Minerva_SOURCE_DIR}/deps/local/include)
set(DEPS_LIBS_DIR ${Minerva_SOURCE_DIR}/deps/local/lib)
set(LIBS_INSTALL_DIR ${Minerva_SOURCE_DIR}/deps/local/)

include(ExternalProject)
set(DEP_LIBS "")
#link_directories(${DEPS_LIBS_DIR})
#include_directories(
#SYSTEM
#${Minerva_SOURCE_DIR}/src
#${DEPS_INCLUDE_DIR}
#)

# CUDA =========================================================================
find_package(CUDA)
if(CUDA_FOUND)
	message(STATUS "CUDA found: include=${CUDA_INCLUDE_DIRS}; version=${CUDA_VERSION_STRING}")
	message(STATUS "CUBLAS lib: ${CUDA_CUBLAS_LIBRARIES}")
	message(STATUS "Curand lib: ${CUDA_curand_LIBRARY}")
	message(STATUS "Cuda RT lib: ${CUDA_LIBRARIES}")
	include_directories(${CUDA_INCLUDE_DIRS})
	add_definitions( -DMINERVA_HAS_CUDA )
	set(DEP_LIBS ${DEP_LIBS} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY} ${CUDA_LIBRARIES})
else(CUDA_FOUND)
	message(STATUS "CUDA is NOT turn on")
endif(CUDA_FOUND)

# MKL =========================================================================
if(MKL_FOUND)
	message(STATUS "MKL found: include=${MKL_INCLUDE_PATH}")
	message(STATUS "MKL lib path: ${MKL_LIBRARY_PATH}")
	include_directories(${MKL_INCLUDE_PATH})
	link_directories( ${INTEL_LIBRARY_PATH})
	link_directories( ${MKL_LIBRARY_PATH})
	add_definitions( -DMINERVA_HAS_MKL )
	set(DEP_LIBS ${DEP_LIBS} mkl_intel_lp64 mkl_intel_thread mkl_core iomp5)
else(MKL_FOUND)
	message(STATUS "MKL is NOT turn on")
endif(MKL_FOUND)

# MPI =========================================================================
# MPI is currently used to establish communication between rpc nodes
find_package(MPI)
if(MPI_FOUND)
	#message(STATUS "MPI_C_FOUND=${MPI_C_FOUND}")
	#message(STATUS "MPI_CXX_FOUND=${MPI_CXX_FOUND}")
	message(STATUS "MPI CXX Compiler: " ${MPI_CXX_COMPILER})
	message(STATUS "MPI CXX Include path: " ${MPI_CXX_INCLUDE_PATH})
	message(STATUS "MPI CXX Libraries: " ${MPI_CXX_LIBRARIES})
	message(STATUS "MPI CXX Link Flags: " ${MPI_CXX_LINK_FLAGS})
	add_definitions(-DMINERVA_HAS_MPI)
	include_directories(${MPI_INCLUDE_PATH})
	set(DEP_LIBS ${DEP_LIBS} ${MPI_CXX_LIBRARIES})
else(MPI_FOUND)
	message(STATUS "MPI Not Found! Distributed Executables will not be compiled")
endif(MPI_FOUND)

# Build Boost =================================================================
# boost library for parsing program options, filesystem support, 
# random number generation, and unordered containers.
if(WIN32)
	set(BOOST_BOOTSTRAP "bootstrap.bat")
	include_directories(${DEPS_INCLUDE_DIR}/boost-1_54/)
else(WIN32)
	set(BOOST_BOOTSTRAP "bootstrap.sh")
	set(Boost_LIBRARIES_FLAGS --layout=tagged)
endif(WIN32)

set(Boost_LIBRARIES_NAMES
	atomic filesystem program_options system date_time thread iostreams chrono
	)
foreach(bname ${Boost_LIBRARIES_NAMES})
	set(Boost_LIBRARIES_FLAGS ${Boost_LIBRARIES_FLAGS} --with-${bname})
endforeach()

ExternalProject_Add(boost
	PREFIX ${Minerva_SOURCE_DIR}/deps/boost
	URL http://sourceforge.net/projects/boost/files/boost/1.54.0/boost_1_54_0.tar.gz
	URL_MD5 efbfbff5a85a9330951f243d0a46e4b9
	BUILD_IN_SOURCE 1
	CONFIGURE_COMMAND 
	./${BOOST_BOOTSTRAP}
	BUILD_COMMAND  
	#C_INCLUDE_PATH=${DEPS_INCLUDE_DIR}
	#CPLUS_INCLUDE_PATH=${DEPS_INCLUDE_DIR}
	#LIBRARY_PATH=${DEPS_LIBS_DIR}
	./bjam install address-model=64 link=static variant=debug,release threading=multi -sNO_BZIP2=1 #runtime-link=static 
	${Boost_LIBRARIES_FLAGS}
	--prefix=<INSTALL_DIR>
	INSTALL_COMMAND ""
	INSTALL_DIR ${LIBS_INSTALL_DIR}
	)

if(UNIX)
	foreach(bname ${Boost_LIBRARIES_NAMES})
		if(${CMAKE_BUILD_TYPE} MATCHES Release)
			set(Boost_LIBRARIES ${Boost_LIBRARIES} ${DEPS_LIBS_DIR}/libboost_${bname}-mt${CMAKE_STATIC_LIBRARY_SUFFIX})
		else()
			set(Boost_LIBRARIES ${Boost_LIBRARIES} ${DEPS_LIBS_DIR}/libboost_${bname}-mt-d${CMAKE_STATIC_LIBRARY_SUFFIX})
		endif()
	endforeach()
	message(STATUS "Boost libs: " ${Boost_LIBRARIES})
	set(DEP_LIBS ${DEP_LIBS} ${Boost_LIBRARIES})
else(UNIX)
	message(STATUS "Boost libs: Use auto-linking")
endif(UNIX)

# Build ZeroMQ =================================================================
if(WIN32)
	set(ZMQ_MSVC_SLN <SOURCE_DIR>/builds/msvc/msvc10.sln)
	set(ZMQ_CONFIGURE_CMD devenv ${ZMQ_MSVC_SLN} /upgrade)
	set(ZMQ_BUILD_CMD devenv ${ZMQ_MSVC_SLN} /build release /project libzmq /projectconfig "Release|x64")
	set(ZMQ_INSTALL_CMD 
		${CMAKE_COMMAND} -E copy <SOURCE_DIR>/bin/x64/libzmq.dll ${DEPS_LIBS_DIR} && 
		${CMAKE_COMMAND} -E copy <SOURCE_DIR>/include/zmq.h ${DEPS_INCLUDE_DIR} &&
		${CMAKE_COMMAND} -E copy <SOURCE_DIR>/include/zmq_utils.h ${DEPS_INCLUDE_DIR}
		)
else(WIN32)
	set(ZMQ_CONFIGURE_CMD <SOURCE_DIR>/configure --prefix=${LIBS_INSTALL_DIR})
	set(ZMQ_BUILD_CMD make)
	set(ZMQ_INSTALL_CMD make install)
endif(WIN32)
#message(STATUS ${ZMQ_CONFIGURE_CMD})
#message(STATUS ${ZMQ_INSTALL_CMD})
ExternalProject_Add(zeromq 
	PREFIX ${Minerva_SOURCE_DIR}/deps/zeromq
	URL http://download.zeromq.org/zeromq-3.2.3.tar.gz
	URL_MD5 1abf8246363249baf5931a065ee38203 
	CONFIGURE_COMMAND ${ZMQ_CONFIGURE_CMD}
	BUILD_COMMAND ${ZMQ_BUILD_CMD}
	INSTALL_COMMAND ${ZMQ_INSTALL_CMD}
	)
#set(ZMQ_LIBRARIES ${DEPS_LIBS_DIR}/libzmq${CMAKE_STATIC_LIBRARY_SUFFIX})
set(DEP_LIBS ${DEP_LIBS} libzmq${CMAKE_STATIC_LIBRARY_SUFFIX})

# Extra libraries =================================================================
if (UNIX AND NOT APPLE)
	set(DEP_LIBS ${DEP_LIBS} rt)
endif()

macro(requires_core_deps NAME)
	target_link_libraries(${NAME} ${DEP_LIBS})
	add_dependencies(${NAME} boost zeromq)
endmacro(requires_core_deps)

add_subdirectory(minerva-common)
add_subdirectory(minerva-ps)
add_subdirectory(minerva)
add_subdirectory(tests)
