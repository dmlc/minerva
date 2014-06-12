project(minerva_common)

set(SRC_LIST
	options/Options.cpp
	rpc/CommBase.cpp
	rpc/CommZMQ.cpp
	rpc/RPCStub.cpp
	)

add_library(minerva-common STATIC ${SRC_LIST})
set(LIBRARY_OUTPUT_PATH ${Minerva_BINARY_DIR}/lib)
requires_core_deps(minerva-common)
