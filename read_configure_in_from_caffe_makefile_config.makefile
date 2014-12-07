include Makefile.config

all:
	@if [ "${USE_CUDNN}" = "1" ]; then\
		echo "\
DEBUG_DIR=debug\n\
RELEASE_DIR=release\n\
CXX_COMPILER=g++-4.8\n\
C_COMPILER=gcc-4.8" > configure.in;\
		echo "EXTERN_INCLUDE_PATH=${INCLUDE_DIRS}" | sed -e "s/\ /,/g" >> configure.in;\
		echo "EXTERN_LIB_PATH=${LIBRARY_DIRS}" | sed -e "s/\ /,/g" >> configure.in;\
		echo "Done";\
	else\
		echo "Please have cuDNN installed";\
		exit 1;\
	fi
