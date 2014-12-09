# Minerva: a fast and flexible system for deep learning

## Goal

Make deep learning a home dish.

## Requirements and Dependencies

Minerva depends on several packages. If they are not in the system path, please set variable `EXTERN_INCLUDE_PATH` and `EXTERN_LIB_PATH` in `configure.in` accordingly.

* CUDA 6.5
* cuDNN
* LMDB
* NumPy

In addition, Minerva also depends on the following packages. A script `resolve_deps` is provided for automatic resolving of them.

* Boost
* Boost.NumPy
* glog
* gflags
* Google Test

## How to build

1. Set `CXX_COMPILER`, and `C_COMPILER` in `configure.in`. `g++-4.8` is recommended.
1. Set `EXTERN_INCLUDE_PATH`, and `EXTERN_LIB_PATH` in `configure.in` to reflect any dependency not in the system path. If there are more than one, use a comma-separated list.
1. `./configure`
1. Change directory into `debug` or `release` and `make`.

## MNIST data

http://pan.baidu.com/s/1ntsQs0x

