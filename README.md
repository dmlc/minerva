# Minerva: a fast and flexible system for deep learning

## Goal

Make deep learning a home dish.

## Features

* Matrix programming interface
* Easy interaction with NumPy
* Multi-GPU, multi-CPU support
* Good performance: ImageNet training achieves 213 images/s with one Titan GPU, 403Images/s with two GPUs


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

## Running apps

There are two ways to use Minerva: writing C++ code or Python code. Python binding is preferred since we provide easy interaction with NumPy. 

We have implemented several applications in Minerva including ImageNet training and MNIST training. After you have built Minerva, you can run both C++ and Python code.

The Python applications are located in `minerva/owl/apps`. After you have built Minerva, you can run the applications with `python {app}.py`.

The source code for C++ applicaionts are located in `minerva/apps` and there compiled executables are located in `minerva/release/apps`. You can run the executables directly.

The MNIST training data can be downloaded in: http://pan.baidu.com/s/1ntsQs0x

## Writing you own app

Minerva allows you to write you own code for machine learning, using a matrix interface just like Matlab or NumPy. You can use C++ or Python, whichever you prefer. The C++ and Python interface are quite similar. With Python, you can load data with NumPy and use it in Minerva, or you can convert Minerva NArrays into NumPy array and plot/print it with the tools provided in NumPy.

The NArray interface provided by Minerva is very intuitive. If you are familiar with either one of the matrix programming tools such as Matlab or NumPy, it should be very easy to get started with Minerva.

Minerva allows you to use multiple GPUs at the same time. By using the `set_device` function, you can specify which device you want the operation to run on. Once set, all the operations you specify will be performed on this device.

Minerva uses `lazy evaluation`, meaning that the operations are carried out only when necessary. For example, when you write `c = a + b`, the matrix addition will not be performed immediately. Instead, a dependency graph is constructed to track the dependency relationship. Once you try to evaluate the matrix c, either by printing some of its elements, or calling `c.WaitForEval()`, Minerva will lookup the dependency graph and try to carry out the operation. In this way, you can "push" multiple operations to different devices, and then trigger the evaluation on both devices at the same time. This is how multi-GPU programming is done in Minerva. Please refer to the code to get more details.

## License and support

Minerva is provided in the Apache V2 open source license.

You can use the "issues" tab in github to report bugs. For non-bug issues, please send up an email at minerva-support@googlegroups.com.
