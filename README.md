# Minerva: a fast and flexible system for deep learning

## Latest News

* Minerva's Tutorial and API documents are released!
* Minerva had migrated to [dmlc](https://github.com/dmlc), where you could find many awesome machine learning repositories!
* Minerva now evolves to use cudnn_v2. Please download and use the new [library](https://developer.nvidia.com/cuDNN).

## Overview

Minerva is a fast and flexible tool for deep learning. It provides NDarray programming interface, just like Numpy. Python bindings and C++ bindings are both available. The resulting code can be run on CPU or GPU. Multi-GPU support is very easy. Please refer to the examples to see how multi-GPU setting is used.Minerva is a fast and flexible tool for deep learning. It provides NDarray programming interface, just like Numpy. Python bindings and C++ bindings are both available. The resulting code can be run on CPU or GPU. Multi-GPU support is very easy. Please refer to the examples to see how multi-GPU setting is used.

## Quick try

After building and installing Minerva and Owl package (python binding) as in [**Install Minerva**](https://github.com/dmlc/minerva/wiki/Install-Minerva). Try run `./run_owl_shell.sh` in Minerva's root directory. And enter:
```python
>>> x = owl.ones([10, 5])
>>> y = owl.ones([10, 5])
>>> z = x + y
>>> z.to_numpy()
```
The result will be a 10x5 array filled by value 2. Minerva supports many `numpy` style ndarray operations. Please see the API [document](http://minerva-developers.github.io/minerva-doc/) for more information.

## Features
* N-D array programming interface and easy integration with `numpy`

  ```python
  >>> import numpy as np
  >>> x = np.array([1, 2, 3])
  >>> y = owl.from_numpy(x)
  >>> y += 1
  >>> y.to_numpy()
  array([ 2., 3., 4., ], dtype=float32)
  ```
  More is in the [**API cheatsheet**](http://minerva-developers.github.io/minerva-doc/cheatsheet.html)
* Automatically parallel execution

  ```python
  >>> x = owl.zeros([256, 128])
  >>> y = owl.randn([1024, 32], 0.0, 0.01)
  ```
  The above `x` and `y` will be executed **concurrently**. How is this achieved?
  
  See [**Feature Highlight: Data-flow and lazy evaluation**](https://github.com/dmlc/minerva/wiki/Feature-Highlight:-Dataflow-engine)
* Multi-GPU, multi-CPU support:

  ```python
  >>> owl.set_device(gpu0)
  >>> x = owl.zeros([256, 128])
  >>> owl.set_device(gpu1)
  >>> y = owl.randn([1024, 32], 0.0, 0.01)
  ```
  The above `x` and `y` will be executed on two cards **simultaneously**. How is this achieved?
  
  See [**Feature Highlight: Multi GPU Training**](https://github.com/dmlc/minerva/wiki/Feature-Highlight:-Multi-GPU-Training)

## Tutorial and Documents
* Tutorials and high-level concepts could be found in [our wiki page](https://github.com/dmlc/minerva/wiki)
* A step-by-step walk through on MNIST example could be found [here](https://github.com/dmlc/minerva/wiki/Walkthrough:-MNIST)
* We also built a tool to directly read Caffe's configure file and train. See [document](https://github.com/dmlc/minerva/wiki/Walkthrough:-AlexNet).
* API documents could be found [here](http://minerva-developers.github.io/minerva-doc/index.html)

## Performance

We will keep updating the latest performance we could achieve in this section.

### Training speed

| Training speed <br> (images/second) | AlexNet | VGGNet | GoogLeNet |
|:------------------------------:|:-------:|:------:|:---------:|
| 1 card | 189.63 | 14.37 | 82.47 |
| 2 cards| 371.01 | 29.58 | 160.53 |
| 4 cards| 632.09 | 50.26 | 309.27 |
* The performance is measured on a machine with 4 GTX Titan cards.
* On each card, we load minibatch size of 256, 24, 120 for AlexNet, VGGNet and GoogLeNet respectively. Therefore, the total minibatch size will increase as the number of cards grows (for example, training AlexNet on 4 cards will use 1024 minibatch size).

### An end-to-end training

We also provide some end-to-end training codes in `owl` package, which could load Caffe's model file and perform training. Note that, Minerva is *not* the same tool as Caffe. We are not focusing on this part of logic. In fact, we implement these just to play with the Minerva's powerful and flexible programming interface (we could implement a Caffe-like network trainer in around 700~800 lines of python codes). Here is the training error with time compared with Caffe. Note that Minerva could finish GoogleNet training in less than four days with four GPU cards.

![Error curve](https://cloud.githubusercontent.com/assets/4057701/6857873/454c44b2-d3e0-11e4-9010-9e62c6c94027.jpg)

### Testing error rate
We trained several models using Minerva from scratch to show the correctness. The following table shows the error rate of different network under different testing settings.

| Testing error rate | AlexNet | VGGNet | GoogLeNet |
|:------------------------------:|:-------:|:------:|:---------:|
| single view top-1 | 41.6% | 31.6% | 32.7% |
| multi view top-1 | 39.7% | 30.1% | 31.3% |
| single view top-5 | 18.8% | 11.4% | 11.8% |
| multi view top-5 | 17.5% | 10.8% | 11.0% |

* AlexNet is trained with the [solver](https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/solver.prototxt) except for we didn't use multi-group convolution.
* GoogLeNet is trained with the [quick_solver](https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/quick_solver.prototxt).
* We didn't train VGGNet from scratch. We just transform the model into Minerva format and testing.

The models can be found in the following link:
[AlexNet](http://pan.baidu.com/s/1bnAT10b) [GoogLeNet](http://pan.baidu.com/s/1df67G) [VGGNet](http://pan.baidu.com/s/1pJIC5sf)

You can try then on your own machine.

## Next Plan
* Get rid of boost library dependency by using Cython.
* Large scale [LSTM](http://en.wikipedia.org/wiki/Long_short_term_memory) example using Minerva.
* Easy support for user-defined new operations.

## License and support

Minerva is provided in the Apache V2 open source license.

You can use the "issues" tab in github to report bugs. For non-bug issues, please send up an email at minerva-support@googlegroups.com. You can subscribe to the discussion group: https://groups.google.com/forum/#!forum/minerva-support.

## Wiki

For more information on how to install, use or contribute to Minerva, please visit our wiki page: https://github.com/minerva-developers/minerva/wiki

