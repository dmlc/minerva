# OwlNet: DNN training framework built on Minerva

## Overview
OwelNet is a DNN training framework build on Minerva python interface. The main purposes of this package are:

1. Provide a simple way for Minerva users to train deep neural network for computer vision problems.
1. Provide a prototype about how to build user applications utilizing the advantages of Minerva.

We borrow Caffe's well-defined network architecture protobuf but the execution is conducted in Minerva engine. It's a showcase of Minerva's flexibile interface (building Caffe's main functionality in several hundreds of lines) and computation efficiency (Multi-GPU training).

See also: https://github.com/dmlc/minerva/wiki/Walkthrough:-AlexNet

## Features
* Training complex DNN by simply running a script
* Easily utilizing Multi-GPU to train at full speed
  * Defining network architecture and training parameters using Caffe's protobuf definition
* Flexible IO (`numpy.ndarray`, Matlab `.mat` file, LMDB, Raw Image, Image Window)

## Usage
### Quick Example: Training GooLeNet Using 4 GPUs

1. Download the training and validation set of ILSVRC12.
1. Download the image list files (train.txt and val.txt) of ILSVRC12 using [get_ilsvrc_aux.sh](https://github.com/BVLC/caffe/blob/master/data/ilsvrc12/get_ilsvrc_aux.sh) provided by Caffe.
1. Create LMDB Data for train and val set using [create_imagenet.sh](https://github.com/BVLC/caffe/blob/master/examples/imagenet/create_imagenet.sh) provided by Caffe.
1. Compute mean_file for the training set of ILSVRC12 using [make_imagenet_mean.sh](https://github.com/BVLC/caffe/blob/master/examples/imagenet/make_imagenet_mean.sh) provided by Caffe.
1. Write the configuration file of GoogleNet. You can use this [train_val.prototxt](https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/train_val.prototxt) and then reset the path of the lmdb and mean file.
1. Set the training parameters through solver file. You can use this [quick_solver.prototxt](https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/quick_solver.prototxt) and the reset the path of `train_val.prototxt` and snapshot_prefix to where you store.
1. With everything ready, we can start training by calling in `/path/to/minerva/scripts/`:

  ```bash
  ./net_trainer /path/to/solver -n 4
  ```
  The number 4 means the network will be trained using 4 GPUs.
1. If you want to restart the training from snapshot `N`. You can call:

  ```bash
  ./net_trainer /path/to/solver -n 4 --snapshot N
  ```

###Data Preparation
  We load data into `owl.NArray` through `numpy.ndarray` by the function 'owl.from_array'. So to handle IO, we only need to provide several ways to load original data into `numpy.ndarray`. When data is prepared, we will inform Minerva about the path and type of the data when configuring network to creat data provider automatically.

  * *Image List Data*: If you do not pursuit IO speed, you can just provide the image list containing image path and label. The format of the image list file is `image_path label`, each line indicates an image. You can find example in [data_list](http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz).

  * *LMDB Data*: LMDB is an ultra-fast, ultra-compact key-value embeded data store. It has relatively good read performance when we access the data in round-robin style. To save dataset into LMDB, we set the keys as `data` and `label`. To change raw image into LMDB, you can use the tool provided by Caffe: [create_imagenet.sh](https://github.com/BVLC/caffe/blob/master/examples/imagenet/create_imagenet.sh).

  * *Matlab Data*: If the dataset is stored in Matlab file, e.g. [mnist_all.mat](https://code.google.com/p/kernelmachine/downloads/detail?name=mnist_all.mat). We could also convert it into `numpy.ndarray`.

###Network Definition
####Layer Abstraction
  We use the same layer abstraction as in Caffe. We implement part of Caffe's functionality using Minerva interface and use Caffe's network protobuf(https://github.com/BVLC/caffe/tree/master/src/caffe/proto) to describe network configuration. Currently OwlNet can run [AlexNet](www.cs.toronto.edu/~fritz/absps/imagenet.pdf), [VGGNet](http://arxiv.org/pdf/1409.1556.pdf) and [GoogLeNet](http://arxiv.org/pdf/1409.4842v1.pdf). Below is the layers we have implemented.

1. Data Layer
  * `LMDBDataUnit`: Create LMDBDataProvider to use LMDB to handle IO.
  * `ImageDataUnit`: Create ImageListDataProvider to read from images as input.
  * `ImageWindowDataUnit`: Create ImageWindowDataProvider to read cropped window from image as input

1. Connection Layer
  * `FullyConnection`: Feed-forward is conducted by inner product of weight and activation.
  * `ConvConnection`: Feed-forward is conducted by using filters to convolve on activation.
  * `PoolingUnit`: Feed-forward is conducted by pooling. Max pooling and average pooling are provided.
  * `ConcatUnit`: Concatenating some layers into one layer on a certain dimension.

1. Activation Layer
  * `LinearUnit`
  * `ReluUnit`
  * `SigmoidUnit`
  * `TanhUnit`

1. Other Layer
  * `SoftmaxUnit`: Softmax loss layer.
  * `AccuracyUnit`: Compute top-1 error.
  * `DropoutUnit`: Introduce dropout on activation.
  * `LRNUnit`: Local response normalization layer.

More information could be found [here](https://github.com/minerva-developers/minerva/blob/master/owl/owl/net/net.py).

###Training
  When you have data prepared and network configured, we can start training. 

####Solver
  Training hyperparameters are described in the solver file. We use Caffe's [solver format](https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/solver.prototxt). The information need to be passed through solver are:
  * network configuration file
  * snapshot saving directory
  * max iteration
  * testing interval
  * test interation
  * snapshot saving interval
  * learning rate tuning strategy
  * momentun
  * weight decay

####Running command
  We have implement the running logic, all users need to do is to call a script. The format of the call is:
  
  ```bash
  /path/to/minerva/scripts/net_trainer <path-to-solver> [-n NUM_GPU] [--snapshot SNAPSHOT]
  ```

####Resume Training
  When `SNAPSHOT` is not zero, our code will try to load the snapshot of that index and continue training. If the weight dimension of a layer is not the same with the snapshot, our code will automatically reinitilize that weight, it allows easily finetuning on other datasets.

####Multi-GPU
  When `NUM_GPU` is greater than one, our code will automatically dispatch data batch to multi-gpu to train in parallel. We apply synchronize update, thus the result is the same with the one training using one GPU.
