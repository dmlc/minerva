Scripts for training and others
===============================

Train network
-------------

Use following command to start training given Caffe's solver and configure file
```bash
./net_trainer.py <solver_file> [-s SNAPSHOT] [-n NUM_GPU]
```
* `solver_file` is the file name in Caffe's [solver](https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/quick_solver.prototxt) format.
* `SNAPSHOT` is the index of the snapshot to start with (default: 0). If SNAPSHOT is not equal to 0, it means we continue training from the formal snapshot
* `NUM_GPU` is the number of gpu to use.
Example:
```bash
./net_trainer.py /path/to/solver.txt -s=0 -n=4
```

Test network
-----------------

Use following command to perform testing on the given trained network. We could get top-1 or top-5 accuracy under single view or multiview
```bash
./net_tester.py <solver_file> <softmax_layer_name> <accuracy_layer_name> [-s SNAPSHOT] [-g GPU_IDX]
```
* `solver_file` is caffe solver configure file.
* `softmax_layer_name` indicate the layer to produce softmax distribution.
* `accuracy_layer_name` indicate the layer to produce accuracy. If you want to get top-5 accuracy, make sure to declare the accuracy_param `top-k: 5` in the network configuration file.
* `SNAPSHOT` is the index of the snapshot to test with (default: 0).
* `GPU_IDX` is the id of the gpu on which you want the testing to be performed (default: 0).
* `MULTIVIEW` indicate whether to use multiview testing (default: 0).
Example:
```bash
./net_tester.py /path/to/solver.txt loss3/loss3 loss3/top-5 -s=0 -g=1 -m=1
```

Feature extracting
------------------

Use following command to extract the feature of a certain layer from the given trained network
```bash
./feature_extractor.py <solver_file> <feature_path> [-s SNAPSHOT] [-g GPU_IDX]
```
* `solver_file` is caffe solver configure file.
* `layer_name` is the name of the layer to extract feature
* The feature will be written to the `feature_path` in readable float format (*not* binary).
* `SNAPSHOT` is the index of the snapshot to test with (default: 0).
* `GPU_IDX` is the id of the gpu on which you want the testing to be performed (default: 0).
Example:
```bash
./feature_extractor.py /path/to/solver.txt fc6 -s=60 -g=1
```

Filter Visualizer
-----------------

Use the following command to show what the nuerons interested in from a given model
```bash
./filter_visualizer.py <solver_file> <layer_name> <result_path> [-s SNAPSHOT] [-g GPU_IDX]
```
* `solver_file` is caffe solver configure file.
* the filters in `layer_name` will be visualized
* The visualization result will be saved in `result_path`, each filter/neuron will get a jpg visualization result.
* `SNAPSHOT` is the index of the snapshot to test with (default: 0).
* `GPU_IDX` is the id of the gpu on which you want the testing to be performed (default: 0).
Example:
```bash
./filter_visualizer.py /path/to/solver.txt conv4 /path/to/result/folder -s=60 -g=0
```

Heatmap Visualizer
-----------------

Use the following command to show which part of the image activate the neurons
```bash
./heatmap_visualizer.py <solver_file> <layer_name> <result_path> [-s SNAPSHOT] [-g GPU_IDX]
```
* `solver_file` is caffe solver configure file.
* the filters in `layer_name` will be visualized
* The visualization result will be saved in `result_path`. For each testing image, we will generate a heatmap in jpg format.
* `SNAPSHOT` is the index of the snapshot to test with (default: 0).
* `GPU_IDX` is the id of the gpu on which you want the testing to be performed (default: 0).

Example:
```bash
./heatmap_visualizer.py /path/to/solver.txt conv4 /path/to/result/folder -s=60 -g=0
```

For more documents on how to use `NetTrainer` and `owl.net` package yourself, please see [here](https://github.com/dmlc/minerva/tree/master/owl/owl/net).
