Scripts for training and others
===============================

Train network
-------------

Use following command to start training given Caffe's solver and configure file
```bash
./net_trainer.py <solver_file> [--snapshot SNAPSHOT] [-n NUM_GPU]
```
* `solver_file` is the file name in Caffe's [solver](https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/quick_solver.prototxt) format.
* `SNAPSHOT` is the index of the snapshot to start with (default: 0).
* `NUM_GPU` is the number of gpu to use.

Multiview testing
-----------------

Use following command to perform multi-view testing on the given trained network
```bash
./multiview_test.py <solver_file> <snapshot> [-g GPU_IDX]
```
* `GPU_IDX` is the id of the gpu on which you want the multi-view testing to be performed (default: 0).

Feature extracting
------------------

Use following command to extract the feature of a certain layer from the given trained network
```bash
./feature_extractor.py <solver_file> <snapshot> <layer_name> <feature_path> [-g GPU_IDX]
```
* `layer_name` is the name of the layer to extract feature
* The feature will be written to the `feature_path` in readable float format (*not* binary).

For more documents on how to use `NetTrainer` and `owl.net` package yourself, please see [here](https://github.com/dmlc/minerva/tree/master/owl/owl/net).
