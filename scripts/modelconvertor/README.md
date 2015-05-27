Scripts for converting models between Minerva and Caffe
=======================================================
Converting from Caffe to Minerva
--------------------------------

Use following command to convert a Caffe model into Minerva model
```bash
./caffe2minerva.py <caffe_model> <minerva_model_dir> <snapshot>
```
* `caffe_model` the path to the caffe model.
* `minerva_model_dir` the directory to save converted minerva model.
* `snapshot` the model snapshot idx to be saved as.

Example:
```bash
./caffe2minerva.py /path/to/caffe/models/bvlc_googlenet/bvlv_googlenet.caffemodel /path/to/minerva/models/ 60
```

Converting from Minerva to Caffe
--------------------------------

Use following command to convert a Minerva model into Caffe model
```bash
./minerva2caffe.py <config_file> <minerva_model_dir> <snapshot> <caffe_model>
```
* `config_file` the configure file describing the network, for [example](https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/train_val.prototxt).
* `minerva_model_dir` the directory to save converted minerva model.
* `snapshot` the model snapshot to be converted.
* `caffe_model` the path to the converted caffe model.

Example:
```bash
./minerva2caffe.py /path/to/caffe/models/bvlc_googlenet/train_val.prototxt /path/to/minerva/models/ 60 /path/to/new/caffe/model
```


