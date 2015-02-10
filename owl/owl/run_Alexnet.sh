#python net_trainer.py /home/minjie/caffe/caffe/models/bvlc_alexnet/train_val_minerva.prototxt /home/minjie/caffe/caffe/models/bvlc_alexnet/solver.prototxt /home/tianjun/releaseversion/minerva/owl/apps/imagenet_googlenet/Alexmodel/epoch0/ loss
#python gradient_checker.py /home/minjie/caffe/caffe/models/bvlc_alexnet/train_val_minerva.prototxt /home/minjie/caffe/caffe/models/bvlc_alexnet/solver.prototxt /home/tianjun/releaseversion/minerva/owl/apps/imagenet_googlenet/Alexmodel/epoch0/ loss fc8

python net_trainer.py /home/tianjun/caffe/caffe/models/bvlc_alexnet/train_val_minerva.prototxt /home/tianjun/caffe/caffe/models/bvlc_alexnet/solver.prototxt /home/tianjun/caffe/caffe/models/bvlc_alexnet/MinervaModel/epoch0/ loss
