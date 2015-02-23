import sys
import math
import time
import numpy as np
from caffe_data_pb2 import NetParameter
from caffe_data_pb2 import LayerParameter
from google.protobuf import text_format
import owl
import owl.conv as conv
import owl.elewise as ele
import owl.net as net
from imageio import ImageNetDataProvider
from createcaffemodel import CaffeModelConfig
from createcaffemodel import MinervaModel


def print_training_accuracy(o, t, minibatch_size):
    predict = o.max_index(0)
    ground_truth = t.max_index(0)
    correct = (predict - ground_truth).count_zero()
    print 'Training error: {}'.format((minibatch_size - correct) * 1.0 / minibatch_size)
    sys.stdout.flush()

def train_network(model, num_epochs = 100, minibatch_size=10,
        dropout_rate = 0.5, eps_w = 0.01, mom = 0.9, wd = 0.0005):
    gpu0 = owl.create_gpu_device(0)
    owl.set_device(gpu0)
    num_weights = 8
    count = 0
    last = time.time()
    cropped_size = 224

    dp = ImageNetDataProvider(mean_file='/home/minjie/data/imagenet/imagenet_mean.binaryproto',
            train_db='/home/minjie/data/imagenet/ilsvrc12_train_lmdb',
            val_db='/home/minjie/data/imagenet/ilsvrc12_val_lmdb',
            test_db='/home/minjie/data/imagenet/ilsvrc12_test_lmdb')

    #mark the output layer
    output_layer = 'loss3/loss3'

    for i in xrange(num_epochs):
        print "---------------------Epoch #", i
        for (samples, labels) in dp.get_train_mb(minibatch_size, cropped_size):
            count = count + 1
            num_samples = samples.shape[0]
            data = owl.from_numpy(samples).reshape([cropped_size, cropped_size, 3, num_samples])
            target = owl.from_numpy(labels)
            model.ff(data, target)
            print_training_accuracy(model.layers[output_layer].get_act(), target, minibatch_size)
            model.bp(data, target)
            model.update(num_samples,eps_w, mom, wd)
            

            exit(0)


if __name__ == '__main__':
    owl.initialize(sys.argv)
    cpu = owl.create_cpu_device()
    owl.set_device(cpu)
    #newconfig = CaffeModelConfig(configfile = '/home/minjie/caffe/caffe/models/VGG/VGG_train_val.prototxt')
    newconfig = CaffeModelConfig(configfile = '/home/tianjun/athena/athena/owl/apps/GoogLeNet/train_val.prototxt')
    model = MinervaModel(newconfig.netconfig, './Googmodel')
    train_network(model)
