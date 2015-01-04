import math
import sys
import time
import numpy as np

import owl
import owl.conv as co
import owl.elewise as ele
from alexnet import AlexModel
from imageio import ImageNetDataProvider

def print_training_accuracy(o, t, minibatch_size):
    predict = o.max_index(0)
    ground_truth = t.max_index(0)
    correct = (predict - ground_truth).count_zero()
    print 'Training error: {}'.format((minibatch_size - correct) * 1.0 / minibatch_size)
    sys.stdout.flush()


def train_network(model, num_epochs = 100, minibatch_size=256,
        dropout_rate = 0.5, eps_w = 0.01, eps_b = 0.01, mom = 0.9, wd = 0.0005):
    gpu0 = owl.create_gpu_device(0)
    owl.set_device(gpu0)
    num_weights = 8
    count = 0
    last = time.time()

    dp = ImageNetDataProvider(mean_file='/home/minjie/data/imagenet/imagenet_mean.binaryproto',
            train_db='/home/minjie/data/imagenet/ilsvrc12_train_lmdb',
            val_db='/home/minjie/data/imagenet/ilsvrc12_val_lmdb',
            test_db='/home/minjie/data/imagenet/ilsvrc12_test_lmdb')

    for i in xrange(num_epochs):
        print "---------------------Epoch #", i
        for (samples, labels) in dp.get_train_mb(minibatch_size):
            count = count + 1
            num_samples = samples.shape[0]
            data = owl.from_nparray(samples).reshape([227, 227, 3, num_samples])
            target = owl.from_nparray(labels)

            out, weightsgrad, biasgrad = model.train_one_mb(data, target, dropout_rate)
            out.start_eval()
            model.update(weightsgrad, biasgrad, num_samples, mom, eps_w, wd)

            if count % 3 == 0:
                print_training_accuracy(out, target, data.shape[-1])
                print "time: %s" % (time.time() - last)
                last = time.time()

if __name__ == '__main__':
    owl.initialize(sys.argv)
    cpu = owl.create_cpu_device()
    owl.set_device(cpu)
    model = AlexModel()
    model.init_random()
    train_network(model)

