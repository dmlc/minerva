import math
import Queue
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

class AsyncQueue:
    def __init__(self, thres):
        self.thres = thres
        self.q = Queue()

    def enqueue(self, i):
        self.q.put(i)
        while thres < self.q.qsize():
            self.q.get_nowait().wait_for_eval()
            print "time: %s" % (time.time() - last)


def train_network(model, num_epochs = 100, minibatch_size=256,
        dropout_rate = 0.5, eps_w = 0.01, eps_b = 0.01, mom = 0.9, wd = 0.0005):
    gpu = [None] * 2
    gpu[0] = owl.create_gpu_device(0)
    gpu[1] = owl.create_gpu_device(1)
    num_layers = 20
    num_weights = 8
    count = 0
    last = time.time()

    dp = ImageNetDataProvider(mean_file='/home/minjie/data/imagenet/imagenet_mean.binaryproto',
            train_db='/home/minjie/data/imagenet/ilsvrc12_train_lmdb',
            val_db='/home/minjie/data/imagenet/ilsvrc12_val_lmdb',
            test_db='/home/minjie/data/imagenet/ilsvrc12_test_lmdb')

    minibatch_size = minibatch_size / 2

    wgrad = [None] * 2
    bgrad = [None] * 2

    for i in xrange(num_epochs):
        print "---------------------Epoch #", i
        for (samples, labels) in dp.get_train_mb(minibatch_size):
            count = count + 1
            gpuid = count % 2
            owl.set_device(gpu[gpuid])

            data = owl.from_nparray(samples).reshape([227, 227, 3, samples.shape[0]])
            label = owl.from_nparray(labels)
            num_samples += data.shape[-1]
            (out[gpuid], wgrad[gpuid], bgrad[gpuid]) = model.train_one_mb(data, label, dropout_rate)
            out.start_eval()

            if count % 2 != 0:
                continue

            for k in range(num_weights):
                wgrad[0][k] += wgrad[1][k]
                bgrad[0][k] += bgrad[1][k]

            model.update(wgrad[0], bgrad[0], num_samples, mom, eps_w, wd)

            if count % 8 == 0:
                print_training_accuracy(out[0], label, data.shape[-1])
                print "time: %s" % (time.time() - last)
                last = time.time()

            num_samples = 0
            wgrad = [None] * 2
            bgrad = [None] * 2

if __name__ == '__main__':
    owl.initialize(sys.argv)
    cpu = owl.create_cpu_device()
    owl.set_device(cpu)
    model = AlexModel()
    model.init_random()
    train_network(model)
