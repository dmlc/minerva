import sys,os
import time
import math
import numpy as np
import subprocess

import imageio
import owl
import owl.elewise as ele
import owl.conv as conv

class MNISTCNNModel:
    def __init__(self):
        self.weights = []
        self.bias = []
        self.weightdelta = []
        self.biasdelta = []
        self.convs = [
            conv.Convolver(0, 0, 1, 1),
            conv.Convolver(2, 2, 1, 1),
        ];
        self.poolings = [
            conv.Pooler(2, 2, 2, 2, conv.pool_op.max),
            conv.Pooler(3, 3, 3, 3, conv.pool_op.max)
        ];
    def init_random(self):
        self.weights = [
            owl.randn([5, 5, 1, 16], 0.0, 0.1),
            owl.randn([5, 5, 16, 32], 0.0, 0.1),
            owl.randn([10, 512], 0.0, 0.1)
        ];
        self.weightdelta = [
            owl.zeros([5, 5, 1, 16]),
            owl.zeros([5, 5, 16, 32]),
            owl.zeros([10, 512])
        ];
        self.bias = [
            owl.zeros([16]),
            owl.zeros([32]),
            owl.zeros([10, 1])
        ];
        self.biasdelta = [
            owl.zeros([16]),
            owl.zeros([32]),
            owl.zeros([10, 1])
        ];


def print_training_accuracy(o, t, mbsize, prefix):
    predict = o.reshape([10, mbsize]).max_index(0)
    ground_truth = t.reshape([10, mbsize]).max_index(0)
    correct = (predict - ground_truth).count_zero()
    print prefix, 'error: {}'.format((mbsize - correct) * 1.0 / mbsize)

def bpprop(model, samples, label):
    num_layers = 9
    num_samples = samples.shape[-1]
    fc_shape = [512, num_samples]

    acts = [None] * num_layers
    errs = [None] * num_layers
    weightgrad = [None] * len(model.weights)
    biasgrad = [None] * len(model.bias)

    acts[0] = samples
    acts[1] = ele.relu(model.convs[0].ff(acts[0], model.weights[0], model.bias[0]))
    acts[2] = model.poolings[0].ff(acts[1])
    acts[3] = ele.relu(model.convs[1].ff(acts[2], model.weights[1], model.bias[1]))
    acts[4] = model.poolings[1].ff(acts[3])
    acts[5] = model.weights[2] * acts[4].reshape(fc_shape) + model.bias[2]

    out = conv.softmax(acts[5], conv.soft_op.instance)

    errs[5] = out - label
    errs[4] = (model.weights[2].trans() * errs[5]).reshape(acts[4].shape)
    errs[3] = ele.relu_back(model.poolings[1].bp(errs[4], acts[4], acts[3]), acts[3])
    errs[2] = model.convs[1].bp(errs[3], model.weights[1])
    errs[1] = ele.relu_back(model.poolings[0].bp(errs[2], acts[2], acts[1]), acts[1])

    weightgrad[2] = errs[5] * acts[4].reshape(fc_shape).trans()
    biasgrad[2] = errs[5].sum(1)
    weightgrad[1] = model.convs[1].weight_grad(errs[3], acts[2])
    biasgrad[1] = model.convs[1].bias_grad(errs[3])
    weightgrad[0] = model.convs[0].weight_grad(errs[1], acts[0])
    biasgrad[0] = model.convs[0].bias_grad(errs[1])
    return (out, weightgrad, biasgrad)

def train_network(model, num_epochs = 100, minibatch_size = 256, lr = 0.01, mom = 0.75, wd = 5e-4):
    owl.set_device(owl.create_gpu_device(0))
    count = 0
    # load data
    (train_data, test_data) = imageio.load_mb_from_mat("mnist_all.mat", minibatch_size)
    num_test_samples = test_data[0].shape[0]
    test_samples = owl.from_numpy(test_data[0]).reshape([28, 28, 1, num_test_samples])
    test_labels = owl.from_numpy(test_data[1])
    for i in xrange(num_epochs):
        print "---Epoch #", i
        last = time.time()
        for (mb_samples, mb_labels) in train_data:
            num_samples = mb_samples.shape[0]
            data = owl.from_numpy(mb_samples).reshape([28, 28, 1, num_samples])
            label = owl.from_numpy(mb_labels)
            out, weightgrad, biasgrad = bpprop(model, data, label)
            for k in range(3):
                model.weightdelta[k] = mom * model.weightdelta[k] - lr / num_samples * weightgrad[k] - lr * wd * model.weights[k]
                model.biasdelta[k] = mom * model.biasdelta[k] - lr / num_samples * biasgrad[k]
                model.weights[k] += model.weightdelta[k]
                model.bias[k] += model.biasdelta[k]
            
            count = count + 1
            if (count % 20) == 0:
                print_training_accuracy(out, label, num_samples, 'Training')
        print '----End of Epoch #', i, 'time:', time.time() - last
        # do test
        out, _, _  = bpprop(model, test_samples, test_labels)
        print_training_accuracy(out, test_labels, num_test_samples, 'Testing')

if __name__ == '__main__':
    owl.initialize(sys.argv)
    owl.create_cpu_device()
    model = MNISTCNNModel()
    model.init_random()
    train_network(model)
