import sys,os
import math
import numpy as np
import scipy.io as si

import owl
import owl.elewise as ele
from owl.conv import *

def extract(prefix, md, max_dig):
    for dig in range(max_dig):
        samples = md[prefix + str(dig)]
        labels = np.empty([samples.shape[0], 1], dtype=np.float32)
        labels.fill(dig * 256)
        yield np.hstack((samples, labels)) / 256

def split_sample_and_label(merged_mb):
    [s, l] = np.hsplit(merged_mb, [merged_mb.shape[1]-1])
    # change label to sparse representation
    n = merged_mb.shape[0]
    ll = np.zeros([n, 10], dtype=np.float32)
    ll[np.arange(n), l.astype(int).flat] = 1
    return (s, ll);

def load_mb_from_mat(mat_file, mb_size):
    # load from mat
    md = si.loadmat(mat_file)
    # merge all data
    train_all = np.concatenate(tuple(extract('train', md, 10)))
    test_all = np.concatenate(tuple(extract('test', md, 10)))
    # shuffle
    np.random.shuffle(train_all)
    # make minibatch
    train_mb = np.vsplit(train_all, range(mb_size, train_all.shape[0], mb_size))
    train_data = map(split_sample_and_label, train_mb)
    test_data = split_sample_and_label(test_all)
    print 'Training data: %d mini-batches' % len(train_mb)
    print 'Test data: %d samples' % test_all.shape[0]
    return (train_data, test_data)

class MNISTCNNModel:
    def __init__(self):
        self.weights = []
        self.bias = []
        self.convs = [
            Convolution(conv_info(0, 0, 1, 1)),
            Convolution(conv_info(2, 2, 1, 1)),
        ];
        self.poolings = [
            Pooling(pooling_info(2, 2, 2, 2, pool_op.max)),
            Pooling(pooling_info(3, 3, 3, 3, pool_op.max))
        ];
    def init_random(self):
        self.weights = [
            owl.randn([5, 5, 1, 16], 0.0, 0.1),
            owl.randn([5, 5, 16, 32], 0.0, 0.1),
            owl.randn([10, 512], 0.0, 0.1)
        ];
        self.bias = [
            owl.zeros([16]),
            owl.zeros([32]),
            owl.zeros([10, 1])
        ];

def print_training_accuracy(o, t, mbsize):
    predict = o.reshape([10, mbsize]).max_index(0)
    ground_truth = t.reshape([10, mbsize]).max_index(0)
    correct = (predict - ground_truth).count_zero()
    print 'Training error: {}'.format((mbsize - correct) * 1.0 / mbsize)

def train_network(model, num_epochs = 100, num_train_samples = 60000, minibatch_size = 256, eps_w = 0.01, eps_b = 0.01):
    np.set_printoptions(linewidth=200)
    owl.set_device(owl.create_gpu_device(0))
    num_layers = 9
    count = 0
    # load data
    (train_data, test_data) = load_mb_from_mat("mnist_all.mat", minibatch_size)
    num_test_samples = test_data[0].shape[0]
    (test_samples, test_labels) = map(lambda npdata : owl.from_nparray(npdata), test_data)
    for i in xrange(num_epochs):
        print "---Epoch #", i
        for (mb_samples, mb_labels) in train_data:
            num_samples = mb_samples.shape[0]

            acts = [None] * num_layers
            sens = [None] * num_layers

            acts[0] = owl.from_nparray(mb_samples).reshape([28, 28, 1, num_samples])
            target = owl.from_nparray(mb_labels)

            acts[1] = model.convs[0].ff(acts[0], model.weights[0], model.bias[0])
            acts[2] = ele.relu(acts[1])
            acts[3] = model.poolings[0].ff(acts[2])
            acts[4] = model.convs[1].ff(acts[3], model.weights[1], model.bias[1])
            acts[5] = ele.relu(acts[4])
            acts[6] = model.poolings[1].ff(acts[5])
            re_acts6 = acts[6].reshape([np.prod(acts[6].shape[0:3]), num_samples])
            acts[7] = model.weights[2] * re_acts6 + model.bias[2]
            acts[8] = softmax(acts[7], soft_op.instance)
            
            sens[7] = acts[8] - target

            sens[6] = (model.weights[2].trans() * sens[7]).reshape(acts[6].shape)
            sens[5] = model.poolings[1].bp(sens[6], acts[6], acts[5])
            sens[4] = ele.relu_back(sens[5], acts[5], acts[4])
            sens[3] = model.convs[1].bp(sens[4], model.weights[1])
            sens[2] = model.poolings[0].bp(sens[3], acts[3], acts[2])
            sens[1] = ele.relu_back(sens[2], acts[2], acts[1])

            model.weights[2] -= eps_w / num_samples * sens[7] * re_acts6.trans()
            model.bias[2] -= eps_b / num_samples * sens[7].sum(1)
            model.weights[1] -= eps_w / num_samples * model.convs[1].weight_grad(sens[4], acts[3])
            model.bias[1] -= eps_b / num_samples * model.convs[1].bias_grad(sens[4])
            model.weights[0] -= eps_w / num_samples * model.convs[0].weight_grad(sens[1], acts[0])
            model.bias[0] -= eps_b / num_samples * model.convs[0].bias_grad(sens[1])

            count = count + 1
            if (count % 40) == 0:
                print_training_accuracy(acts[-1], target, num_samples)

if __name__ == '__main__':
    owl.initialize(sys.argv)
    owl.create_cpu_device()
    model = MNISTCNNModel()
    model.init_random()
    train_network(model)
