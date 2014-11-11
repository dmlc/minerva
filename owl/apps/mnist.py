import sys,os
import math
<<<<<<< HEAD
import scipy
import libowl
import sys
import random
import numpy

from libowl import *
from scipy.io import *
from random import shuffle
from numpy import reshape

class Layer(object):

    def __init__(self, length):
        self.length = length
        self.bias = libowl.zeros([length, 1])

def make_vector(x, y):
    result = []
    for i in xrange(x):
        if (i == y):
            result.append(1)
        else:
            result.append(0)
    return result

def init_network(layers, weights, data, label, minibatch_size = 256, num_minibatches = 235):
    print 'init network'
    num_layers = len(layers)
    for i in xrange(num_layers - 1):
        row = layers[i + 1].length
        col = layers[i].length
        var = math.sqrt(4.0 / (row + col))  # XXX: variance or stddev?
        weights.append(libowl.randn([row, col], 0.0, var))

    loader = scipy.io.loadmat('mnist_all.mat')
    tmp_data = []
    tmp_label = []
    for i in xrange(10):
        key = 'train' + str(i)
        lvector = make_vector(10, i)
        for j in xrange(len(loader[key])):
            tmp_data.append(loader[key][j])
            tmp_label.append(lvector)
    idx = []
    for i in xrange(len(tmp_data)):
        idx.append(i)
    random.shuffle(idx)

    s = 0
    t = minibatch_size
    l = layers[0].length
    for i in xrange(num_minibatches):
        minibatch_data = [tmp_data[j] for j in idx[s:t]]
        minibatch_label = [tmp_label[j] for j in idx[s:t]]

        data.append(reshape(minibatch_data, (t - s) * l).tolist())
        label.append(reshape(minibatch_label, (t - s) * 10).tolist())

        s = s + minibatch_size
        t = min(t + minibatch_size, len(tmp_data))

def softmax(m):
    maxval = m.max(0)
    centered = m.norm_arithmetic(maxval, libowl.arithmetic.sub)
    class_normalizer = libowl.ln(libowl.exp(centered).sum(0)) + maxval
    return libowl.exp(m.norm_arithmetic(class_normalizer, libowl.arithmetic.sub))

def print_training_accuracy(o, t, minibatch_size):
    predict = o.max_index(0)
    ground_truth = t.max_index(0)
    correct = (predict - ground_truth).count_zero()
    print 'Training error: {}'.format((minibatch_size - correct) * 1.0 / minibatch_size)

def train_network(layers, weights, data, label,
                  num_epochs=100, num_train_samples=60000, minibatch_size=256,
                  num_minibatches=235, eps_w=0.01, eps_b=0.01):
    cpuDevice = create_cpu_device()
    gpuDevice0 = create_gpu_device(0)
    gpuDevice1 = create_gpu_device(1)

    num_layers = len(layers)
    for i in xrange(num_epochs):
        print "epoch ", i
        for j in xrange(num_minibatches):
            set_device(cpuDevice)
            acts = [None] * num_layers
            sens = [None] * num_layers
            #acts[0] = libowl.zeros([layers[0].length, len(data[j]) / layers[0].length])
            #target = libowl.zeros([10, len(label[j]) / 10])
            acts[0] = make_narray([layers[0].length, len(data[j]) / layers[0].length], data[j])
            target = make_narray([10, len(label[j]) / 10], label[j])

            set_device(gpuDevice0)
            # FF
            for k in xrange(1, num_layers):
                acts[k] = weights[k - 1] * acts[k - 1]
                acts[k] = acts[k].norm_arithmetic(layers[k].bias, libowl.arithmetic.add)
                if k < (num_layers - 1):
                    acts[k] = sigmoid(acts[k])
            # Error
            acts[-1] = softmax(acts[-1])
            sens[-1] = acts[-1] - target
            # BP

            #set_device(gpuDevice1)
            for k in reversed(xrange(num_layers - 1)):
                d_act = libowl.mult(acts[k], 1 - acts[k])
                sens[k] = weights[k].trans() * sens[k + 1];
                # This is element-wise.
                sens[k] = libowl.mult(sens[k], d_act)
            # Update bias
            for k in xrange(1, num_layers):
                layers[k].bias -= eps_b * sens[k].sum(1) / minibatch_size
            # Update weight
            for k in xrange(num_layers - 1):
                weights[k] -= eps_w * sens[k + 1] * acts[k].trans() / minibatch_size
=======
import owl
import owl.elewise
import scipy.io as si
import numpy as np

def extract(prefix, md, max_dig):
    for dig in range(max_dig):
        samples = md[prefix + str(dig)]
        labels = np.empty([samples.shape[0], 1], dtype=np.float32)
        labels.fill(dig)
        yield np.hstack((samples, labels))

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
    #print train_data[0]
    #print train_data[0][0].dtype, train_data[0][1].dtype
    #print test_data
    print 'Training data: %d mini-batches' % len(train_mb)
    print 'Test data: %d samples' % test_all.shape[0]
    return (train_data, test_data)


class MnistTrainer:
    def __init__(self, data_file='mnist_all.mat', num_epochs=100, mb_size=256, eps_w=0.01, eps_b=0.01):
        self.cpu = owl.create_cpu_device()
        self.gpu = owl.create_gpu_device(0)
        self.data_file = data_file
        self.num_epochs=num_epochs
        self.mb_size=mb_size
        self.eps_w=eps_w
        self.eps_b=eps_b
        # init weight
        l1 = 784; l2 = 256; l3 = 10
        self.l1 = l1; self.l2 = l2; self.l3 = l3
        self.w1 = owl.randn([l2, l1], 0.0, math.sqrt(4.0 / (l1 + l2)))
        self.w2 = owl.randn([l3, l2], 0.0, math.sqrt(4.0 / (l2 + l3)))
        self.b1 = owl.zeros([l2, 1])
        self.b2 = owl.zeros([l3, 1])

    def run(self):
        (train_data, test_data) = load_mb_from_mat(self.data_file, self.mb_size)
        np.set_printoptions(linewidth=200)
        num_test_samples = test_data[0].shape[0]
        (test_samples, test_labels) = map(lambda npdata : owl.from_nparray(npdata.T), test_data)
        count = 1
        for epoch in range(self.num_epochs):
            print '---Start epoch #%d' % epoch
            # train
            for (mb_samples, mb_labels) in train_data:
                num_samples = mb_samples.shape[0]

                owl.set_device(self.cpu)
                a1 = owl.from_nparray(mb_samples.T)
                target = owl.from_nparray(mb_labels.T)
                owl.set_device(self.gpu)

                # ff
                a2 = owl.elewise.sigmoid((self.w1 * a1).norm_arithmetic(self.b1, owl.op.add))
                a3 = owl.elewise.sigmoid((self.w2 * a2).norm_arithmetic(self.b2, owl.op.add))
                # softmax & error
                out = owl.softmax(a3)
                s3 = out - target
                # bp
                s3 = owl.elewise.mult(s3, 1 - s3)
                s2 = self.w2.trans() * s3
                s2 = owl.elewise.mult(s2, 1 - s2)
                # grad
                gw1 = s2 * a1.trans() / num_samples
                gb1 = s2.sum(1) / num_samples
                gw2 = s3 * a2.trans() / num_samples
                gb2 = s3.sum(1) / num_samples
                # update
                self.w1 -= self.eps_w * gw1
                self.w2 -= self.eps_w * gw2
                self.b1 -= self.eps_b * gb1
                self.b2 -= self.eps_b * gb2

                if (count % 40 == 0):
                    correct = out.max_index(0) - target.max_index(0)
                    val = correct.tolist()
                    print 'Training error:', (float(num_samples) - val.count(0.0)) / num_samples
                    # test
                    a1 = test_samples
                    a2 = owl.elewise.sigmoid((self.w1 * a1).norm_arithmetic(self.b1, owl.op.add))
                    a3 = owl.elewise.sigmoid((self.w2 * a2).norm_arithmetic(self.b2, owl.op.add))
                    correct = a3.max_index(0) - test_labels.max_index(0)
                    val = correct.tolist()
                    #print val
                    print 'Testing error:', (float(num_test_samples) - val.count(0.0)) / num_test_samples
                count = count + 1

            # test
            #a1 = test_samples
            #a2 = owl.elewise.sigmoid((self.w1 * a1).norm_arithmetic(self.b1, owl.op.add))
            #a3 = owl.elewise.sigmoid((self.w2 * a2).norm_arithmetic(self.b2, owl.op.add))
            #out = owl.softmax(a3)
            #correct = out.max_index(0) - test_labels.max_index(0)
            #val = correct.tolist()
            #print 'Testing error:', (float(num_test_samples) - val.count(0.0)) / num_test_samples

            print '---Finish epoch #%d' % epoch
>>>>>>> master

            if ((j % 20) == 0):
                set_device(cpuDevice)
                print_training_accuracy(acts[-1], target, minibatch_size)

if __name__ == '__main__':
    owl.initialize(sys.argv)
    trainer = MnistTrainer(num_epochs = 10)
    trainer.run()
