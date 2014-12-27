import math
import Queue
import sys
import time
import numpy as np

import owl
import owl.conv as co
import owl.elewise as ele
from imageio import ImageNetDataProvider
from PIL import Image

class AlexModel:
    def __init__(self):
        self.weights = []
	self.weightsdelta = []
        self.bias = []
        self.biasdelta = []
	self.convs = [
            co.Convolver(0, 0, 4, 4), # conv1
            co.Convolver(2, 2, 1, 1), # conv2
            co.Convolver(1, 1, 1, 1), # conv3
            co.Convolver(1, 1, 1, 1), # conv4
            co.Convolver(1, 1, 1, 1)  # conv5
        ];
        self.poolings = [
            co.Pooler(3, 3, 2, 2, co.pool_op.max), # pool1
            co.Pooler(3, 3, 2, 2, co.pool_op.max), # pool2
            co.Pooler(3, 3, 2, 2, co.pool_op.max)  # pool5
        ];

    def init_random(self):
        self.weights = [
            owl.randn([11, 11, 3, 96], 0.0, 0.01),
            owl.randn([5, 5, 96, 256], 0.0, 0.01),
            owl.randn([3, 3, 256, 384], 0.0, 0.01),
            owl.randn([3, 3, 384, 384], 0.0, 0.01),
            owl.randn([3, 3, 384, 256], 0.0, 0.01),
            owl.randn([4096, 9216], 0.0, 0.01),
            owl.randn([4096, 4096], 0.0, 0.01),
            owl.randn([1000, 4096], 0.0, 0.01)
        ];

	self.weightsdelta = [
            owl.zeros([11, 11, 3, 96]),
            owl.zeros([5, 5, 96, 256]),
            owl.zeros([3, 3, 256, 384]),
            owl.zeros([3, 3, 384, 384]),
            owl.zeros([3, 3, 384, 256]),
            owl.zeros([4096, 9216]),
            owl.zeros([4096, 4096]),
            owl.zeros([1000, 4096])
        ];

        self.bias = [
            owl.zeros([96]),
            owl.zeros([256]),
            owl.zeros([384]),
            owl.zeros([384]),
            owl.zeros([256]),
            owl.zeros([4096, 1]),
            owl.zeros([4096, 1]),
            owl.zeros([1000, 1])
        ];

        self.biasdelta = [
            owl.zeros([96]),
            owl.zeros([256]),
            owl.zeros([384]),
            owl.zeros([384]),
            owl.zeros([256]),
            owl.zeros([4096, 1]),
            owl.zeros([4096, 1]),
            owl.zeros([1000, 1])
        ];

def print_training_accuracy(o, t, minibatch_size):
    predict = o.max_index(0)
    ground_truth = t.max_index(0)
    correct = (predict - ground_truth).count_zero()
    print 'Training error: {}'.format((minibatch_size - correct) * 1.0 / minibatch_size)
    sys.stdout.flush()

def train_one_mb(model, data, label, dropout_rate):
    num_samples = data.shape[-1]
    num_layers = 20
    acts = [None] * num_layers
    sens = [None] * num_layers
    weightsgrad = [None] * 8
    biasgrad = [None] * 8

    # FF
    acts[0] = data
    acts[1] = ele.relu(model.convs[0].ff(acts[0], model.weights[0], model.bias[0])) # conv1
    acts[2] = model.poolings[0].ff(acts[1]) # pool1
    acts[3] = ele.relu(model.convs[1].ff(acts[2], model.weights[1], model.bias[1])) # conv2
    acts[4] = model.poolings[1].ff(acts[3]) # pool2
    acts[5] = ele.relu(model.convs[2].ff(acts[4], model.weights[2], model.bias[2])) # conv3
    acts[6] = ele.relu(model.convs[3].ff(acts[5], model.weights[3], model.bias[3])) # conv4
    acts[7] = ele.relu(model.convs[4].ff(acts[6], model.weights[4], model.bias[4])) # conv5
    acts[8] = model.poolings[2].ff(acts[7]) # pool5
    re_acts8 = acts[8].reshape([np.prod(acts[8].shape[0:3]), num_samples])
    acts[9] = ele.relu(model.weights[5] * re_acts8 + model.bias[5]) # fc6
    mask6 = owl.randb(acts[9].shape, dropout_rate)
    acts[9] = ele.mult(acts[9], mask6) # drop6
    acts[10] = ele.relu(model.weights[6] * acts[9] + model.bias[6]) # fc7
    mask7 = owl.randb(acts[10].shape, dropout_rate)
    acts[10] = ele.mult(acts[10], mask7) # drop7
    acts[11] = model.weights[7] * acts[10] + model.bias[7] # fc8

    out = co.softmax(acts[11], co.soft_op.instance) # prob

    sens[11] = out - label
    sens[10] = model.weights[7].trans() * sens[11] # fc8
    sens[10] = ele.mult(sens[10], mask7) # drop7
    sens[10] = ele.relu_back(sens[10], acts[10]) # relu7
    sens[9] = model.weights[6].trans() * sens[10]
    sens[9] = ele.mult(sens[9], mask6) # drop6
    sens[9] = ele.relu_back(sens[9], acts[9]) # relu6
    sens[8] = (model.weights[5].trans() * sens[9]).reshape(acts[8].shape) # fc6
    sens[7] = ele.relu_back(model.poolings[2].bp(sens[8], acts[8], acts[7]), acts[7]) # pool5, relu5
    sens[6] = ele.relu_back(model.convs[4].bp(sens[7], model.weights[4]), acts[6]) # conv5, relu4
    sens[5] = ele.relu_back(model.convs[3].bp(sens[6], model.weights[3]), acts[5]) # conv4, relu3
    sens[4] = model.convs[2].bp(sens[5], model.weights[2]) # conv3
    sens[3] = ele.relu_back(model.poolings[1].bp(sens[4], acts[4], acts[3]), acts[3]) # pool2, relu2
    sens[2] = model.convs[1].bp(sens[3], model.weights[1]) # conv2
    sens[1] = model.poolings[0].bp(sens[2], acts[2], acts[1]) # pool1
    sens[1] = ele.relu_back(sens[1], acts[1]) # relu1

    weightsgrad[7] = sens[11] * acts[10].trans()
    weightsgrad[6] = sens[10] * acts[9].trans()
    weightsgrad[5] = sens[9] * re_acts8.trans()
    weightsgrad[4] = model.convs[4].weight_grad(sens[7], acts[6])
    weightsgrad[3] = model.convs[3].weight_grad(sens[6], acts[5])
    weightsgrad[2] = model.convs[2].weight_grad(sens[5], acts[4])
    weightsgrad[1] = model.convs[1].weight_grad(sens[3], acts[2])
    weightsgrad[0] = model.convs[0].weight_grad(sens[1], acts[0])
    biasgrad[7] = sens[11].sum(1)
    biasgrad[6] = sens[10].sum(1)
    biasgrad[5] = sens[9].sum(1)
    biasgrad[4] = model.convs[4].bias_grad(sens[7])
    biasgrad[3] = model.convs[3].bias_grad(sens[6])
    biasgrad[2] = model.convs[2].bias_grad(sens[5])
    biasgrad[1] = model.convs[1].bias_grad(sens[3])
    biasgrad[0] = model.convs[0].bias_grad(sens[1])
    return (out, weightsgrad, biasgrad)

def train_network(model, num_epochs = 100, minibatch_size=256,
        dropout_rate = 0.5, eps_w = 0.01, eps_b = 0.01, mom = 0.9, wd = 0.0005):
    gpu0 = owl.create_gpu_device(0)
    gpu1 = owl.create_gpu_device(1)
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

            out, weightsgrad, biasgrad = train_one_mb(model, data, target, dropout_rate)

            for k in range(num_weights):
                model.weightsdelta[k] = mom * model.weightsdelta[k] - eps_w / num_samples  * weightsgrad[k] - eps_w * wd * model.weights[k]
                model.biasdelta[k] = mom * model.biasdelta[k] - eps_b / num_samples  * biasgrad[k]
                model.weights[k] += model.weightsdelta[k]
                model.weights[k].start_eval()
                model.bias[k] += model.biasdelta[k]
                model.bias[k].start_eval()

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

