import numpy as np
import owl
import owl.conv as co
import owl.elewise as ele

class AlexModel:
    def __init__(self):
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
        self.num_weights = 8
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

    def train_one_mb(self, data, label, dropout_rate):
        num_samples = data.shape[-1]
        num_layers = 12
        acts = [None] * num_layers
        sens = [None] * num_layers
        weightsgrad = [None] * self.num_weights
        biasgrad = [None] * self.num_weights

        # FF
        acts[0] = data
        acts[1] = ele.relu(self.convs[0].ff(acts[0], self.weights[0], self.bias[0])) # conv1
        acts[2] = self.poolings[0].ff(acts[1]) # pool1
        acts[3] = ele.relu(self.convs[1].ff(acts[2], self.weights[1], self.bias[1])) # conv2
        acts[4] = self.poolings[1].ff(acts[3]) # pool2
        acts[5] = ele.relu(self.convs[2].ff(acts[4], self.weights[2], self.bias[2])) # conv3
        acts[6] = ele.relu(self.convs[3].ff(acts[5], self.weights[3], self.bias[3])) # conv4
        acts[7] = ele.relu(self.convs[4].ff(acts[6], self.weights[4], self.bias[4])) # conv5
        acts[8] = self.poolings[2].ff(acts[7]) # pool5
        re_acts8 = acts[8].reshape([np.prod(acts[8].shape[0:3]), num_samples])
        acts[9] = ele.relu(self.weights[5] * re_acts8 + self.bias[5]) # fc6
        mask6 = owl.randb(acts[9].shape, dropout_rate)
        acts[9] = ele.mult(acts[9], mask6) # drop6
        acts[10] = ele.relu(self.weights[6] * acts[9] + self.bias[6]) # fc7
        mask7 = owl.randb(acts[10].shape, dropout_rate)
        acts[10] = ele.mult(acts[10], mask7) # drop7
        acts[11] = self.weights[7] * acts[10] + self.bias[7] # fc8

        out = co.softmax(acts[11], co.soft_op.instance) # prob

        sens[11] = out - label
        sens[10] = self.weights[7].trans() * sens[11] # fc8
        sens[10] = ele.mult(sens[10], mask7) # drop7
        sens[10] = ele.relu_back(sens[10], acts[10]) # relu7
        sens[9] = self.weights[6].trans() * sens[10]
        sens[9] = ele.mult(sens[9], mask6) # drop6
        sens[9] = ele.relu_back(sens[9], acts[9]) # relu6
        sens[8] = (self.weights[5].trans() * sens[9]).reshape(acts[8].shape) # fc6
        sens[7] = ele.relu_back(self.poolings[2].bp(sens[8], acts[8], acts[7]), acts[7]) # pool5, relu5
        sens[6] = ele.relu_back(self.convs[4].bp(sens[7], self.weights[4]), acts[6]) # conv5, relu4
        sens[5] = ele.relu_back(self.convs[3].bp(sens[6], self.weights[3]), acts[5]) # conv4, relu3
        sens[4] = self.convs[2].bp(sens[5], self.weights[2]) # conv3
        sens[3] = ele.relu_back(self.poolings[1].bp(sens[4], acts[4], acts[3]), acts[3]) # pool2, relu2
        sens[2] = self.convs[1].bp(sens[3], self.weights[1]) # conv2
        sens[1] = self.poolings[0].bp(sens[2], acts[2], acts[1]) # pool1
        sens[1] = ele.relu_back(sens[1], acts[1]) # relu1

        weightsgrad[7] = sens[11] * acts[10].trans()
        weightsgrad[6] = sens[10] * acts[9].trans()
        weightsgrad[5] = sens[9] * re_acts8.trans()
        weightsgrad[4] = self.convs[4].weight_grad(sens[7], acts[6])
        weightsgrad[3] = self.convs[3].weight_grad(sens[6], acts[5])
        weightsgrad[2] = self.convs[2].weight_grad(sens[5], acts[4])
        weightsgrad[1] = self.convs[1].weight_grad(sens[3], acts[2])
        weightsgrad[0] = self.convs[0].weight_grad(sens[1], acts[0])
        biasgrad[7] = sens[11].sum(1)
        biasgrad[6] = sens[10].sum(1)
        biasgrad[5] = sens[9].sum(1)
        biasgrad[4] = self.convs[4].bias_grad(sens[7])
        biasgrad[3] = self.convs[3].bias_grad(sens[6])
        biasgrad[2] = self.convs[2].bias_grad(sens[5])
        biasgrad[1] = self.convs[1].bias_grad(sens[3])
        biasgrad[0] = self.convs[0].bias_grad(sens[1])
        return (out, weightsgrad, biasgrad)

    def update(self, weightsgrad, biasgrad, num_samples, mom, lr, wd):
        for k in range(self.num_weights):
            self.weightsdelta[k] = mom * self.weightsdelta[k] - lr / num_samples  * weightsgrad[k] - lr * wd * self.weights[k]
            self.biasdelta[k] = mom * self.biasdelta[k] - lr / num_samples  * biasgrad[k]
            self.weights[k] += self.weightsdelta[k]
            self.bias[k] += self.biasdelta[k]
