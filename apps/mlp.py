import math

import owl


class Layer(object):

    def __init__(length):
        self.length = length
        self.bias = owl.zeros((length, 1))


def init_network(layers, weights):
    num_layers = len(layers)
    for i in xrange(num_layers - 1):
        row = layers[i + 1].length
        col = layers[i].length
        var = math.sqrt(4.0 / (row + col))  # XXX: variance or stddev?
        weights.append(owl.random.randn(row, col) * var)

def softmax(m):
    maxval = m.max(0)
    centered = m - owl.tile(maxval, (m.shape[0], 1))
    class_normalizer = owl.log(owl.exp(centered).sum(0)) + maxval
    return owl.exp(m - owl.tile(class_normalizer, (m.shape[0], 1)))

def print_training_accuracy(o, t):
    predict = o.argmax(0)
    ground_truth = t.argmax(0)
    correct = len((predict - ground_truth).nonzero())
    print 'Training error: {}'.format((minibatch_size - correct) /
                                      minibatch_size)

def train_network(layers, weights,
                  num_epochs=100, num_train_samples=60000, minibatch_size=256,
                  num_minibatches=235, eps_w=0.001, eps_b=0.001):
    num_layers = len(layers)
    loader = DBLoader('./data/mnist')
    for i in xrange(num_epochs):
        for j in xrange(num_minibatches):
            loader.load_next(minibatch_size)
            acts = [None] * num_layers
            sens = [None] * num_layers
            acts[0] = loader.get_data()
            # FF
            for k in xrange(1, num_layers):
                act[k] = (weights[k] * act[k - 1] +
                          owl.tile(layers[k].bias, (1, num_minibatches)))
                act[k] = owl.sigmoid(acts[k])
            # Error
            acts[-1] = softmax(acts[-1])
            target = loader.get_labels()
            print_training_accuracy(acts[-1], target)
            sens[-1] = target - acts[-1]
            # BP
            for k in reversed(xrange(num_layers - 1)):
                sens[k] = weights[k].T * sens[k + 1]
                # This is element-wise.
                sens[k] = owl.multiply(sens[k], 1 - sens[k])
            # Update bias
            for k in xrange(1, num_layers):
                layers[k].bias -= eps_b * sens[k].sum(1) / minibatch_size
            # Update weight
            for k in xrange(num_layers - 1):
                weights[k] -= eps_w * sens[k + 1] * acts[k].T / minibatch_size

if __name__ == '__main__':
    layers = [Layer(28 * 28), Layer(256), Layer(10)]
    weights = []
    init_network(layers, weights)
    train_network(layers, weights)
