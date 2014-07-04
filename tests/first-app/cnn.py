import math

import owl


class ConvInfo(object):

    def __init__(self, num_filters, filter_size, stride, padding_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding_size = padding_size


def init_network(layers, weights):
    global convinfo, layer0, layer1, layer2, w1, w2, b1, b2
    # init convolution weight
    row1 = convinfo.num_filters
    col1 = convinfo.filter_size.prod()
    var1 = math.sqrt(4.0 / (row1 + col1))
    w1dim = (convinfo.num_filters, ) + convinfo.filter_size
    w1 = owl.random.randn(*w1dim) * var1
    # init fully weight
    row2 = layer2.prod()
    col2 = layer1.prod()
    var2 = math.sqrt(4.0 / (row2 + col2))
    w2 = owl.random.randn(row2, col2) * var2

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

def train_network(num_epochs=100, num_train_samples=60000, minibatch_size=256,
                  num_minibatches=235, eps_w=0.001, eps_b=0.001):
    global convinfo, layer0, layer1, layer2, w1, w2, b1, b2
    loader = DBLoader('./data/mnist')
    for i in xrange(num_epochs):
        for j in xrange(num_minibatches):
            loader.load_next(minibatch_size)
            data = loader.get_data()
            realmbsize = data.shape[1]
            b1_tile_times = (layer1[0], layer1[1], 1, realmbsize)
            b2_tile_times = (1, realmbsize)
            l0_data_size = layer0 + (realmbsize, )
            l1_data_size_conv = layer1 + (realmbsize, )
            l1_data_size_fully = (layer1.prod(), realmbsize)
            l2_data_size = layer2 + (realmbsize, )

            a0 = data.reshape(l0_data_size)

            # FF-conv
            y1 = owl.convff(w1, a0, convinfo) + owl.tile(b1, b1_tile_times)
            a1 = owl.sigmoid(y1)
            # FF-fully
            y2 = w2 * a1.reshape(l1_data_size_fully) + owl.tile(b2, b2_tile_times)
            a2 = owl.sigmoid(y2)
            # Error
            target = loader.get_label()
            predict = softmax(a2)
            print_training_accuracy(predict, target)
            s2 = target - predict
            # BP-fully
            s1 = w2.T * s2
            s1 = owl.multiply(s1, 1 - s1)
            # No BP-conv
            # Update bias
            b2 -= eps_b * owl.sum(s2, 1) / realmbsize
            b1 -= eps_b * owl.sum(s1.reshape(l1_data_size_conv), (0, 1, 3)) / realmbsize
            # Update weight
            w2 -= eps_w * s2 * a1.reshape(l1_data_size_fully).T / realmbsize
            w1 -= eps_w * owl.get_grad(s1.reshape(l1_data_size_conv), a0, convinfo) / realmbsize


if __name__ == '__main__':
    layer0 = (28, 28, 1)
    layer1 = (28, 28, 16)
    layer2 = (10, )
    convinfo = ConvInfo(16, (5, 5, 1), (1, 1, 1), (2, 2, 0))
    w1 = None
    w2 = None
    b1 = owl.zeros((1, 1, 16, 1))
    b2 = owl.zeros((10, 1))
    init_network()
    train_network()
