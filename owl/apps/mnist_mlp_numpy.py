import sys,os
import math
import scipy.io as si
import numpy as np
import numpy.random

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
    #print train_data[0]
    #print train_data[0][0].dtype, train_data[0][1].dtype
    #print test_data
    print 'Training data: %d mini-batches' % len(train_mb)
    print 'Test data: %d samples' % test_all.shape[0]
    return (train_data, test_data)

def sigmoid(m):
    return 1 / (1 + np.exp(-m))

def softmax(m):
    maxval = np.max(m, axis=0)
    centered = m - maxval
    class_normalizer = np.log(np.max(np.exp(centered), axis=0)) + maxval
    return np.exp(m - class_normalizer)

class MnistTrainer:
    def __init__(self, data_file='mnist_all.mat', num_epochs=100, mb_size=256, eps_w=0.01, eps_b=0.01):
        self.data_file = data_file
        self.num_epochs=num_epochs
        self.mb_size=mb_size
        self.eps_w=eps_w
        self.eps_b=eps_b
        # init weight
        l1 = 784; l2 = 256; l3 = 10
        self.l1 = l1; self.l2 = l2; self.l3 = l3
        self.w1 = np.random.randn(l2, l1) * math.sqrt(4.0 / (l1 + l2))
        self.w2 = np.random.randn(l3, l2) * math.sqrt(4.0 / (l2 + l3))
        self.b1 = np.zeros([l2, 1])
        self.b2 = np.zeros([l3, 1])

    def run(self):
        (train_data, test_data) = load_mb_from_mat(self.data_file, self.mb_size)
        np.set_printoptions(linewidth=200)
        num_test_samples = test_data[0].shape[0]
        (test_samples, test_labels) = test_data
        count = 1
        for epoch in range(self.num_epochs):
            print '---Start epoch #%d' % epoch
            # train
            for (mb_samples, mb_labels) in train_data:
                num_samples = mb_samples.shape[0]

                a1 = mb_samples.T
                target = mb_labels.T

                # ff
                a2 = sigmoid(np.dot(self.w1, a1) + self.b1)
                a3 = sigmoid(np.dot(self.w2, a2) + self.b2)
                # softmax & error
                out = softmax(a3)
                s3 = out - target
                # bp
                s3 = s3 * (1 - s3)
                s2 = np.dot(self.w2.T, s3)
                s2 = s2 * (1 - s2)
                # grad
                gw1 = np.dot(s2, a1.T) / num_samples
                gb1 = np.sum(s2, axis=1, keepdims=True) / num_samples
                gw2 = np.dot(s3, a2.T) / num_samples
                gb2 = np.sum(s3, axis=1, keepdims=True) / num_samples
                # update
                self.w1 -= self.eps_w * gw1
                self.w2 -= self.eps_w * gw2
                self.b1 -= self.eps_b * gb1
                self.b2 -= self.eps_b * gb2

                if (count % 40 == 0):
                    correct = np.argmax(out, axis=0) - np.argmax(target, axis=0)
                    print 'Training error:', float(np.count_nonzero(correct)) / num_samples
                    # test
                    a1 = test_samples.T
                    a2 = sigmoid(np.dot(self.w1, a1) + self.b1)
                    a3 = sigmoid(np.dot(self.w2, a2) + self.b2)
                    correct = np.argmax(a3, axis=0) - np.argmax(test_labels.T, axis=0)
                    #print correct
                    print 'Testing error:', float(np.count_nonzero(correct)) / num_test_samples
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

if __name__ == '__main__':
    trainer = MnistTrainer(num_epochs = 10)
    trainer.run()
