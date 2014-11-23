import sys,os
import math
import owl
import owl.elewise
import scipy.io as si
import numpy as np

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
        (test_samples, test_labels) = map(lambda npdata : owl.from_nparray(npdata), test_data)
        count = 1
        for epoch in range(self.num_epochs):
            print '---Start epoch #%d' % epoch
            # train
            for (mb_samples, mb_labels) in train_data:
                num_samples = mb_samples.shape[0]

                owl.set_device(self.cpu)
                a1 = owl.from_nparray(mb_samples)
                target = owl.from_nparray(mb_labels)
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

if __name__ == '__main__':
    owl.initialize(sys.argv)
    trainer = MnistTrainer(num_epochs = 10)
    trainer.run()
