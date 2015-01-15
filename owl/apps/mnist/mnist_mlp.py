import sys,os
import math
import owl
import owl.elewise as ele
import owl.conv as co
import numpy as np
import mnist_io

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
        (train_data, test_data) = mnist_io.load_mb_from_mat(self.data_file, self.mb_size)
        np.set_printoptions(linewidth=200)
        num_test_samples = test_data[0].shape[0]
        (test_samples, test_labels) = map(lambda npdata : owl.from_numpy(npdata), test_data)
        count = 1
        for epoch in range(self.num_epochs):
            print '---Start epoch #%d' % epoch
            # train
            for (mb_samples, mb_labels) in train_data:
                num_samples = mb_samples.shape[0]

                owl.set_device(self.cpu)
                a1 = owl.from_numpy(mb_samples)
                target = owl.from_numpy(mb_labels)
                owl.set_device(self.gpu)

                # ff
                a2 = ele.sigm(self.w1 * a1 + self.b1)
                a3 = ele.sigm(self.w2 * a2 + self.b2)
                # softmax & error
                out = co.softmax(a3)
                s3 = out - target
                # bp
                s3 = ele.mult(s3, 1 - s3)
                s2 = self.w2.trans() * s3
                s2 = ele.mult(s2, 1 - s2)
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
                    correct = out.argmax(0) - target.argmax(0)
                    val = correct.to_numpy()
                    print 'Training error:', float(np.count_nonzero(val)) / num_samples
                    # test
                    a1 = test_samples
                    a2 = ele.sigm(self.w1 * a1 + self.b1)
                    a3 = ele.sigm(self.w2 * a2 + self.b2)
                    correct = a3.argmax(0) - test_labels.argmax(0)
                    val = correct.to_numpy()
                    #print val
                    print 'Testing error:', float(np.count_nonzero(val)) / num_test_samples
                count = count + 1

            print '---Finish epoch #%d' % epoch

if __name__ == '__main__':
    owl.initialize(sys.argv)
    trainer = MnistTrainer(num_epochs = 10)
    trainer.run()
