#!/usr/bin/env python
import sys,owl
from owl import SimpleFileLoader
from owl import FileFormat
from owl import MBLoader
import owl.elewise as ele

weight_init_files = ['w12_init.dat', 'w23_init.dat']
weight_out_files = ['w12_trained.dat', 'w23_trained.dat']
bias_out_files = ['b2_trained.dat', 'b3_trained.dat']
train_data_file = "data/mnist/traindata.dat"
train_label_file = "data/mnist/trainlabel.dat"
test_data_file = "data/mnist/testdata.dat"
test_label_file = "data/mnist/testlabel.dat"
l1 = 784
l2 = 256
l3 = 10
epsW = 0.001
epsB = 0.001
numepochs = 1
mbsize = 256
num_mb = 235

def print_accuracy(o, t):
    predict = o.argmax(0)
    groundtruth = t.argmax(0)
    correct = (predict - groundtruth).count_zero()
    print 'Training error:', (float(mbsize) - correct) / mbsize

def main(args=sys.argv[1:]):
    print sys.argv
    owl.initialize(sys.argv)
    # init weights
    loader = SimpleFileLoader()
    w12 = owl.load_from_file([l2, l1], weight_init_files[0], loader)
    w23 = owl.load_from_file([l3, l2], weight_init_files[1], loader)
    b2 = owl.zeros([l2, 1])
    b3 = owl.zeros([l3, 1])
    # training
    print 'Training procedures'
    train_data_loader = MBLoader(train_data_file, [l1])
    train_label_loader = MBLoader(train_label_file, [l3])
    for epoch in xrange(numepochs):
        print 'epoch', epoch
        for mb in xrange(num_mb):
            data = train_data_loader.load_next(mbsize)
            label = train_label_loader.load_next(mbsize)
            # ff
            a1 = data
            a2 = ele.sigmoid((w12 * a1).normalize(b2, owl.op.add)) #TODO
            a3 = ele.sigmoid((w23 * a2).normalize(b3, owl.op.add)) #TODO
            # softmax
            a3 = owl.softmax(a3)
            # bp
            s3 = a3 - label
            s2 = w23.trans() * s3
            s2 = ele.mult(s2, 1 - s2)
            # update
            w12 = w12 - epsW * s2 * a1.trans() / mbsize
            b2 = b2 - epsB * s2.sum(1) / mbsize
            w23 = w23 - epsW * s3 * a2.trans() / mbsize
            b3 = b3 - epsB * s3.sum(1) / mbsize
            
            if mb % 20 == 0:
                print_accuracy(a3, label)

    format = FileFormat()
    format.binary = True
    w12.to_file(weight_out_files[0], format)
    w23.to_file(weight_out_files[1], format)
    print 'Training finished.'

if __name__ == '__main__':
    sys.exit(main())
