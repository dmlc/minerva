#!/usr/bin/env python
import sys,owl
from owl import SimpleFileLoader
from owl import FileFormat
import owl.elewise as ele
import struct
import array

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

class MBLoader:
    def __init__(self, fname):
        self.fname = fname
        self.fin = open(fname, 'rb')
        self.num_samples = struct.unpack('i', self.fin.read(4))[0]
        self.sample_len = struct.unpack('i', self.fin.read(4))[0]
        self.sample_pos = 0

    def load_next(self, num_samples_to_read):
        data = []
        nread_remain = num_samples_to_read
        while nread_remain > 0:
            nread = min(nread_remain, self.num_samples - self.sample_pos)
            buf = array.array('f')
            buf.fromfile(self.fin, nread * self.sample_len)
            data += buf.tolist()
            nread_remain -= nread
            self.sample_pos = (self.sample_pos + nread) % self.num_samples
            if self.sample_pos == 0:
                self.fin.seek(8, 0)
        return data

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
    train_data_loader = MBLoader(train_data_file)
    train_label_loader = MBLoader(train_label_file)
    for epoch in xrange(numepochs):
        print 'epoch', epoch
        for mb in xrange(num_mb):
            data = owl.make_narray([l1, mbsize], train_data_loader.load_next(mbsize))
            label = owl.make_narray([l3, mbsize], train_label_loader.load_next(mbsize))
            # ff
            a1 = data
            a2 = ele.sigmoid((w12 * a1).normalize(b2, owl.op.add))
            a3 = ele.sigmoid((w23 * a2).normalize(b3, owl.op.add))
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
                correct = a3.argmax(0) - label.argmax(0)
                owl.wait_eval()
                if mb != 0:
                    val = last_correct.tolist()
                    print 'Training error:', (float(mbsize) - val.count(0.0)) / mbsize
                last_correct = correct
                correct.eval_async()

    owl.wait_eval()
    format = FileFormat()
    format.binary = True
    w12.tofile(weight_out_files[0], format)
    w23.tofile(weight_out_files[1], format)
    print 'Training finished.'

if __name__ == '__main__':
    sys.exit(main())
