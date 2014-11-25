import sys,os
import lmdb
import numpy as np
import numpy.random
from caffe_pb2 import Datum
from caffe_pb2 import BlobProto

class ImageNetDataProvider:
    def __init__(self, mean_file, train_db, val_db, test_db):
        bp = BlobProto()
        with open(mean_file, 'rb') as f:
            bp.ParseFromString(f.read())
        #print 'num=', bp.num, 'channels=', bp.channels, 'height=', bp.height, 'width=', bp.width
        #np.set_printoptions(linewidth=200)
        #print mean_data[:,:,0]
        #print mean_data[:,:,1]
        #print mean_data[:,:,2]
        #print 'diff=', bp.diff
        self.mean_data = np.array(bp.data, dtype=np.float32).reshape([256, 256, 3])
        self.train_db = train_db
        self.val_db = val_db
        self.test_db = test_db

    def get_train_mb(self, mb_size, cropped_size=224):
        env = lmdb.open(self.train_db, readonly=True)
        # print env.stat()
        d = Datum()
        samples = []
        labels = []
        with env.begin(write=False, buffers=False) as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                d.ParseFromString(value)
                #print '#channels=', d.channels, 'height=', d.height, 'width=', d.width, 'label=', d.label
                im = np.fromstring(d.data, dtype=np.uint8).reshape([256, 256, 3]) - self.mean_data
                # random crop
                [crop_h, crop_w] = np.random.randint(256 - cropped_size, size=2)
                im_cropped = im[crop_h:crop_h+cropped_size, crop_w:crop_w+cropped_size, :]
                # make labels
                im_label = np.zeros(1000, dtype=np.float32)
                im_label[d.label] = 1
                samples.append(im_cropped.flatten().tolist())
                labels.append(im_label.tolist())
                if len(samples) == mb_size:
                    yield (samples, labels)
                    samples = []
                    labels = []
        if samples != None:
            yield (samples, labels)

    def get_test_mb(self):
        return None


if __name__ == '__main__':
    dp = ImageNetDataProvider(mean_file='/home/minjie/data/imagenet/imagenet_mean.binaryproto',
            train_db='/home/minjie/data/imagenet/ilsvrc12_train_lmdb',
            val_db='/home/minjie/data/imagenet/ilsvrc12_val_lmdb',
            test_db='/home/minjie/data/imagenet/ilsvrc12_test_lmdb')
    for (samples, labels) in dp.get_train_mb(256):
        #print samples.shape
        #print labels.shape
        break
