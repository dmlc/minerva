import sys,os,gc
import lmdb
import numpy as np
import numpy.random
import scipy.io as si
from owl.caffe.caffe_pb2 import Datum
from owl.caffe.caffe_pb2 import BlobProto
from PIL import Image

class ImageNetDataProvider:
    def __init__(self, mean_file, train_db, mb_size, cropped_size):
        bp = BlobProto()
        with open(mean_file, 'rb') as f:
            bp.ParseFromString(f.read())
        self.mean_data = np.array(bp.data, dtype=np.float32).reshape([3, 256, 256])
        self.train_db = train_db
        self.mb_size = mb_size
        self.cropped_size = cropped_size

    def get_train_mb(self):
        mb_size = self.mb_size
        cropped_size = self.cropped_size
        env = lmdb.open(self.train_db, readonly=True)
        samples = np.zeros([mb_size, cropped_size ** 2 * 3], dtype=np.float32)
        labels = np.zeros([mb_size, 1000], dtype=np.float32)
        count = 0
        with env.begin(write=False, buffers=False) as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                d = Datum()
                d.ParseFromString(value)
                im = np.fromstring(d.data, dtype=np.uint8).reshape([3, 256, 256]) - self.mean_data
                [crop_h, crop_w] = np.random.randint(256 - cropped_size, size=2)
                im_cropped = im[:, crop_h:crop_h+cropped_size, crop_w:crop_w+cropped_size]
                samples[count, :] = im_cropped.reshape(cropped_size ** 2 * 3).astype(np.float32)
                labels[count, d.label] = 1
                count = count + 1
                if count == mb_size:
                    yield (samples, labels)
                    labels = np.zeros([mb_size, 1000], dtype=np.float32)
                    count = 0
        if count != mb_size:
            delete_idx = np.arange(count, mb_size)
            yield (np.delete(samples, delete_idx, 0), np.delete(labels, delete_idx, 0))

    def get_test_mb(self):
        # TODO
        return None

if __name__ == '__main__':
    dp = ImageNetDataProvider('/home/minjie/data/imagenet/imagenet_mean.binaryproto', '/home/minjie/data/imagenet/ilsvrc12_train_lmdb', 10, 224)
    count = 0
    for (samples, labels) in dp.get_train_mb():
        print count, ':', samples.shape
        count = count + 1
        if count % 10 == 0:
            break
