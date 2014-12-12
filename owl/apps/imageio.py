import sys,os,gc
import lmdb
import numpy as np
import numpy.random
import scipy.io as si
from caffe_pb2 import Datum
from caffe_pb2 import BlobProto
from PIL import Image

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
        self.mean_data = np.array(bp.data, dtype=np.float32).reshape([3, 256, 256])
        
        self.train_db = train_db
        self.val_db = val_db
        self.test_db = test_db

    def get_train_mb(self, mb_size, cropped_size=227):
        env = lmdb.open(self.train_db, readonly=True)
        # print env.stat()
        samples = np.zeros([mb_size, cropped_size ** 2 * 3], dtype=np.float32)
        labels = np.zeros([mb_size, 1000], dtype=np.float32)
        count = 0
        with env.begin(write=False, buffers=False) as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                d = Datum()
                d.ParseFromString(value)
                #print '#channels=', d.channels, 'height=', d.height, 'width=', d.width, 'label=', d.label
                im = np.fromstring(d.data, dtype=np.uint8).reshape([3, 256, 256]) - self.mean_data
                
                [crop_h, crop_w] = np.random.randint(256 - cropped_size, size=2)
                
                im_cropped = im[:, crop_h:crop_h+cropped_size, crop_w:crop_w+cropped_size]

                '''
                iim = np.transpose(im_cropped.reshape(cropped_size*cropped_size*3).reshape([3, cropped_size*cropped_size])).reshape([cropped_size, cropped_size, 3])
                img = Image.fromarray(iim)
                img.save('cropimg.jpg', format='JPEG')
                exit(0)
                '''
                
                samples[count, :] = im_cropped.reshape(cropped_size ** 2 * 3).astype(np.float32)
                labels[count, d.label] = 1
                count = count + 1
                if count == mb_size:
                    yield (samples, labels)
                    #samples = np.zeros([mb_size, cropped_size ** 2 * 3])
                    labels = np.zeros([mb_size, 1000], dtype=np.float32)
                    count = 0
        if count != mb_size:
            delete_idx = np.arange(count, mb_size)
            yield (np.delete(samples, delete_idx, 0), np.delete(labels, delete_idx, 0))

    def get_test_mb(self):
        return None

def _extract(prefix, md, max_dig):
    for dig in range(max_dig):
        samples = md[prefix + str(dig)]
        labels = np.empty([samples.shape[0], 1], dtype=np.float32)
        labels.fill(dig * 256)
        yield np.hstack((samples, labels)) / 256

def _split_sample_and_label(merged_mb):
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
    train_all = np.concatenate(tuple(_extract('train', md, 10)))
    test_all = np.concatenate(tuple(_extract('test', md, 10)))
    # shuffle
    np.random.shuffle(train_all)
    # make minibatch
    train_mb = np.vsplit(train_all, range(mb_size, train_all.shape[0], mb_size))
    train_data = map(_split_sample_and_label, train_mb)
    test_data = _split_sample_and_label(test_all)
    print 'Training data: %d mini-batches' % len(train_mb)
    print 'Test data: %d samples' % test_all.shape[0]
    return (train_data, test_data)

if __name__ == '__main__':
    dp = ImageNetDataProvider(mean_file='/home/minjie/data/imagenet/imagenet_mean.binaryproto',
            train_db='/home/minjie/data/imagenet/ilsvrc12_train_lmdb',
            val_db='/home/minjie/data/imagenet/ilsvrc12_val_lmdb',
            test_db='/home/minjie/data/imagenet/ilsvrc12_test_lmdb')
    count = 0
    for (samples, labels) in dp.get_train_mb(256):
        print count, ':', samples.shape
        #print labels.shape
        #print samples[0,0:10]
        #print np.argmax(labels, axis=1)
        # training
        count = count + 1
        if count % 10 == 0:
            break
