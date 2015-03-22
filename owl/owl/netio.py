import sys,os,gc
import lmdb
import numpy as np
import numpy.random
import scipy.io as si
from caffe import *
from PIL import Image
from google.protobuf import text_format

class ImageListDataProvider:
    def __init__(self, image_data_param, transform_param, mm_batch_num):
        bp = BlobProto()
        if len(transform_param.mean_file) == 0:
            self.mean_data = np.ones([3, image_data_param.new_height, image_data_param.new_width], dtype=np.float32)
            assert(len(transform_param.mean_value) == 3)
            self.mean_data[0] = transform_param.mean_value[0]
            self.mean_data[1] = transform_param.mean_value[1]
            self.mean_data[2] = transform_param.mean_value[2]           
        else:    
            with open(transform_param.mean_file, 'rb') as f:
                bp.ParseFromString(f.read())
            self.mean_data = np.array(bp.data, dtype=np.float32).reshape([3, image_data_param.new_height, image_data_param.new_width])
        self.source = image_data_param.source
        self.new_height = image_data_param.new_height
        self.new_width = image_data_param.new_width
        self.batch_size = image_data_param.batch_size / mm_batch_num
        self.crop_size = transform_param.crop_size
        self.mirror = transform_param.mirror


    def get_mb(self, phase = 'TRAIN'):
        sourcefile = open(self.source, 'r')
        samples = np.zeros([self.batch_size, self.crop_size ** 2 * 3], dtype = np.float32)
        num_label = -1

        count = 0
        line = sourcefile.readline()
        while line:
            line_info = line.split(' ')
            assert(len(line_info) >= 2)
            if num_label == -1:
                num_label = len(line_info) - 1
                labels = np.zeros([self.batch_size, num_label], dtype = np.float32)
            #labels
            for labelidx in range(num_label):
                labels[count][labelidx] = float(line_info[labelidx+1])
            
            #read img
            try:
                img = Image.open(line_info[0])
            except IOError, e:
                print e
                print "not an image file %s" % (line_info[0])
            #convert to rgb
            if img.mode not in ('RGB'):
                img = img.convert('RGB')
            #resize
            try:
                img = img.resize((self.new_height, self.new_width), Image.ANTIALIAS)
            except IOError, e:
                print e
                print "resize error occur %s" % (line_info[0])
            
            orinpimg = np.array(img, dtype = np.uint8)
            npimg = [orinpimg[:,:,2], orinpimg[:,:,1], orinpimg[:,:,0]]
            #change to pixels
            pixels = npimg - self.mean_data

            #crop 
            if phase == 'TRAIN':
                crop_h = np.random.randint(np.shape(pixels)[1] - self.crop_size)
                crop_w = np.random.randint(np.shape(pixels)[2]- self.crop_size)
            else:
                crop_h = (np.shape(pixels)[1]- self.crop_size) / 2
                crop_w = (np.shape(pixels)[2] - self.crop_size) / 2
            
            im_cropped = pixels[:, crop_h:crop_h+self.crop_size, crop_w:crop_w+self.crop_size]
            if self.mirror == True and numpy.random.rand() > 0.5:
                im_cropped = im_cropped[:,:,::-1]
            samples[count, :] = im_cropped.reshape(self.crop_size ** 2 * 3).astype(np.float32)
            
            count = count + 1
            if count == self.batch_size:
                yield (samples, labels)
                labels = np.zeros([self.batch_size, num_label], dtype=np.float32)
                count = 0
            line = sourcefile.readline()


        if count != self.batch_size:
            delete_idx = np.arange(count, self.batch_size)
            fclose(sourcefile)
            yield (np.delete(samples, delete_idx, 0), np.delete(labels, delete_idx, 0))


class LMDBDataProvider:
    def __init__(self, data_param, transform_param, mm_batch_num):
        bp = BlobProto()
        if len(transform_param.mean_file) == 0:
            self.mean_data = np.ones([3, 256, 256], dtype=np.float32)
            assert(len(transform_param.mean_value) == 3)
            self.mean_data[0] = transform_param.mean_value[0]
            self.mean_data[1] = transform_param.mean_value[1]
            self.mean_data[2] = transform_param.mean_value[2]           
        else:    
            with open(transform_param.mean_file, 'rb') as f:
                bp.ParseFromString(f.read())
            mean_narray = np.array(bp.data, dtype=np.float32)
            h_w = np.sqrt(np.shape(mean_narray)[0] / 3)
            self.mean_data = np.array(bp.data, dtype=np.float32).reshape([3, h_w, h_w])
        self.source = data_param.source
        self.batch_size = data_param.batch_size / mm_batch_num
        self.crop_size = transform_param.crop_size
        self.mirror = transform_param.mirror

    def get_mb(self, phase = 'TRAIN'):
        env = lmdb.open(self.source, readonly=True)
        samples = np.zeros([self.batch_size, self.crop_size ** 2 * 3], dtype=np.float32)
        num_label = -1
        count = 0
        with env.begin(write=False, buffers=False) as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                d = Datum()
                d.ParseFromString(value)
                ori_size = np.sqrt(len(d.data) / 3)
                im = np.fromstring(d.data, dtype=np.uint8).reshape([3, ori_size, ori_size]) - self.mean_data
                if phase == 'TRAIN':
                    [crop_h, crop_w] = np.random.randint(ori_size - self.crop_size, size=2)
                else:
                    crop_h = (ori_size - self.crop_size) / 2
                    crop_w = (ori_size - self.crop_size) / 2
                
                im_cropped = im[:, crop_h:crop_h+self.crop_size, crop_w:crop_w+self.crop_size]
                if self.mirror == True and numpy.random.rand() > 0.5:
                    im_cropped = im_cropped[:,:,::-1]
                
                samples[count, :] = im_cropped.reshape(self.crop_size ** 2 * 3).astype(np.float32)
                
                if num_label == -1:
                    num_label = len(d.label)
                    labels = np.zeros([self.batch_size, num_label], dtype=np.float32)
                labels[count, :] = d.label
                
                count = count + 1
                if count == self.batch_size:
                    yield (samples, labels)
                    labels = np.zeros([self.batch_size, num_label], dtype=np.float32)
                    count = 0
        if count != self.batch_size:
            delete_idx = np.arange(count, self.batch_size)
            yield (np.delete(samples, delete_idx, 0), np.delete(labels, delete_idx, 0))

if __name__ == '__main__':
    if sys.argv[1] == 'lmdb':
        #Test 
        net_file = '/home/tianjun/configfile/Googmodel/train_val_CUB_lmdb.prototxt'
        with open(net_file, 'r') as f:
            netconfig = NetParameter()
            text_format.Merge(str(f.read()), netconfig)
        layerinfo = netconfig.layers[0]
        dp = LMDBDataProvider(layerinfo.data_param, layerinfo.transform_param, 1)
        count = 0
        for (samples, labels) in dp.get_mb():
            print count, ':', samples.shape
            count = count + 1
    else:
        #Test 
        net_file = '/home/tianjun/configfile/Googmodel/train_val_CUB_list.prototxt'
        with open(net_file, 'r') as f:
            netconfig = NetParameter()
            text_format.Merge(str(f.read()), netconfig)
        layerinfo = netconfig.layers[0]
        dp = ImageListDataProvider(layerinfo.image_data_param, layerinfo.transform_param, 1)
        count = 0
        for (samples, labels) in dp.get_mb():
            print count, ':', samples.shape
            count = count + 1


