import sys,os,gc
import time
import lmdb
import numpy as np
import numpy.random
import scipy.io as si
from PIL import Image
from google.protobuf import text_format

from caffe import *

class ImageWindowDataProvider:
    ''' Class for Image Window Data Provider. This data provider will read the original image and crop out patches according to the given box position, then resize the patch to form batch. 

    .. note::
        layer type in configure file:
        WINDOW_DATA
        
        Data source file format for each image:
        # Img_ind \n
        Img_path \n
        C \n
        H \n 
        W \n
        Window_num \n
        label overlap_ratio upper left lower right \n
        label overlap_ratio upper left lower right \n
        ''......'' \n
        label overlap_ratio upper left lower right 
        
        - ``Img_ind``: image index
        - ``Img_path``: image path
        - ``C``: number of image channels (feature maps)
        - ``H``: image height
        - ``W``: image width
        - ``Window_num``: number of window patches
        - ``label``: label
        - ``overlap_ratio``: overlap ratio between the window and object bouding box
        - ``upper left lower right``: position of the window

    '''



    def __init__(self, window_data_param, mm_batch_num):
        bp = BlobProto()
        self.source = window_data_param.source
        self.batch_size = window_data_param.batch_size / mm_batch_num
        self.crop_size = window_data_param.crop_size
        self.mirror = window_data_param.mirror

        if len(window_data_param.mean_file) == 0:
            self.mean_data = np.ones([3, window_data_param.crop_size, window_data_param.crop_size], dtype=np.float32)
            assert(len(window_data_param.mean_value) == 3)
            self.mean_data[0] = window_data_param.mean_value[0]
            self.mean_data[1] = window_data_param.mean_value[1]
            self.mean_data[2] = window_data_param.mean_value[2]           
        else:    
            with open(window_data_param.mean_file, 'rb') as f:
                bp.ParseFromString(f.read())
            np_mean = np.array(bp.data, dtype=np.float32)
            mean_size = np.sqrt(np.shape(np_mean)[0] / 3)
            self.mean_data = np_mean.reshape([3, mean_size, mean_size])
            self.mean_data = self.mean_data[:, (mean_size-self.crop_size)/2:mean_size-(mean_size-self.crop_size)/2, (mean_size-self.crop_size)/2:mean_size-(mean_size-self.crop_size)/2] 


    def get_mb(self, phase = 'TRAIN'):
        sourcefile = open(self.source, 'r')
        samples = np.zeros([self.batch_size, self.crop_size ** 2 * 3], dtype = np.float32)
        labels = np.zeros([self.batch_size, 1], dtype=np.float32)
        num_label = -1

        count = 0
        line = sourcefile.readline()
        while line:
            #process each image 
            assert(line[0] == '#')
            path = sourcefile.readline()
            path = path[0:-1]
            channel = int(sourcefile.readline())
            height = int(sourcefile.readline())
            width = int(sourcefile.readline())
            boxnum = int(sourcefile.readline())
            
            #open image
            try:
                img = Image.open(path)
            except IOError, e:
                print e
                print "not an image file %s" % (path[0])
            #convert to rgb
            if img.mode not in ('RGB'):
                img = img.convert('RGB')
            #read boxes 
            for i in range(boxnum):
                line = sourcefile.readline()
                box_info = line.split(' ')
                
                x1 = int(box_info[2]) - 1
                y1 = int(box_info[3]) - 1
                x2 = int(box_info[4]) 
                y2 = int(box_info[5])

                #crop image
                patch = img.crop((y1, x1, y2, x2))
                
                #resize
                try:
                    patch = patch.resize((self.crop_size, self.crop_size), Image.ANTIALIAS)
                except IOError, e:
                    print e
                    print "resize error occur %s" % (line_info[0])

                orinpimg = np.array(patch, dtype = np.uint8)
                npimg = np.transpose(orinpimg.reshape([self.crop_size * self.crop_size, 3])).reshape(np.shape(self.mean_data))
                npimg = npimg[::-1,:,:]
                
                '''
                #output
                imgdata = np.zeros([self.crop_size, self.crop_size, 3], dtype=np.uint8)
                imgdata[:,:,0] = npimg[2,:,:]
                imgdata[:,:,1] = npimg[1,:,:]
                imgdata[:,:,2] = npimg[0,:,:]
                cropimg = Image.fromarray(imgdata)
                nnn = '/home/tianjun/tests/img_%d.jpg' % (i)
                cropimg.save(nnn, format = 'JPEG')
                '''

                pixels = npimg - self.mean_data
                if self.mirror == True and numpy.random.rand() > 0.5:
                    pixels = pixels[:,:,::-1]
                samples[count, :] = pixels.reshape(self.crop_size ** 2 * 3).astype(np.float32)
                
                count = count + 1
                if count == self.batch_size:
                    yield (samples, labels)
                    labels = np.zeros([self.batch_size, 1], dtype=np.float32)
                    count = 0
            #finish one  
            if count  > 0:
                delete_idx = np.arange(count, self.batch_size)
                yield (np.delete(samples, delete_idx, 0), np.delete(labels, delete_idx, 0))
                count = 0
            line = sourcefile.readline()
        sourcefile.close()


class ImageListDataProvider:
    ''' Class for Image Data Provider. This data provider will read from original data into RGB value, then resize the patch to form batch. 

    .. note::
        layer type in configure file:
        IMAGE_DATA
        
        Data source file format for each image:
        Img_path label_0 label_1 ... label_n
        
        - ``Img_path``: image path
        - ``label_0 label_1 ... label_n``: we support multi-label for a single image

    '''
    
    
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
            np_mean = np.array(bp.data, dtype=np.float32)
            mean_size = np.sqrt(np.shape(np_mean)[0] / 3)
            self.mean_data = np_mean.reshape([3, mean_size, mean_size])
            self.mean_data = self.mean_data[:, (mean_size-image_data_param.new_height)/2:mean_size-(mean_size-image_data_param.new_height)/2, (mean_size-image_data_param.new_width)/2:mean_size-(mean_size-image_data_param.new_width)/2] 
        
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
            npimg = np.transpose(orinpimg.reshape([self.new_height * self.new_width, 3])).reshape(np.shape(self.mean_data))
            npimg = npimg[::-1,:,:]
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
            sourcefile.close()
            yield (np.delete(samples, delete_idx, 0), np.delete(labels, delete_idx, 0))


class LMDBDataProvider:
    ''' Class for LMDB Data Provider. 

    .. note::
        layer type in configure file:
        DATA

    '''

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
               
                '''
                #output
                imgdata = np.zeros([self.crop_size, self.crop_size, 3], dtype=np.uint8)
                imgdata[:,:,0] = im_cropped[2,:,:]
                imgdata[:,:,1] = im_cropped[1,:,:]
                imgdata[:,:,2] = im_cropped[0,:,:]
                cropimg = Image.fromarray(imgdata)
                nnn = '/home/tianjun/tests/img_%d.jpg' % (count)
                cropimg.save(nnn, format = 'JPEG')
                '''

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

    def get_multiview_mb(self):
        '''  Multiview testing will get better accuracy than single view testing. For each image, it will crop out the left-top, right-top, left-down, right-down, central patches and their hirizontal flipped version. The final prediction is averaged according to the 10 views. Thus, for each original batch, get_multiview_mb will produce 10 consecutive batches for the batch.

        '''


        env = lmdb.open(self.source, readonly=True)
        view_num = 10
        ori_size = -1
        samples = np.zeros([view_num, self.batch_size, self.crop_size ** 2 * 3], dtype=np.float32)
        num_label = -1
        count = 0
        with env.begin(write=False, buffers=False) as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                d = Datum()
                d.ParseFromString(value)
                if ori_size == -1:
                    ori_size = np.sqrt(len(d.data) / 3)
                    diff_size = ori_size - self.crop_size
                    start_h = [0, diff_size, 0, diff_size, diff_size/2]
                    start_w = [0, 0, diff_size, diff_size, diff_size/2]

                im = np.fromstring(d.data, dtype=np.uint8).reshape([3, ori_size, ori_size]) - self.mean_data
                
                for i in range(view_num):
                    crop_h = start_h[i/2]
                    crop_w = start_w[i/2]
                    im_cropped = im[:, crop_h:crop_h+self.crop_size, crop_w:crop_w+self.crop_size]
                    if i%2 == 1:
                        im_cropped = im_cropped[:,:,::-1]
                    samples[i, count, :] = im_cropped.reshape(self.crop_size ** 2 * 3).astype(np.float32)
                   
                if num_label == -1:
                    num_label = len(d.label)
                    labels = np.zeros([self.batch_size, num_label], dtype=np.float32)
                labels[count, :] = d.label
                
                count = count + 1
                if count == self.batch_size:
                    for i in range(view_num):
                        yield (samples[i,:,:], labels)
                    labels = np.zeros([self.batch_size, num_label], dtype=np.float32)
                    count = 0
        if count != self.batch_size:
            delete_idx = np.arange(count, self.batch_size)
            left_samples = np.delete(samples, delete_idx, 1)
            left_labels = np.delete(labels, delete_idx, 0)
            for i in range(view_num):
                yield (left_samples[i,:,:], left_labels)


if __name__ == '__main__':
    ''' 
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
    '''
    
    net_file = '/home/tianjun/configfile/Alexmodel/filternet.prototxt'
    with open(net_file, 'r') as f:
        netconfig = NetParameter()
        text_format.Merge(str(f.read()), netconfig)
    layerinfo = netconfig.layers[0]
    dp = ImageWindowDataProvider(layerinfo.window_data_param, 1)
    count = 0
    
    last = time.time()

    for (samples, labels) in dp.get_mb():
        print count, ':', samples.shape
        thistime = time.time() - last
        print thistime
        last = time.time()
        count = count + 1
    

