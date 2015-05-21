import math
import sys
import time
import numpy as np
import owl
from net import Net
from net_helper import CaffeNetBuilder
from caffe import *
from PIL import Image
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class HeatmapVisualizer:
    ''' Class of heatmap visualizer.
    Heat map can reveal which part of the activation is important. This information is useful when conducting detection and segmentation tasks.

    :ivar str solver_file: name of the solver_file, it will tell Minerva the network configuration and model saving path 
    :ivar snapshot: saved model snapshot index
    :ivar str layer_name: name of the layer whose activation will be viusualized as heatmap
    :ivar str result_path: path for the result of visualization, heatmapvisualizer will generate a heatmap jpg for each testing image and save the image under result path. 
    :ivar gpu: the gpu to run testing

    '''
    def __init__(self, solver_file, snapshot, gpu_idx = 0):
        self.solver_file = solver_file
        self.snapshot = snapshot
        self.gpu = owl.create_gpu_device(gpu_idx)
        owl.set_device(self.gpu)

    def build_net(self):
        self.owl_net = Net()
        self.builder = CaffeNetBuilder(self.solver_file)
        self.snapshot_dir = self.builder.snapshot_dir
        self.builder.build_net(self.owl_net)
        self.owl_net.compute_size('TEST')
        self.builder.init_net_from_file(self.owl_net, self.snapshot_dir, self.snapshot)

    def run(s, layer_name, result_path):
        ''' Run heatmap visualizer

        :param str layer_name: the layer to visualize
        :param str result_path: the path to save heatmap
        '''
        feature_unit = s.owl_net.units[s.owl_net.name_to_uid[layer_name][0]] 
        #We need the testing data unit
        data_unit = None
        for i in range(len(s.owl_net.name_to_uid['data'])):
            if s.owl_net.units[s.owl_net.name_to_uid['data'][i]].params.include[0].phase == 1:
                data_unit = s.owl_net.units[s.owl_net.name_to_uid['data'][i]]
        assert(data_unit)
        
        #get the mean data
        bp = BlobProto()
        #get mean file
        if len(data_unit.params.transform_param.mean_file) == 0:
            mean_data = np.ones([3, 256, 256], dtype=np.float32)
            assert(len(data_unit.params.transform_param.mean_value) == 3)
            mean_data[0] = data_unit.params.transform_param.mean_value[0]
            mean_data[1] = data_unit.params.transform_param.mean_value[1]
            mean_data[2] = data_unit.params.transform_param.mean_value[2]
            h_w = 256
        else:    
            with open(data_unit.params.transform_param.mean_file, 'rb') as f:
                bp.ParseFromString(f.read())
            mean_narray = np.array(bp.data, dtype=np.float32)
            h_w = np.sqrt(np.shape(mean_narray)[0] / 3)
            mean_data = np.array(bp.data, dtype=np.float32).reshape([3, h_w, h_w])
        #get the cropped img
        crop_size = data_unit.params.transform_param.crop_size
        crop_h_w = (h_w - crop_size) / 2
        mean_data = mean_data[:, crop_h_w:crop_h_w + crop_size, crop_h_w:crop_h_w + crop_size]
        
        cur_img = 0
        for testiteridx in range(s.owl_net.solver.test_iter[0]):
            s.owl_net.forward('TEST')
            feature = feature_unit.out.to_numpy()
            feature_shape = np.shape(feature)
            data = data_unit.out.to_numpy()
            img_num = feature_shape[0]
            #processing each image
            for imgidx in range(img_num):
                img_feature = feature[imgidx,:]
                #get the image
                gbr_img_data = data[imgidx,:] + mean_data
                img_data = np.zeros([data_unit.crop_size, data_unit.crop_size, 3], dtype=np.float32)
                img_data[:,:,0] = gbr_img_data[2,:,:]
                img_data[:,:,1] = gbr_img_data[1,:,:]
                img_data[:,:,2] = gbr_img_data[0,:,:]
                img_data /= 256
                #get the heatmap
                f_h = feature_shape[2]
                f_w = feature_shape[3]
                f_c = feature_shape[1]
                heatmap = np.zeros([f_h, f_w], dtype=np.float32)
                for cidx in range(f_c):
                    feature_map = img_feature[cidx,:]
                    f = np.max(np.max(feature_map)) - np.mean(np.mean(feature_map))
                    heatmap = heatmap + f * f * feature_map
                #resize
                heatmap = scipy.misc.imresize(heatmap,[data_unit.crop_size, data_unit.crop_size])
                #save
                fig, ax = plt.subplots(1,2)
                ax[0].axis('off')
                ax[1].axis('off')
                ax[0].imshow(img_data, aspect='equal')
                ax[1].imshow(heatmap, aspect='equal')
                #ax[1] = plt.pcolor(heatmap)

                info = '%s/%d.jpg' % (result_path, cur_img) 
                print info
                fig.savefig(info)
                plt.close('all')
                cur_img += 1
            
            print "Finish One Batch %d" % (testiteridx)
        feature_file.close()




