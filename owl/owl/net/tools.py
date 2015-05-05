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
        data_unit = s.owl_net.units[s.owl_net.name_to_uid['data'][0]] 
        cur_img = 0
        for testiteridx in range(s.owl_net.solver.test_iter[0]):
            s.owl_net.forward('TEST')
            feature = feature_unit.out.to_numpy()
            feature_shape = np.shape(feature)
            img_num = feature_shape[0]
            #processing each image
            for imgidx in range(img_num):
                img_feature = feature[imgidx,:]
                f_h = feature_shape[2]
                f_w = feature_shape[3]
                f_c = feature_shape[1]
                heatmap = np.zeros([f_h, f_w], dtype=np.float32)
                for cidx in range(f_c):
                    feature_map = img_feature[cidx,:]
                    f = np.max(np.max(feature_map)) - np.mean(np.mean(feature_map))
                    #
                    #heatmap = heatmap + f * f * feature_map
                    heatmap = heatmap + feature_map
                #resize
                heatmap = scipy.misc.imresize(heatmap,[data_unit.crop_size, data_unit.crop_size])
                #save
                fig, ax = plt.subplots()
                ax = plt.pcolor(heatmap)
                info = '%s/%d.jpg' % (result_path, cur_img) 
                fig.savefig(info)
                cur_img += 1
                if cur_img == 100:
                    exit(0)
            
            print "Finish One Batch %d" % (batch_dir)
            batch_dir += 1
        feature_file.close()




