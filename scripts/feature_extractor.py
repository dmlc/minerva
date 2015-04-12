#!/env/python

import math
import sys, argparse
import time
import numpy as np
import owl
import owl.net as net
from owl.net_helper import CaffeNetBuilder

class NetTrainer:
    def __init__(self, solver_file, snapshot, gpu_idx = 1):
        self.solver_file = solver_file
        self.snapshot = snapshot
        self.num_gpu = 1
        self.gpu = [owl.create_gpu_device(i) for i in range(gpu_idx)]
        owl.set_device(self.gpu[gpu_idx-1])

    def build_net(self):
        self.owl_net = net.Net()
        self.builder = CaffeNetBuilder(self.solver_file)
        self.snapshot_dir = self.builder.snapshot_dir
        self.builder.build_net(self.owl_net, self.num_gpu)
        self.owl_net.init_layer_size('TEST')
        self.builder.init_net_from_file(self.owl_net, self.snapshot_dir, self.snapshot)

    def run(s, layer_name, feature_path):
        '''
        # decide whether to test
        acc_num = 0
        test_num = 0
        for testiteridx in range(s.owl_net.solver.test_iter[0]):
            s.owl_net.forward('TEST')
            all_accunits = s.owl_net.get_accuracy_units()
            accunit = all_accunits[len(all_accunits)-1]
            test_num += accunit.batch_size
            acc_num += (accunit.batch_size * accunit.acc)
            print "Accuracy the %d mb: %f" % (testiteridx, accunit.acc)
            sys.stdout.flush()
        print "Testing Accuracy: %f" % (float(acc_num)/test_num)
        exit(1)
        '''
        feature_unit = s.owl_net.units[s.owl_net.name_to_uid[layer_name][0]] 
        feature_file = open(feature_path, 'w')
        batch_dir = 0
        for testiteridx in range(s.owl_net.solver.test_iter[0]):
            s.owl_net.forward('TEST')
            feature = feature_unit.out.to_numpy()
            feature_shape = np.shape(feature)
            img_num = feature_shape[0]
            feature_length = np.prod(feature_shape[1:len(feature_shape)])
            feature = np.reshape(feature, [img_num, feature_length])
            for imgidx in range(img_num):
                for feaidx in range(feature_length):
                    info ='%f ' % (feature[imgidx, feaidx])
                    feature_file.write(info)
                feature_file.write('\n')
            print "Finish One Batch %d" % (batch_dir)
            batch_dir += 1
        feature_file.close()
            



if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('solver_file', help='caffe solver configure file')
    parser.add_argument('layer_name', help='layer_name')
    parser.add_argument('feature_path', help='feature_path')
    parser.add_argument('-n', '--gpu_idx', help='gpu to use', action='store', type=int, default=1)
    parser.add_argument('--snapshot', help='the snapshot idx to start from', action='store', type=int)
    (args, remain) = parser.parse_known_args()
    solver_file = args.solver_file
    layer_name = args.layer_name
    feature_path = args.feature_path
    gpu_idx = args.gpu_idx
    snapshot = args.snapshot

    print ' === Using the %dth gpu, start from snapshot %d === ' % (gpu_idx, snapshot)

    sys_args = [sys.argv[0]] + remain
    owl.initialize(sys_args)
    trainer = NetTrainer(solver_file, snapshot, gpu_idx)
    trainer.build_net()
    trainer.run(layer_name, feature_path)
