import math
import sys
import time
import numpy as np
import owl
from net import Net
from net_helper import CaffeNetBuilder

class NetTrainer:
    def __init__(self, solver_file, snapshot, num_gpu = 1):
        self.solver_file = solver_file
        self.snapshot = snapshot
        self.num_gpu = num_gpu
        self.gpu = [owl.create_gpu_device(i) for i in range(num_gpu)]

    def build_net(self):
        self.owl_net = Net()
        self.builder = CaffeNetBuilder(self.solver_file)
        self.snapshot_dir = self.builder.snapshot_dir
        self.builder.build_net(self.owl_net, self.num_gpu)
        self.owl_net.compute_size()
        self.builder.init_net_from_file(self.owl_net, self.snapshot_dir, self.snapshot)

    def run(s):
        wgrad = [[] for i in range(s.num_gpu)]
        bgrad = [[] for i in range(s.num_gpu)]
        last = time.time()
        wunits = s.owl_net.get_weighted_unit_ids()
        last_start = time.time()

        for iteridx in range(s.snapshot * s.owl_net.solver.snapshot, s.owl_net.solver.max_iter):
            # get the learning rate
            if s.owl_net.solver.lr_policy == "poly":
                s.owl_net.current_lr = s.owl_net.base_lr * pow(1 - float(iteridx) / s.owl_net.solver.max_iter, s.owl_net.solver.power)
            elif s.owl_net.solver.lr_policy == "step":
                s.owl_net.current_lr = s.owl_net.base_lr * pow(s.owl_net.solver.gamma, iteridx / s.owl_net.solver.stepsize)

            # train on multi-gpu
            for gpuid in range(s.num_gpu):
                owl.set_device(s.gpu[gpuid])
                s.owl_net.forward('TRAIN')
                s.owl_net.backward('TRAIN')
                for wid in wunits:
                    wgrad[gpuid].append(s.owl_net.units[wid].weightgrad)
                    bgrad[gpuid].append(s.owl_net.units[wid].biasgrad)

            # weight update
            for i in range(len(wunits)):
                wid = wunits[i]
                upd_gpu = i * num_gpu / len(wunits)
                owl.set_device(s.gpu[upd_gpu])
                for gid in range(s.num_gpu):
                    if gid == upd_gpu:
                        continue
                    wgrad[upd_gpu][i] += wgrad[gid][i]
                    bgrad[upd_gpu][i] += bgrad[gid][i]
                s.owl_net.units[wid].weightgrad = wgrad[upd_gpu][i]
                s.owl_net.units[wid].biasgrad = bgrad[upd_gpu][i]
                s.owl_net.update(wid)

            if iteridx % 2 == 0:
                owl.wait_for_all()
                thistime = time.time() - last
                print "Finished training %d minibatch (time: %s)" % (iteridx, thistime)
                last = time.time()

            wgrad = [[] for i in range(s.num_gpu)] # reset gradients
            bgrad = [[] for i in range(s.num_gpu)]

            # decide whether to display loss
            if (iteridx + 1) % (s.owl_net.solver.display) == 0:
                lossunits = s.owl_net.get_loss_units()
                for lu in lossunits:
                    print "Training Loss %s: %f" % (lu.name, lu.getloss())

            # decide whether to test
            if (iteridx + 1) % (s.owl_net.solver.test_interval) == 0:
                acc_num = 0
                test_num = 0
                for testiteridx in range(s.owl_net.solver.test_iter[0]):
                    s.owl_net.forward('TEST')
                    all_accunits = s.owl_net.get_accuracy_units()
                    accunit = all_accunits[len(all_accunits)-1]
                    #accunit = all_accunits[0]
                    test_num += accunit.batch_size
                    acc_num += (accunit.batch_size * accunit.acc)
                    print "Accuracy the %d mb: %f" % (testiteridx, accunit.acc)
                    sys.stdout.flush()
                print "Testing Accuracy: %f" % (float(acc_num)/test_num)

            # decide whether to save model
            if (iteridx + 1) % (s.owl_net.solver.snapshot) == 0:
                print "Save to snapshot %d, current lr %f" % ((iteridx + 1) / (s.owl_net.solver.snapshot), s.owl_net.current_lr)
                s.builder.save_net_to_file(s.owl_net, s.snapshot_dir, (iteridx + 1) / (s.owl_net.solver.snapshot))
            sys.stdout.flush()

class MultiviewTester:
    def __init__(self, solver_file, snapshot, gpu_idx = 0):
        self.solver_file = solver_file
        self.snapshot = snapshot
        self.gpu = owl.create_gpu_device(gpu_idx)
        owl.set_device(self.gpu)

    def build_net(self):
        self.owl_net = net.Net()
        self.builder = CaffeNetBuilder(self.solver_file)
        self.snapshot_dir = self.builder.snapshot_dir
        self.builder.build_net(self.owl_net)
        self.owl_net.compute_size('MULTI_VIEW')
        self.builder.init_net_from_file(self.owl_net, self.snapshot_dir, self.snapshot)

    def run(s):
        #multi-view test
        acc_num = 0
        test_num = 0
        loss_unit = s.owl_net.units[s.owl_net.name_to_uid['loss'][0]] 
        for testiteridx in range(s.owl_net.solver.test_iter[0]):
            for i in range(10): 
                s.owl_net.forward('MULTI_VIEW')
                if i == 0:
                    softmax_val = loss_unit.ff_y
                    batch_size = softmax_val.shape[1]
                    softmax_label = loss_unit.y
                else:
                    softmax_val = softmax_val + loss_unit.ff_y
            
            test_num += batch_size
            predict = softmax_val.argmax(0)
            truth = softmax_label.argmax(0)
            correct = (predict - truth).count_zero()
            acc_num += correct
            print "Accuracy the %d mb: %f, batch_size: %d" % (testiteridx, correct, batch_size)
            sys.stdout.flush()
        print "Testing Accuracy: %f" % (float(acc_num)/test_num)

class FeatureExtractor:
    def __init__(self, solver_file, snapshot, gpu_idx = 0):
        self.solver_file = solver_file
        self.snapshot = snapshot
        self.gpu = owl.create_gpu_device(gpu_idx)
        owl.set_device(self.gpu)

    def build_net(self):
        self.owl_net = net.Net()
        self.builder = CaffeNetBuilder(self.solver_file)
        self.snapshot_dir = self.builder.snapshot_dir
        self.builder.build_net(self.owl_net)
        self.owl_net.compute_size('TEST')
        self.builder.init_net_from_file(self.owl_net, self.snapshot_dir, self.snapshot)

    def run(s, layer_name, feature_path):
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
