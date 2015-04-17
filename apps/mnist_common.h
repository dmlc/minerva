#pragma once

#include <minerva.h>
#include <iostream>
#include <fstream>
#include <gflags/gflags.h>
#include <cstring>
#include <memory>
#include <utility>

using namespace std;
using namespace minerva;

DEFINE_double(alpha, 0.01, "Learning rate");
DEFINE_int32(epoch, 10, "Number of epochs to train");
DEFINE_int32(mb, 256, "Minibatch size");
DEFINE_int32(num_mb, 234, "Number of minibatch to train for each epoch");
DEFINE_string(train_data_file, "./traindata.dat", "Path of MNIST training data");
DEFINE_string(train_label_file, "./trainlabel.dat", "Path of MNIST training label");
DEFINE_string(test_data_file, "./testdata.dat", "Path to the MNIST testing data");
DEFINE_string(test_label_file, "./testlabel.dat", "Path to the MNIST testing label");

typedef struct {
  float alpha;
  int num_epochs;
  int mb_size;
  int num_mb;
  string train_data_file, train_label_file;
  string test_data_file, test_label_file;
} MnistParam;

ostream& operator << (ostream& os, const MnistParam& param) {
  return os << "Parameters:\n"
    << "  Learning rate: " << param.alpha << "\n"
    << "  Epochs: " << param.num_epochs << "\n"
    << "  Minibatch size: " << param.mb_size << "\n"
    << "  Num minibatches run per epoch: " << param.num_mb << "\n"
    << "  Training data/label path: " << param.train_data_file << "\n"
    << "                            " << param.train_label_file << "\n"
    << "  Testing data/label path: " << param.test_data_file << "\n"
    << "                           " << param.test_label_file
    ;
}

inline void PrintTrainingAccuracy(NArray o, NArray t, const MnistParam& param) {
  NArray predict = o.Reshape({10, param.mb_size}).MaxIndex(0);
  shared_ptr<float> pp = o.Get();
  NArray groundtruth = t.Reshape({10, param.mb_size}).MaxIndex(0);
  float correct = (predict - groundtruth).CountZero();
  cout << "Training Error: " << (param.mb_size - correct) / param.mb_size << endl;
}

inline shared_ptr<float> CreateFPtr(size_t len) {
  return shared_ptr<float> ( new float[len], [](float* ptr) { delete[] ptr; } );
}

inline pair<shared_ptr<float>, shared_ptr<float>> GetNextBatch(ifstream& data_file_in, ifstream& label_file_in, size_t data_length, size_t label_length) {
  auto data_ptr = CreateFPtr(data_length);
  auto label_ptr = CreateFPtr(label_length);
  data_file_in.read(reinterpret_cast<char*>(data_ptr.get()), data_length * sizeof(float));
  label_file_in.read(reinterpret_cast<char*>(label_ptr.get()), label_length * sizeof(float));
  return make_pair(data_ptr, label_ptr);
}

inline void PrintImgAndLabel(shared_ptr<float> data_ptr, shared_ptr<float> label_ptr, int img_idx = 0) {
  for(int i = img_idx * 784; i < (img_idx + 1) * 784; ++i) {
    float x = data_ptr.get()[i];
    cout << (x > 1e-6? "1" : "") << " ";
    if( i % 28 == 0) {
      cout << endl;
    }
  }
  cout << endl;
  for(int i = img_idx * 10; i < (img_idx + 1) * 10; ++i) {
    cout << label_ptr.get()[i] << " ";
  }
  cout << endl;
}

class MnistAlgorithm {
 public:
  MnistAlgorithm(const MnistParam& p): param_(p) {}
  virtual void Init() = 0;
  virtual NArray FF(shared_ptr<float> data_ptr, bool test) = 0;
  virtual NArray BP(shared_ptr<float> label_ptr) = 0;
  virtual void Update() = 0;
 protected:
  MnistParam param_;
};

inline MnistParam InitMnistApps(int argc, char** argv) {
  // parse commandline
  MnistParam param;
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  param.alpha = FLAGS_alpha;
  param.num_epochs = FLAGS_epoch;
  param.mb_size = FLAGS_mb;
  param.num_mb = FLAGS_num_mb;
  param.train_data_file = FLAGS_train_data_file;
  param.train_label_file = FLAGS_train_label_file;
  param.test_data_file = FLAGS_test_data_file;
  param.test_label_file = FLAGS_test_label_file;
  // initial minerva system
  MinervaSystem::Initialize(&argc, &argv);
  MinervaSystem& ms = MinervaSystem::Instance();
  uint64_t cpuDevice = ms.CreateCpuDevice();
  ms.SetDevice(cpuDevice);
#ifdef HAS_CUDA
  uint64_t gpuDevice = ms.CreateGpuDevice(0);
  ms.SetDevice(gpuDevice);
#endif
  return param;
}

class MnistCnnAlgo : public MnistAlgorithm {
 public:
  MnistCnnAlgo(const MnistParam& p): MnistAlgorithm(p) { }
  void Init() override {
    data_size = Scale{28, 28, 1, param_.mb_size};
    label_size = Scale{10, 1, 1, param_.mb_size};
    // convolution
    conv_info[0].pad_height = conv_info[0].pad_width = 0;
    conv_info[0].stride_vertical = conv_info[0].stride_horizontal = 1;
    conv_info[1].pad_height = conv_info[1].pad_width = 2;
    conv_info[1].stride_vertical = conv_info[1].stride_horizontal = 1;
    // pooling
    pool_info[0].algorithm = PoolingInfo::Algorithm::kMax;
    pool_info[0].height = pool_info[0].width = 2;
    pool_info[0].stride_vertical = pool_info[0].stride_horizontal = 2;
    pool_info[1].algorithm = PoolingInfo::Algorithm::kMax;
    pool_info[1].height = pool_info[1].width = 3;
    pool_info[1].stride_vertical = pool_info[1].stride_horizontal = 3;

    weights[0] = Filter(NArray::Randn({5, 5, 1, 16}, 0.0, 0.1));
    bias[0] = NArray::Randn({16}, 0.0, 0.1);
    weights[1] = Filter(NArray::Randn({5, 5, 16, 32}, 0.0, 0.1));
    bias[1] = NArray::Randn({32}, 0.0, 0.1);
    weights[2] = NArray::Randn({10, 512}, 0.0, 0.1);
    bias[2] = NArray::Randn({10, 1}, 0.0, 0.1);

    ResetGrad();
  }
  NArray FF(shared_ptr<float> data_ptr, bool test) override {
    acts[0] = NArray::MakeNArray(data_size, data_ptr);
    acts[1] = Convolution::ConvForward(acts[0], weights[0], bias[0], conv_info[0]);
    acts[2] = Convolution::ActivationForward(acts[1], ActivationAlgorithm::kRelu);
    acts[3] = Convolution::PoolingForward(acts[2], pool_info[0]);
    acts[4] = Convolution::ConvForward(acts[3], weights[1], bias[1], conv_info[1]);
    acts[5] = Convolution::ActivationForward(acts[4], ActivationAlgorithm::kRelu);
    acts[6] = Convolution::PoolingForward(acts[5], pool_info[1]);
    re_acts6 = acts[6].Reshape({acts[6].Size().Prod() / param_.mb_size, param_.mb_size});
    acts[7] = weights[2] * re_acts6 + bias[2];
    acts[8] = Convolution::SoftmaxForward(acts[7].Reshape({10, 1, 1, param_.mb_size}), SoftmaxAlgorithm::kInstance);
    return acts[8];
  }
  NArray BP(shared_ptr<float> label_ptr) override {
    NArray label = NArray::MakeNArray(label_size, label_ptr);
    sens[8] = acts[8] - label;
    sens[7] = sens[8].Reshape({10, param_.mb_size});
    sens[6] = (weights[2].Trans() * sens[7]).Reshape(acts[6].Size());
    sens[5] = Convolution::PoolingBackward(sens[6], acts[6], acts[5], pool_info[1]);
    sens[4] = Convolution::ActivationBackward(sens[5], acts[5], acts[4], ActivationAlgorithm::kRelu);
    sens[3] = Convolution::ConvBackwardData(sens[4], acts[3], weights[1], conv_info[1]);
    sens[2] = Convolution::PoolingBackward(sens[3], acts[3], acts[2], pool_info[0]);
    sens[1] = Convolution::ActivationBackward(sens[2], acts[2], acts[1], ActivationAlgorithm::kRelu);
    // gradient
    grad_weights[0] += Convolution::ConvBackwardFilter(sens[1], acts[0], weights[0], conv_info[0]);
    grad_bias[0] += Convolution::ConvBackwardBias(sens[1]);
    grad_weights[1] += Convolution::ConvBackwardFilter(sens[4], acts[3], weights[1], conv_info[1]);
    grad_bias[1] += Convolution::ConvBackwardBias(sens[4]);
    grad_weights[2] += sens[7] * re_acts6.Trans();
    grad_bias[2] += sens[7].Sum(1);
    return label;
  }
  void Update() override {
    float alpha = param_.alpha;
    int mb_size = param_.mb_size;
    weights[0] -= alpha / mb_size * grad_weights[0];
    bias[0] -= alpha / mb_size * grad_bias[0];
    weights[1] -= alpha / mb_size * grad_weights[1];
    bias[1] -= alpha / mb_size * grad_bias[1];
    weights[2] -= alpha / mb_size * grad_weights[2];
    bias[2] -= alpha / mb_size * grad_bias[2];

    ResetGrad();
  }
 private:
  void ResetGrad() {
    grad_weights[0] = NArray::Zeros({5, 5, 1, 16});
    grad_bias[0] = NArray::Zeros({16});
    grad_weights[1] = NArray::Zeros({5, 5, 16, 32});
    grad_bias[1] = NArray::Zeros({32});
    grad_weights[2] = NArray::Zeros({10, 512});
    grad_bias[2] = NArray::Zeros({10, 1});
  }
 private:
  Scale data_size, label_size;
  ConvInfo conv_info[2];
  PoolingInfo pool_info[2];
  NArray weights[3], bias[3], grad_weights[3], grad_bias[3];
  NArray acts[9], sens[9];
  NArray re_acts6;
};

class MnistMlpAlgo : public MnistAlgorithm {
 public:
  MnistMlpAlgo(const MnistParam& p): MnistAlgorithm(p) { }
  void Init() override {
    data_size = Scale{784, param_.mb_size};
    label_size = Scale{10, param_.mb_size};

    weights[0] = NArray::Randn({256, 784}, 0.0, 0.1);
    bias[0] = NArray::Randn({256, 1}, 0.0, 0.1);
    weights[1] = NArray::Randn({10, 256}, 0.0, 0.1);
    bias[1] = NArray::Randn({10, 1}, 0.0, 0.1);

    ResetGrad();
  }
  NArray FF(shared_ptr<float> data_ptr, bool test) override {
    acts[0] = NArray::MakeNArray(data_size, data_ptr);
    acts[1] = Elewise::ReluForward(weights[0] * acts[0] + bias[0]);
    acts[2] = weights[1] * acts[1] + bias[1];
    acts[3] = Convolution::SoftmaxForward(acts[2].Reshape({10, 1, 1, param_.mb_size}), SoftmaxAlgorithm::kInstance).Reshape({10, param_.mb_size});
    return acts[3];
  }
  NArray BP(shared_ptr<float> label_ptr) override {
    NArray label = NArray::MakeNArray(label_size, label_ptr);
    sens[2] = acts[3] - label;
    sens[1] = Elewise::ReluBackward(weights[1].Trans() * sens[2], acts[1], acts[1]);
    // gradient
    grad_weights[0] += sens[1] * acts[0].Trans();
    grad_bias[0] += sens[1].Sum(1);
    grad_weights[1] += sens[2] * acts[1].Trans();
    grad_bias[1] += sens[2].Sum(1);
    return label;
  }
  void Update() override {
    float alpha = param_.alpha;
    int mb_size = param_.mb_size;
    weights[0] -= alpha / mb_size * grad_weights[0];
    bias[0] -= alpha / mb_size * grad_bias[0];
    weights[1] -= alpha / mb_size * grad_weights[1];
    bias[1] -= alpha / mb_size * grad_bias[1];

    ResetGrad();
  }
 private:
  void ResetGrad() {
    grad_weights[0] = NArray::Zeros({256, 784});
    grad_bias[0] = NArray::Zeros({256, 1});
    grad_weights[1] = NArray::Zeros({10, 256});
    grad_bias[1] = NArray::Zeros({10, 1});
  }
 private:
  Scale data_size, label_size;
  NArray weights[2], bias[2], grad_weights[2], grad_bias[2];
  NArray acts[9], sens[9];
};
