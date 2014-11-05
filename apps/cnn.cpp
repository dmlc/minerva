#include <minerva.h>
#include <fstream>
#include <gflags/gflags.h>

DEFINE_bool(init, false, "Initialize weights");

using namespace std;
using namespace minerva;

const int numepochs = 10;
const int mb_size = 256;
const int num_mb_per_epoch = 60000 / mb_size;
float alpha = 0.01 / mb_size;

const string train_data_file = "/home/cs_user/data/mnist/traindata.dat";
const string train_label_file = "/home/cs_user/data/mnist/trainlabel.dat";

vector<NArray> weights;
vector<NArray> bias;
vector<NArray> acts;
vector<NArray> sens;

void PrintTrainingAccuracy(NArray o, NArray t) {
  //get predict
  NArray predict = o.Reshape({10, mb_size}).MaxIndex(0);
  NArray groundtruth = t.Reshape({10, mb_size}).MaxIndex(0);
  float correct = (predict - groundtruth).CountZero();
  cout << "Training Error: " << (mb_size - correct) / mb_size << endl;
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  MinervaSystem& ms = MinervaSystem::Instance();
  ms.Initialize(&argc, &argv);
  uint64_t cpuDevice = ms.CreateCpuDevice();
  uint64_t gpuDevice = ms.CreateGpuDevice(0);
  ms.current_device_id_ = cpuDevice;

  weights.resize(3);
  bias.resize(3);
  acts.resize(9);
  sens.resize(9);
  if (FLAGS_init) {
    weights[0] = Filter(NArray::Randn({5, 5, 1, 8}, 0.0, 0.1));
    bias[0] = NArray::Randn({8}, 0.0, 0.1);
    weights[1] = Filter(NArray::Randn({5, 5, 8, 16}, 0.0, 0.1));
    bias[1] = NArray::Randn({16}, 0.0, 0.1);
    weights[2] = NArray::Randn({10, 256}, 0.0, 0.1);
    bias[2] = NArray::Randn({10, 1}, 0.0, 0.1);
    ofstream cnn_init("cnn_init", ios::binary);
    auto ptr = weights[0].Get();
    cnn_init.write(reinterpret_cast<char*>(ptr.get()), weights[0].Size().Prod() * sizeof(float));
    ptr = bias[0].Get();
    cnn_init.write(reinterpret_cast<char*>(ptr.get()), bias[0].Size().Prod() * sizeof(float));
    ptr = weights[1].Get();
    cnn_init.write(reinterpret_cast<char*>(ptr.get()), weights[1].Size().Prod() * sizeof(float));
    ptr = bias[1].Get();
    cnn_init.write(reinterpret_cast<char*>(ptr.get()), bias[1].Size().Prod() * sizeof(float));
    ptr = weights[2].Get();
    cnn_init.write(reinterpret_cast<char*>(ptr.get()), weights[2].Size().Prod() * sizeof(float));
    ptr = bias[2].Get();
    cnn_init.write(reinterpret_cast<char*>(ptr.get()), bias[2].Size().Prod() * sizeof(float));
    cnn_init.close();
  } else {
    ifstream cnn_init("cnn_init", ios::binary);
    Scale weights_size[] = {{5, 5, 1, 8}, {5, 5, 8, 16}, {10, 256}};
    Scale bias_size[] = {{8}, {16}, {10, 1}};
    for (int i = 0; i < 3; ++i) {
      shared_ptr<float> weight_ptr(new float[weights_size[i].Prod()], [](float* ptr) { delete[] ptr; });
      cnn_init.read(reinterpret_cast<char*>(weight_ptr.get()), weights_size[i].Prod() * sizeof(float));
      weights[i] = NArray::MakeNArray(weights_size[i], weight_ptr);
      shared_ptr<float> bias_ptr(new float[bias_size[i].Prod()], [](float* ptr) { delete[] ptr; });
      cnn_init.read(reinterpret_cast<char*>(bias_ptr.get()), bias_size[i].Prod() * sizeof(float));
      bias[i] = NArray::MakeNArray(bias_size[i], bias_ptr);
    }
    cnn_init.close();
  }
  cout << "Training procedure:" << endl;
  for(int epoch = 0; epoch < numepochs; ++ epoch) {
    ifstream data_file_in(train_data_file.c_str(), ios::binary);
    ifstream label_file_in(train_label_file.c_str(), ios::binary);
    data_file_in.ignore(2 * sizeof(float));
    label_file_in.ignore(2 * sizeof(float));
    cout << "Epoch #" << epoch << endl;
    for(int mb = 0; mb < num_mb_per_epoch; ++mb) {
      ms.current_device_id_ = cpuDevice;
      Scale data_size{28, 28, 1, mb_size};
      Scale label_size{10, 1, 1, mb_size};
      shared_ptr<float> data_ptr(new float[data_size.Prod()], [](float* ptr) { delete[] ptr; });
      shared_ptr<float> label_ptr(new float[label_size.Prod()], [](float* ptr) { delete[] ptr; });
      data_file_in.read(reinterpret_cast<char*>(data_ptr.get()), data_size.Prod() * sizeof(float));
      label_file_in.read(reinterpret_cast<char*>(label_ptr.get()), label_size.Prod() * sizeof(float));
      acts[0] = NArray::MakeNArray(data_size, data_ptr);
      NArray label = NArray::MakeNArray(label_size, label_ptr);

      ms.current_device_id_ = gpuDevice;
      ConvInfo conv_info;
      conv_info.pad_height = 0;
      conv_info.pad_width = 0;
      conv_info.stride_vertical = 1;
      conv_info.stride_horizontal = 1;
      PoolingInfo pool_info;
      pool_info.algorithm = PoolingInfo::Algorithm::kMax;
      pool_info.height = 2;
      pool_info.width = 2;
      pool_info.stride_vertical = 2;
      pool_info.stride_horizontal = 2;
      acts[1] = Convolution::ConvForward(acts[0], weights[0], bias[0], conv_info);
      acts[2] = Convolution::ActivationForward(acts[1], ActivationAlgorithm::kRelu);
      acts[3] = Convolution::PoolingForward(acts[2], pool_info);
      conv_info.pad_height = 2;
      conv_info.pad_width = 2;
      acts[4] = Convolution::ConvForward(acts[3], weights[1], bias[1], conv_info);
      acts[5] = Convolution::ActivationForward(acts[4], ActivationAlgorithm::kRelu);
      pool_info.height = 3;
      pool_info.width = 3;
      pool_info.stride_vertical = 3;
      pool_info.stride_horizontal = 3;
      acts[6] = Convolution::PoolingForward(acts[5], pool_info);
      auto re_acts6 = acts[6].Reshape({acts[6].Size().Prod() / mb_size, mb_size});
      acts[7] = (weights[2] * re_acts6).NormArithmetic(bias[2], ArithmeticType::kAdd);
      acts[8] = Convolution::SoftmaxForward(acts[7].Reshape({10, 1, 1, mb_size}), SoftmaxAlgorithm::kInstance);

      sens[8] = acts[8] - label;

      // sens[7] = Convolution::SoftmaxBackward(sens[8], acts[8], SoftmaxAlgorithm::kInstance).Reshape({10, mb_size});
      sens[7] = sens[8].Reshape({10, mb_size});
      sens[6] = (weights[2].Trans() * sens[7]).Reshape(acts[6].Size());
      sens[5] = Convolution::PoolingBackward(sens[6], acts[6], acts[5], pool_info);
      sens[4] = Convolution::ActivationBackward(sens[5], acts[5], acts[4], ActivationAlgorithm::kRelu);
      sens[3] = Convolution::ConvBackwardData(sens[4], weights[1], conv_info);
      pool_info.height = 2;
      pool_info.width = 2;
      pool_info.stride_vertical = 2;
      pool_info.stride_horizontal = 2;
      sens[2] = Convolution::PoolingBackward(sens[3], acts[3], acts[2], pool_info);
      sens[1] = Convolution::ActivationBackward(sens[2], acts[2], acts[1], ActivationAlgorithm::kRelu);

      bias[2] -= alpha * sens[7].Sum(1);
      weights[2] -= alpha * sens[7] * re_acts6.Trans();
      bias[1] -= alpha * Convolution::ConvBackwardBias(sens[4]);
      weights[1] -= alpha * Convolution::ConvBackwardFilter(sens[4], acts[3], conv_info);
      conv_info.pad_height = conv_info.pad_width = 0;
      bias[0] -= alpha * Convolution::ConvBackwardBias(sens[1]);
      weights[0] -= alpha * Convolution::ConvBackwardFilter(sens[1], acts[0], conv_info);

      if (mb % 20 == 0) {
        PrintTrainingAccuracy(acts[8], label);
      }
    }
    data_file_in.close();
    label_file_in.close();
  }
  ms.current_device_id_ = cpuDevice;

  // output weights
  cout << "Write weight to files" << endl;
  weights.clear();
  bias.clear();
  acts.clear();
  sens.clear();
  ms.dag_scheduler().GCNodes();
  cout << ms.device_manager().GetDevice(gpuDevice)->GetMemUsage() << endl;
  cout << ms.device_manager().GetDevice(cpuDevice)->GetMemUsage() << endl;
  cout << ms.physical_dag().PrintDag<ExternRCPrinter>() << endl;
  cout << "Training finished." << endl;
  return 0;
}
