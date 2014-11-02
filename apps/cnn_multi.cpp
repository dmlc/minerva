#include <minerva.h>
#include <fstream>

using namespace std;
using namespace minerva;

float alpha = 0.01;
const int numepochs = 10;
const int mb_size = 256;
const int num_mb_per_epoch = 60000 / mb_size;

const string train_data_file = "/home/cs_user/data/mnist/traindata.dat";
const string train_label_file = "/home/cs_user/data/mnist/trainlabel.dat";

uint64_t cpu_device;

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

void Train(uint64_t gpu_device, ifstream& data_file_in, ifstream& label_file_in, bool print_error) {
  auto& ms = MinervaSystem::Instance();
  ms.current_device_id_ = cpu_device;
  Scale data_size{28, 28, 1, mb_size};
  Scale label_size{10, 1, 1, mb_size};
  shared_ptr<float> data_ptr(new float[data_size.Prod()], [](float* ptr) { delete[] ptr; });
  shared_ptr<float> label_ptr(new float[label_size.Prod()], [](float* ptr) { delete[] ptr; });
  data_file_in.read(reinterpret_cast<char*>(data_ptr.get()), data_size.Prod() * sizeof(float));
  label_file_in.read(reinterpret_cast<char*>(label_ptr.get()), label_size.Prod() * sizeof(float));
  acts[0] = NArray::MakeNArray(data_size, data_ptr);
  NArray label = NArray::MakeNArray(label_size, label_ptr);

  ms.current_device_id_ = gpu_device;
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

  sens[7] = Convolution::SoftmaxBackward(sens[8], acts[8], SoftmaxAlgorithm::kInstance).Reshape({10, mb_size});
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

  if (print_error) {
    PrintTrainingAccuracy(acts[8], label);
  }
}

int main(int argc, char** argv) {
  MinervaSystem& ms = MinervaSystem::Instance();
  ms.Initialize(&argc, &argv);
  cpu_device = ms.CreateCpuDevice();
  uint64_t gpu_devices[2];
  gpu_devices[0] = ms.CreateGpuDevice(0);
  gpu_devices[1] = ms.CreateGpuDevice(1);
  ms.current_device_id_ = cpu_device;

  weights.resize(3);
  bias.resize(3);
  acts.resize(9);
  sens.resize(9);
  // Initialize
  weights[0] = Filter(NArray::Randn({5, 5, 1, 8}, 0.0, 0.1));
  bias[0] = NArray::Randn({8}, 0.0, 0.1);
  weights[1] = Filter(NArray::Randn({5, 5, 8, 16}, 0.0, 0.1));
  bias[1] = NArray::Randn({16}, 0.0, 0.1);
  weights[2] = NArray::Randn({10, 256}, 0.0, 0.1);
  bias[2] = NArray::Randn({10, 1}, 0.0, 0.1);
  cout << "Training procedure:" << endl;

  for(int epoch = 0; epoch < numepochs; ++ epoch) {
    ifstream data_file_in(train_data_file.c_str(), ios::binary);
    ifstream label_file_in(train_label_file.c_str(), ios::binary);
    data_file_in.ignore(2 * sizeof(float));
    label_file_in.ignore(2 * sizeof(float));
    cout << "Epoch #" << epoch << endl;
    for (int mb = 0; mb < num_mb_per_epoch; ++mb) {
      bool print_error = !(mb % 20);
      Train(gpu_devices[mb % 2], data_file_in, label_file_in, print_error);
    }
    data_file_in.close();
    label_file_in.close();
  }
  ms.current_device_id_ = cpu_device;

  // output weights
  cout << "Write weight to files" << endl;
  weights.clear();
  bias.clear();
  acts.clear();
  sens.clear();
  ms.dag_scheduler().GCNodes();
  cout << ms.device_manager().GetDevice(gpu_devices[0])->GetMemUsage() << endl;
  cout << ms.device_manager().GetDevice(gpu_devices[1])->GetMemUsage() << endl;
  cout << ms.device_manager().GetDevice(cpu_device)->GetMemUsage() << endl;
  cout << ms.physical_dag().PrintDag<ExternRCPrinter>() << endl;
  cout << "Training finished." << endl;
  return 0;
}
