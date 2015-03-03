#include <minerva.h>
#include <fstream>
#include <gflags/gflags.h>

using namespace std;
using namespace minerva;

DEFINE_bool(init, false, "Initialize weights");
DEFINE_int32(mb, 256, "Minibatch size");

const int numepochs = 10;
int mb_size = 256;
float alpha = 0.01;

const string train_data_file = "/home/cs_user/data/mnist/traindata.dat";
const string train_label_file = "/home/cs_user/data/mnist/trainlabel.dat";

vector<NArray> weights;
vector<NArray> bias;

void PrintTrainingAccuracy(NArray o, NArray t) {
  NArray predict = o.Reshape({10, mb_size}).MaxIndex(0);
  NArray groundtruth = t.Reshape({10, mb_size}).MaxIndex(0);
  float correct = (predict - groundtruth).CountZero();
  cout << "Training Error: " << (mb_size - correct) / mb_size << endl;
}

void InitWeights() {
  if (FLAGS_init) {
    cout << "Initialize weights" << endl;
    weights[0] = Filter(NArray::Randn({5, 5, 1, 16}, 0.0, 0.1));
    bias[0] = NArray::Randn({16}, 0.0, 0.1);
    weights[1] = Filter(NArray::Randn({5, 5, 16, 32}, 0.0, 0.1));
    bias[1] = NArray::Randn({32}, 0.0, 0.1);
    weights[2] = NArray::Randn({10, 512}, 0.0, 0.1);
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
    cout << "Read existing weights" << endl;
    ifstream cnn_init("cnn_init", ios::binary);
    Scale weights_size[] = {{5, 5, 1, 16}, {5, 5, 16, 32}, {10, 512}};
    Scale bias_size[] = {{16}, {32}, {10, 1}};
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
}

vector<NArray> TrainMB(ifstream& data_file_in, ifstream& label_file_in, bool print) {
  Scale data_size{28, 28, 1, mb_size};
  Scale label_size{10, 1, 1, mb_size};
  ConvInfo conv_info[2];
  conv_info[0].pad_height = conv_info[0].pad_width = 0;
  conv_info[0].stride_vertical = conv_info[0].stride_horizontal = 1;
  conv_info[1].pad_height = conv_info[1].pad_width = 2;
  conv_info[1].stride_vertical = conv_info[1].stride_horizontal = 1;
  PoolingInfo pool_info[2];
  pool_info[0].algorithm = PoolingInfo::Algorithm::kMax;
  pool_info[0].height = pool_info[0].width = 2;
  pool_info[0].stride_vertical = pool_info[0].stride_horizontal = 2;
  pool_info[1].algorithm = PoolingInfo::Algorithm::kMax;
  pool_info[1].height = pool_info[1].width = 3;
  pool_info[1].stride_vertical = pool_info[1].stride_horizontal = 3;

  NArray acts[9], sens[9], label;
  shared_ptr<float> data_ptr(new float[data_size.Prod()], [](float* ptr) { delete[] ptr; });
  shared_ptr<float> label_ptr(new float[label_size.Prod()], [](float* ptr) { delete[] ptr; });
  data_file_in.read(reinterpret_cast<char*>(data_ptr.get()), data_size.Prod() * sizeof(float));
  label_file_in.read(reinterpret_cast<char*>(label_ptr.get()), label_size.Prod() * sizeof(float));
  acts[0] = NArray::MakeNArray(data_size, data_ptr);
  label = NArray::MakeNArray(label_size, label_ptr);

  acts[1] = Convolution::ConvForward(acts[0], weights[0], bias[0], conv_info[0]);
  acts[2] = Convolution::ActivationForward(acts[1], ActivationAlgorithm::kRelu);
  acts[3] = Convolution::PoolingForward(acts[2], pool_info[0]);
  acts[4] = Convolution::ConvForward(acts[3], weights[1], bias[1], conv_info[1]);
  acts[5] = Convolution::ActivationForward(acts[4], ActivationAlgorithm::kRelu);
  acts[6] = Convolution::PoolingForward(acts[5], pool_info[1]);
  auto re_acts6 = acts[6].Reshape({acts[6].Size().Prod() / mb_size, mb_size});
  acts[7] = (weights[2] * re_acts6).NormArithmetic(bias[2], ArithmeticType::kAdd);
  acts[8] = Convolution::SoftmaxForward(acts[7].Reshape({10, 1, 1, mb_size}), SoftmaxAlgorithm::kInstance);

  sens[8] = acts[8] - label;

  // sens[7] = Convolution::SoftmaxBackward(sens[8], acts[8], SoftmaxAlgorithm::kInstance).Reshape({10, mb_size});
  sens[7] = sens[8].Reshape({10, mb_size});
  sens[6] = (weights[2].Trans() * sens[7]).Reshape(acts[6].Size());
  sens[5] = Convolution::PoolingBackward(sens[6], acts[6], acts[5], pool_info[1]);
  sens[4] = Convolution::ActivationBackward(sens[5], acts[5], acts[4], ActivationAlgorithm::kRelu);
  sens[3] = Convolution::ConvBackwardData(sens[4], acts[3], weights[1], conv_info[1]);
  sens[2] = Convolution::PoolingBackward(sens[3], acts[3], acts[2], pool_info[0]);
  sens[1] = Convolution::ActivationBackward(sens[2], acts[2], acts[1], ActivationAlgorithm::kRelu);

  if (print) {
    acts[8].StartEval();
    // PrintTrainingAccuracy(acts[8], label);
  }

  vector<NArray> ret;
  ret.push_back(Convolution::ConvBackwardFilter(sens[1], weights[0], acts[0], conv_info[0]));
  ret.push_back(Convolution::ConvBackwardBias(sens[1]));
  ret.push_back(Convolution::ConvBackwardFilter(sens[4], weights[0], acts[3], conv_info[1]));
  ret.push_back(Convolution::ConvBackwardBias(sens[4]));
  ret.push_back(sens[7] * re_acts6.Trans());
  ret.push_back(sens[7].Sum(1));
  return ret;
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = 1;
  MinervaSystem::Initialize(&argc, &argv);
  MinervaSystem& ms = MinervaSystem::Instance();
  uint64_t cpu_device = ms.CreateCpuDevice();
  uint64_t gpu_device[2] = {ms.CreateGpuDevice(0), ms.CreateGpuDevice(1)};
  ms.current_device_id_ = gpu_device[0];
  if (0 < FLAGS_mb && FLAGS_mb <= 512) {
    mb_size = FLAGS_mb;
  }

  cout << "Minibatch size: " << mb_size << endl;
  weights.resize(3);
  bias.resize(3);
  InitWeights();

  cout << "Start training:" << endl;
  for (int epoch = 0; epoch < numepochs; ++epoch) {
    ifstream data_file_in(train_data_file.c_str(), ios::binary);
    ifstream label_file_in(train_label_file.c_str(), ios::binary);
    data_file_in.ignore(2 * sizeof(float));
    label_file_in.ignore(2 * sizeof(float));
    LOG(INFO) << "Epoch #" << epoch;
    for (int mb = 0; mb < 60000 / mb_size; ++mb) {
      vector<NArray> res1, res2;
      bool second = false;
      ms.current_device_id_ = gpu_device[0];
      res1 = TrainMB(data_file_in, label_file_in, mb % 40 == 0);
      if (++mb < 60000 / mb_size) {
        ms.current_device_id_ = gpu_device[1];
        res2 = TrainMB(data_file_in, label_file_in, mb % 40 == 0);
        second = true;
      }
      weights[0] -= alpha / mb_size / 2 * res1[0];
      bias[0] -= alpha / mb_size / 2 * res1[1];
      weights[1] -= alpha / mb_size / 2 * res1[2];
      bias[1] -= alpha / mb_size / 2 * res1[3];
      weights[2] -= alpha / mb_size / 2 * res1[4];
      bias[2] -= alpha / mb_size / 2 * res1[5];
      if (second) {
        weights[0] -= alpha / mb_size / 2 * res2[0];
        bias[0] -= alpha / mb_size / 2 * res2[1];
        weights[1] -= alpha / mb_size / 2 * res2[2];
        bias[1] -= alpha / mb_size / 2 * res2[3];
        weights[2] -= alpha / mb_size / 2 * res2[4];
        bias[2] -= alpha / mb_size / 2 * res2[5];
      }
    }
    weights[0].WaitForEval();
    data_file_in.close();
    label_file_in.close();
  }
  weights.clear();
  bias.clear();
  //ms.dag_scheduler().GCNodes();
  cout << ms.device_manager().GetDevice(gpu_device[0])->GetMemUsage() << endl;
  cout << ms.device_manager().GetDevice(gpu_device[1])->GetMemUsage() << endl;
  cout << ms.device_manager().GetDevice(cpu_device)->GetMemUsage() << endl;
  cout << ms.physical_dag().NumNodes() << endl;
  cout << "Training finished" << endl;
  return 0;
}
