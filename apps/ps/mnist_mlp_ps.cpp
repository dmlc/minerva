#include <glog/logging.h>
#include <minerva.h>
#include <fstream>
#include "ps.h"

using namespace std;
using namespace minerva;

#define LL LOG(ERROR)

template <typename V>
inline std::string arrstr(const V* data, int n) {
  std::stringstream ss;
  ss << "[" << n << "]: ";
  for (int i = 0; i < n; ++i) ss << data[i] << " ";
  return ss.str();
}

template<typename T>
inline std::ostream & operator << (std::ostream & os, const std::vector<T> & v)
{
  os << "[" << v.size() << "]: ";
  for (size_t i = 0; i < v.size(); ++i) os << v[i] << " ";
  return os;
}

bool StartsWith(const std::string & str, const std::string & pattern)
{
  return str.size() >= pattern.size() && str.substr(0, pattern.size()) == pattern;
}

const float epsW = 0.01, epsB = 0.01;
const int numepochs = 100;
const int mb_size = 256;
const int num_mb_per_epoch = 235;

const string weight_init_files[] = { "w12.dat", "w23.dat" };
const string weight_out_files[] = { "w12.dat", "w23.dat" };
const string bias_out_files[] = { "b2_trained.dat", "b3_trained.dat" };
const string train_data_file = "/home/hct/mnist/traindata.dat";
const string train_label_file = "/home/hct/mnist/trainlabel.dat";
const string test_data_file = "/home/hct/mnist/testdata.dat";
const string test_label_file = "/home/hct/mnist/testlabel.dat";

const int num_layers = 3;
const int lsize[num_layers] = { 784, 256, 10 };
vector<NArray> weights;
vector<NArray> bias;

string GetWeightName(int i) {
  return string("weights_") + to_string(i);
}

string GetBiasName(int i) {
  return string("bias_") + to_string(i);
}

bool IsWeightName(const std::string & name)
{
  return StartsWith(name, "weights_");
}

bool IsBiasName(const std::string & name)
{
  return StartsWith(name, "bias_");
}

void DumpParams(int epoch)
{
  //FileFormat format;
  //format.binary = false;
  //for (int i = 0; i < num_layers - 1; ++i)
  //  weights[i].ToFile(string("w_") + to_string(i + 1) + to_string(i + 2) + "_" + to_string(epoch), format);
  //for (int i = 0; i < num_layers - 1; ++i)
  //  bias[i].ToFile(string("b_") + to_string(i + 1) + "_" + to_string(epoch), format);
}

void GenerateInitWeight() {
  for (int i = 0; i < num_layers - 1; ++i)
  {
    weights[i] = NArray::Randn({ lsize[i + 1], lsize[i] }, 0.0, sqrt(4.0 / (lsize[0] + lsize[1])));
    bias[i] = NArray::Constant({ lsize[i + 1], 1 }, 1.0);
  }
}

NArray Softmax(NArray m) {
  NArray maxval = m.Max(0);
  // NArray centered = m - maxval.Tile({m.Size(0), 1});
  NArray centered = m.NormArithmetic(maxval, ArithmeticType::kSub);
  NArray class_normalizer = Elewise::Ln(Elewise::Exp(centered).Sum(0)) + maxval;
  // return Elewise::Exp(m - class_normalizer.Tile({m.Size(0), 1}));
  return Elewise::Exp(m.NormArithmetic(class_normalizer, ArithmeticType::kSub));
}

void PrintTrainingAccuracy(NArray o, NArray t) {
  //get predict
  NArray predict = o.MaxIndex(0);
  //get groundtruth
  NArray groundtruth = t.MaxIndex(0);

  float correct = (predict - groundtruth).CountZero();
  LL << "Training Error: " << (mb_size - correct) / mb_size << endl;
}



//============================================================
// here is function required by PS
#include <chrono>
#include <random>
#include <algorithm>

void InitLayer(const std::string &name, float* weight, size_t size) {
  if (IsBiasName(name)) {
    // fill with 1.0
    std::fill(weight, weight + size, 1.0);
  }
  else {
    // weight are filled with random
    default_random_engine generator(chrono::system_clock::now().time_since_epoch().count());
    normal_distribution<float> distribution(0.0, sqrt(4.0 / (lsize[0] + lsize[1])));
    for (size_t i = 0; i < size; ++i) {
      weight[i] = distribution(generator);
    }
  }  
}

void UpdateLayer(const std::string &name, float* weight, float* gradient, size_t size) {
  // weight -= eta * gradient
  float eta = .1;
  if (IsWeightName(name)) {
    eta = epsW;
  }
  else {
    eta = epsB;
  }

  for (size_t i = 0; i < size; ++i) {
    weight[i] -= eta * gradient[i];
  }
}

int MinervaWorkerMain(int rank, int size, int argc, char *argv[]) {
  LL << "this is worker " << rank << " of " << size;

  MinervaSystem& ms = MinervaSystem::Instance();
  ms.Initialize(&argc, &argv);
  uint64_t cpuDevice = ms.device_manager().CreateCpuDevice();
#ifdef HAS_CUDA
  uint64_t gpuDevice = ms.device_manager().CreateGpuDevice(0);
#endif
  ms.current_device_id_ = cpuDevice;

  weights.resize(num_layers - 1);
  bias.resize(num_layers - 1);
  GenerateInitWeight();
  for (int i = 0; i < num_layers - 1; ++i)
  {
    weights[i].Pull(GetWeightName(i));
    bias[i].Pull(GetBiasName(i));
  }
 
  cout << "Training procedure:" << endl;
  NArray acts[num_layers], sens[num_layers];
  for (int epoch = 0; epoch < numepochs; ++epoch) {
    cout << "  Epoch #" << epoch << endl;
    ifstream data_file_in(train_data_file.c_str());
    ifstream label_file_in(train_label_file.c_str());
    data_file_in.seekg(rank * lsize[0] * mb_size * sizeof(float));
    label_file_in.seekg(rank * lsize[num_layers - 1] * mb_size * sizeof(float));
    DumpParams(0);
    for (int mb = rank; mb < num_mb_per_epoch; mb+=size) {

      ms.current_device_id_ = cpuDevice;

      Scale data_size{ lsize[0], mb_size };
      Scale label_size{ lsize[num_layers - 1], mb_size };
      shared_ptr<float> data_ptr(new float[data_size.Prod()]);
      shared_ptr<float> label_ptr(new float[label_size.Prod()]);
      data_file_in.read(reinterpret_cast<char*>(data_ptr.get()), data_size.Prod() * sizeof(float));
      label_file_in.read(reinterpret_cast<char*>(label_ptr.get()), label_size.Prod() * sizeof(float));
      data_file_in.seekg((size - 1) * data_size.Prod() * sizeof(float), ios_base::cur);
      label_file_in.seekg((size - 1) * label_size.Prod() * sizeof(float), ios_base::cur);

      acts[0] = NArray::MakeNArray(data_size, data_ptr);
      NArray label = NArray::MakeNArray(label_size, label_ptr);

#ifdef HAS_CUDA
      ms.current_device_id_ = gpuDevice;
#endif

      // ff
      for (int k = 1; k < num_layers - 1; ++k) {
        NArray wacts = weights[k - 1] * acts[k - 1];
        NArray wactsnorm = wacts.NormArithmetic(bias[k - 1], ArithmeticType::kAdd);
        acts[k] = Elewise::SigmoidForward(wactsnorm);
      }
      // softmax
      acts[num_layers - 1] = Softmax((weights[num_layers - 2] * acts[num_layers - 2]).NormArithmetic(bias[num_layers - 2], ArithmeticType::kAdd));
      // bp
      sens[num_layers - 1] = acts[num_layers - 1] - label;
      for (int k = num_layers - 2; k >= 1; --k) {
        NArray d_act = Elewise::Mult(acts[k], 1 - acts[k]);
        sens[k] = weights[k].Trans() * sens[k + 1];
        sens[k] = Elewise::Mult(sens[k], d_act);
      }

      ms.current_device_id_ = cpuDevice;
      for (int k = 0; k < num_layers - 1; ++k) {
        bias[k] = NArray::PushGradAndPullWeight(sens[k + 1].Sum(1) / mb_size, GetBiasName(k));
        weights[k] = NArray::PushGradAndPullWeight(sens[k + 1] * acts[k].Trans() / mb_size, GetWeightName(k));
      }
      ms.current_device_id_ = gpuDevice;

      if ((mb - rank) % 20 == 0) {
        ms.current_device_id_ = cpuDevice;
        PrintTrainingAccuracy(acts[num_layers - 1], label);
      }
      DumpParams(mb + 1);
    }
    data_file_in.close();
    label_file_in.close();
  }
  ms.current_device_id_ = cpuDevice;

  // output weights
  cout << "Write weight to files" << endl;
  //FileFormat format;
  //format.binary = true;
  weights.clear();
  bias.clear();
  cout << "Training finished." << endl;

  return 0;
}

