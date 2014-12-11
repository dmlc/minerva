#include <minerva.h>
#include <fstream>

using namespace std;
using namespace minerva;

const float epsW = 0.01, epsB = 0.01;
const int numepochs = 100;
const int mb_size = 256;
const int num_mb_per_epoch = 235;

const string weight_init_files[] = { "w12.dat", "w23.dat" };
const string weight_out_files[] = { "w12.dat", "w23.dat" };
const string bias_out_files[] = { "b2_trained.dat", "b3_trained.dat" };
const string train_data_file = "/home/cs_user/data/mnist/traindata.dat";
const string train_label_file = "/home/cs_user/data/mnist/trainlabel.dat";
const string test_data_file = "/home/cs_user/data/mnist/testdata.dat";
const string test_label_file = "/home/cs_user/data/mnist/testlabel.dat";

const int num_layers = 3;
const int lsize[num_layers] = {784, 256, 10};
vector<NArray> weights;
vector<NArray> bias;

void GenerateInitWeight() {
  for (int i = 0; i < num_layers - 1; ++ i)
  {
    weights[i] = NArray::Randn({lsize[i + 1], lsize[i]}, 0.0, sqrt(4.0 / (lsize[0] + lsize[1])));
    bias[i] = NArray::Constant({lsize[i + 1], 1}, 1.0);
  }
  FileFormat format;
  format.binary = true;
  for (int i = 0; i < num_layers - 1; ++ i)
    weights[i].ToFile(weight_init_files[i], format);
}

void InitWeight() {
  shared_ptr<SimpleFileLoader> loader(new SimpleFileLoader());
  for (int i = 0; i < num_layers - 1; ++ i) {
    weights[i] = NArray::LoadFromFile({lsize[i + 1], lsize[i]}, weight_init_files[i], loader);
    bias[i] = NArray::Constant({lsize[i + 1], 1}, 1.0);
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
  cout << "Training Error: " << (mb_size - correct) / mb_size << endl;
}

int main(int argc, char** argv) {
  MinervaSystem& ms = MinervaSystem::Instance();
  ms.Initialize(&argc, &argv);
  uint64_t cpuDevice = ms.CreateCpuDevice();
  uint64_t gpuDevice = ms.CreateGpuDevice(0);
  ms.current_device_id_ = cpuDevice;

  weights.resize(num_layers - 1);
  bias.resize(num_layers - 1);
  GenerateInitWeight();
/*  if(FLAGS_init) {
    cout << "Generate initial weights" << endl;
    GenerateInitWeight();
    cout << "Finished!" << endl;
    return 0;
  } else {
    cout << "Init weights and bias" << endl;
    InitWeight();
  }
*/
  cout << "Training procedure:" << endl;
  NArray acts[num_layers], sens[num_layers];
  for(int epoch = 0; epoch < numepochs; ++ epoch) {
    cout << "  Epoch #" << epoch << endl;
    ifstream data_file_in(train_data_file.c_str());
    ifstream label_file_in(train_label_file.c_str());
    for(int mb = 0; mb < num_mb_per_epoch; ++ mb) {

      ms.current_device_id_ = cpuDevice;

      Scale data_size{lsize[0], mb_size};
      Scale label_size{lsize[num_layers - 1], mb_size};
      shared_ptr<float> data_ptr( new float[data_size.Prod()] );
      shared_ptr<float> label_ptr( new float[label_size.Prod()] );
      data_file_in.read(reinterpret_cast<char*>(data_ptr.get()), data_size.Prod() * sizeof(float));
      label_file_in.read(reinterpret_cast<char*>(label_ptr.get()), label_size.Prod() * sizeof(float));

      acts[0] = NArray::MakeNArray(data_size, data_ptr);
      NArray label = NArray::MakeNArray(label_size, label_ptr);

      ms.current_device_id_ = gpuDevice;

      // ff
      for (int k = 1; k < num_layers - 1; ++ k) {
        NArray wacts = weights[k - 1] * acts[k - 1];
        NArray wactsnorm = wacts.NormArithmetic(bias[k - 1], ArithmeticType::kAdd);
        acts[k] = Elewise::SigmoidForward(wactsnorm);
      }
      // softmax
      acts[num_layers - 1] = Softmax((weights[num_layers - 2] * acts[num_layers - 2]).NormArithmetic(bias[num_layers - 2], ArithmeticType::kAdd));
      // bp
      sens[num_layers - 1] = acts[num_layers - 1] - label;
      for (int k = num_layers - 2; k >= 1; -- k) {
        NArray d_act = Elewise::Mult(acts[k], 1 - acts[k]);
        sens[k] = weights[k].Trans() * sens[k + 1];
        sens[k] = Elewise::Mult(sens[k], d_act);
      }

      // Update bias
      for(int k = 0; k < num_layers - 1; ++ k) { // no input layer
        bias[k] -= epsB * sens[k + 1].Sum(1) / mb_size;
      }
      // Update weight
      for(int k = 0; k < num_layers - 1; ++ k) {
        weights[k] -= epsW * sens[k + 1] * acts[k].Trans() / mb_size;
      }

      if (mb % 20 == 0) {
        ms.current_device_id_ = cpuDevice;
        PrintTrainingAccuracy(acts[num_layers - 1], label);
      }
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
