#include <minerva.h>
#include <fstream>
#include <gflags/gflags.h>

DEFINE_bool(init, false, "Only generate init weights");

using namespace std;
using namespace minerva;

const float epsW = 0.001, epsB = 0.001;
const int numepochs = 1;
const int mb_size = 256;
const int num_mb_per_epoch = 1;//235;

const string weight_init_files[] = { "w12_init.dat", "w23_init.dat", };
const string weight_out_files[] = { "w12_trained.dat", "w23_trained.dat", };
const string bias_out_files[] = { "b2_trained.dat", "b3_trained.dat" };
const string train_data_file = "data/mnist/traindata.dat";
const string train_label_file = "data/mnist/trainlabel.dat";
const string test_data_file = "data/mnist/testdata.dat";
const string test_label_file = "data/mnist/testlabel.dat";

const int l1 = 784, l2 = 256, l3 = 10;
NArray w12, w23, b2, b3;

void GenerateInitWeight() {
  w12 = NArray::Randn({l2, l1}, 0.0, sqrt(4.0 / (l1 + l2)), {1, 1});
  w23 = NArray::Randn({l3, l2}, 0.0, sqrt(4.0 / (l2 + l3)), {1, 1});
  FileFormat format;
  format.binary = true;
  w12.ToFile(weight_init_files[0], format);
  w23.ToFile(weight_init_files[1], format);
}

void InitWeight() {
  SimpleFileLoader* loader = new SimpleFileLoader;
  w12 = NArray::LoadFromFile({l2, l1}, weight_init_files[0], loader, {1, 1});
  w23 = NArray::LoadFromFile({l3, l2}, weight_init_files[1], loader, {1, 1});
  b2 = NArray::Constant({l2, 1}, 0.0, {1, 1});
  b3 = NArray::Constant({l3, 1}, 0.0, {1, 1});
}

NArray Softmax(NArray a) {
  // TODO
  return a;
}

int main(int argc, char** argv) {
  MinervaSystem::Instance().Initialize(&argc, &argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if(FLAGS_init) {
    cout << "Generate initial weights" << endl;
    GenerateInitWeight();
    cout << "Finished!" << endl;
    return 0;
  } else {
    cout << "Init weights and bias" << endl;
    InitWeight();
  }

  cout << "Training procedure:" << endl;
  OneFileMBLoader train_data_loader(train_data_file, {l1});
  OneFileMBLoader train_label_loader(train_label_file, {l3});
  for(int epoch = 0; epoch < numepochs; ++epoch) {
    cout << "  Epoch #" << epoch << endl;
    for(int mb = 0; mb < num_mb_per_epoch; ++mb) {
      NArray data = train_data_loader.LoadNext(mb_size);
      NArray label = train_label_loader.LoadNext(mb_size);
      // ff
      NArray a1 = data;
      NArray a2 = Elewise::Sigmoid((w12 * a1).NormArithmetic(b2, ADD));
      NArray a3 = Elewise::Sigmoid((w23 * a2).NormArithmetic(b3, ADD));
      // softmax
      a3 = Softmax(a3);
      // bp
      NArray s3 = a3 - label;
      NArray s2 = w23.Trans() * s3;
      s2 = Elewise::Mult(s2, 1 - s2);
      // gradient
      NArray gw12 = s2 * a1.Trans() / mb_size;
      NArray gb2 = s2.Sum(1) / mb_size;
      NArray gw23 = s3 * a2.Trans() / mb_size;
      NArray gb3 = s3.Sum(1) / mb_size;
      // update
      w12 -= epsW * gw12;
      w23 -= epsW * gw23;
      b2 -= epsB * gb2;
      b3 -= epsB * gb3;
    }
  }
  // [optional] output logical dag file
  ofstream ldagfout("mnist_ldag.txt");
  ldagfout << MinervaSystem::Instance().logical_dag().PrintDag() << endl;
  ldagfout.close();
  // output weights
  cout << "Write weight to files" << endl;
  FileFormat format;
  format.binary = true;
  w12.ToFile(weight_out_files[0], format);
  w23.ToFile(weight_out_files[1], format);

  cout << "Training finished." << endl;
  return 0;
}
