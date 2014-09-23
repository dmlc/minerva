#include <minerva.h>
#include <fstream>
#include <gflags/gflags.h>

DEFINE_bool(init, false, "Only generate init weights");

using namespace std;
using namespace minerva;

const float epsW = 0.01, epsB = 0.01;
const int numepochs = 200;
const int mb_size = 256;
const int num_mb_per_epoch = 235;

const string weight_init_files[] = { "w12.dat", "w23.dat", };
const string weight_out_files[] = { "w12.dat", "w23.dat", };
const string bias_out_files[] = { "b2_trained.dat", "b3_trained.dat" };
const string train_data_file = "/home/serailhydra/data/mnist/traindata.dat";
const string train_label_file = "/home/serailhydra/data/mnist/trainlabel.dat";
const string test_data_file = "/home/serailhydra/data/mnist/testdata.dat";
const string test_label_file = "/home/serailhydra/data/mnist/testlabel.dat";

const int num_layers = 3;
const int lsize[num_layers] = {784, 256, 10};
const int l1parts = 1, l2parts = 1, l3parts = 1;
const NVector<Scale> l1_part_shape = Scale{lsize[0]}.EquallySplit({l1parts});
const NVector<Scale> l2_part_shape = Scale{lsize[1]}.EquallySplit({l2parts});
const NVector<Scale> l3_part_shape = Scale{lsize[2]}.EquallySplit({l3parts});
NArray weights[num_layers - 1], bias[num_layers - 1];

void GenerateInitWeight() {
  for (int i = 0; i < num_layers - 1; ++ i)
  {
    weights[i] = NArray::Randn({lsize[i + 1], lsize[i]}, 0.0, sqrt(4.0 / (lsize[0] + lsize[1])), {1, 1});
    bias[i] = NArray::Constant({lsize[i + 1], 1}, 1.0, {1, 1});
  }
  FileFormat format;
  format.binary = true;
  for (int i = 0; i < num_layers - 1; ++ i)
    weights[i].ToFile(weight_init_files[i], format);
}

void InitWeight() {
  SimpleFileLoader* loader = new SimpleFileLoader;
  for (int i = 0; i < num_layers - 1; ++ i) {
    weights[i] = NArray::LoadFromFile({lsize[i + 1], lsize[i]}, weight_init_files[i], loader, {1, 1});
    bias[i] = NArray::Constant({lsize[i + 1], 1}, 1.0, {1, 1});
  }
}

NArray Softmax(NArray m) {
  NArray maxval = m.Max(0);
  // NArray centered = m - maxval.Tile({m.Size(0), 1});
  NArray centered = m.NormArithmetic(maxval, SUB);
  NArray class_normalizer = Elewise::Ln(Elewise::Exp(centered).Sum(0)) + maxval;
  // return Elewise::Exp(m - class_normalizer.Tile({m.Size(0), 1}));
  return Elewise::Exp(m.NormArithmetic(class_normalizer, SUB));
}

void PrintTrainingAccuracy(NArray o, NArray t) {
  //get predict
  NArray predict = o.MaxIndex(0);
  //get groundtruth
  NArray groundtruth = t.MaxIndex(0);;
  float correct = (predict - groundtruth).CountZero();
  cout << "Training Error: " << (mb_size - correct) / mb_size << endl;
}

void Print(NArray m) {
  //cout << MinervaSystem::Instance().logical_dag().PrintDag() << endl;
  float* ptr = m.Get();
  for(int i = 0; i < 10; ++i)
    cout << ptr[i] << " ";
  cout << endl;
  delete [] ptr;
}

int main(int argc, char** argv) {
  MinervaSystem::Instance().Initialize(&argc, &argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

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
  OneFileMBLoader train_data_loader(train_data_file, {lsize[0]});
  OneFileMBLoader train_label_loader(train_label_file, {lsize[num_layers - 1]});
  train_data_loader.set_partition_shapes_per_sample(l1_part_shape);
  train_label_loader.set_partition_shapes_per_sample(l3_part_shape);

  NArray acts[num_layers], sens[num_layers];
  for(int epoch = 0; epoch < numepochs; ++ epoch) {
    cout << "  Epoch #" << epoch << endl;
    for(int mb = 0; mb < num_mb_per_epoch; ++ mb) {
      NArray data = train_data_loader.LoadNext(mb_size);
      NArray label = train_label_loader.LoadNext(mb_size);
      // ff
      acts[0] = data;
      for (int k = 1; k < num_layers - 1; ++ k) {
        acts[k] = Elewise::Sigmoid((weights[k - 1] * acts[k - 1]).NormArithmetic(bias[k - 1], ADD));
      }
      // softmax
      acts[num_layers - 1] = Softmax((weights[num_layers - 2] * acts[num_layers - 2]).NormArithmetic(bias[num_layers - 2], ADD));
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
        PrintTrainingAccuracy(acts[num_layers - 1], label);
      }
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
  weights[0].ToFile(weight_out_files[0], format);
  weights[1].ToFile(weight_out_files[1], format);

  cout << "Training finished." << endl;
  return 0;
}
