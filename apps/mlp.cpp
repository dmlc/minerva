/**
* Function explanation:
* NArray NArray::Tile(NArray m, vector<int> reptimes);
*     // This function repeat the original matrix m by the times given on each dimension.
*     // For example:
*     // If m is a 3x5 matrix, m1 = NArray::Tile(m, {1, 2}); then m1 is a 3x10 matrix.
*
* NArray NArray::NormRandom(vector<int> dims, float mu, float var);
*     // This function generates a matrix of given dims. The value is filled by a normal distribution
*     // of parameter (mu, var)
*
* NArray Elewise::XXX(NArray m);
*     // All such functions are element-wise functions. The name is just its functionality, e.g. Exp, Ln, ...
*
* NArray Reduction::Max(NArray m, int dim);
* NArray Reduction::Sum(NArray m, int dim);
*     // Those functions perform aggregation on the given dimension.
*     // For example:
*     // If m is a 3x5 matrix, m1 = Reduction::Max(m, 0); then m1 is a 1x5 matrix.
*
* NArray Reduction::MaxIndex(NArray m, int dim);
*     // Similar to the reduction functions above, but the returned function contains the indices of the max values.
*/

#include <vector>
#include <cmath>
#include <iostream>

#include "minerva.h"

using namespace std;
using namespace minerva;

//Renew params
float epsW = 0.001;
float epsB = 0.001;

int num_epochs = 100;

//dataset description
int num_train_samples = 60000;
int minibatch_size = 256;
int num_minibatches = 235;

uint64_t cpuDevice;
uint64_t gpuDevice;

#if 1

//Fully layer
struct Layer {
  int length;
  NArray bias;
  Layer(int len): length(len) {
    bias = NArray::Constant({ length, 1 }, 0.0, {1, 1});
  }
};

vector<Layer> layers;
vector<NArray> weights;
int num_layers;

void InitNetwork() {
  layers = {
    Layer(28 * 28), Layer(256), Layer(10)
  };
  num_layers = layers.size();

  for(int i = 0; i < num_layers - 1; ++i) {
    int row = layers[i + 1].length;
    int col = layers[i].length;
    float var = sqrt(4.0 / (row + col));
    weights.push_back(NArray::Randn({ row, col }, 0.0, var, {1, 1}));
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
  int correct = (predict - groundtruth).CountZero();
  cout << "Training Error: " << (minibatch_size - correct) / minibatch_size << endl;
}

//give initial value to the weight in the DNN
void TrainNetwork() {
  ////////////////////////////////TRAIN NETWORK/////////////////////////////////////
  //load data
  const string train_data_file = "/home/data/data/mnist/train_small/traindata_0";
  const string train_label_file = "/home/data/data/mnist/train_small/trainlabel_0";
  const string test_data_file = "/home/data/data/mnist/test/testdata_0";
  const string test_label_file = "/home/data/data/mnist/test/testlabel_0";

  OneFileMBLoader train_data_loader(train_data_file, {layers.front().length});
  OneFileMBLoader train_label_loader(train_label_file, {layers.back().length});

  for(int i = 0; i < num_epochs; i++) {
    for(int j = 0; j < num_minibatches; j++) {

      vector<NArray> acts, sens;
      acts.resize(num_layers);
      sens.resize(num_layers);

      MinervaSystem & ms = MinervaSystem::Instance();
      ms.set_device_id(cpuDevice);

      acts[0] = train_data_loader.LoadNext(minibatch_size);
      NArray target = train_label_loader.LoadNext(minibatch_size);

      ms.set_device_id(gpuDevice);
      // FF
      for(int k = 1; k < num_layers; ++k) {
        acts[k] = (weights[k - 1] * acts[k - 1]).NormArithmetic(layers[k].bias, ADD);
        acts[k] = Elewise::Sigmoid(acts[k]);
      }
      // Error
      acts[num_layers-1] = Softmax(acts[num_layers-1]);
      

      PrintTrainingAccuracy(acts[num_layers-1], target);
      sens[num_layers-1] = target - acts[num_layers-1];
      // BP
      for(int k = num_layers - 2; k > 0; --k) {
        sens[k] = weights[k].Trans() * sens[k+1];
        sens[k] = Elewise::Mult(sens[k], 1 - sens[k]);
      }
      // Update bias
      for(int k = 1; k < num_layers; ++k) { // no input layer
        layers[k].bias -= epsB * sens[k].Sum(1) / minibatch_size;
      }
      // Update weight
      for(int k = 0; k < num_layers - 1; ++k) {
        weights[k] -= epsW * sens[k+1] * acts[k].Trans() / minibatch_size;
      }
    }
  }
}

#endif

int main(int argc, char** argv) {
  MinervaSystem & ms = MinervaSystem::Instance();
  ms.Initialize(&argc, &argv);
  cpuDevice = ms.CreateCPUDevice();
  gpuDevice = ms.CreateGPUDevice(0);
  
  
  InitNetwork();
  TrainNetwork();
}
