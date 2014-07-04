/**
* Function explanation:
* NArray NArray::Tile(NArray m, vector<size_t> reptimes);
*     // This function repeat the original matrix m by the times given on each dimension.
*     // For example:
*     // If m is a 3x5 matrix, m1 = NArray::Tile(m, {1, 2}); then m1 is a 3x10 matrix.
*
* NArray NArray::NormRandom(vector<size_t> dims, float mu, float var);
*     // This function generates a matrix of given dims. The value is filled by a normal distribution
*     // of parameter (mu, var)
*
* NArray Elewise::XXX(NArray m);
*     // All such functions are element-wise functions. The name is just its functionality, e.g. Exp, Ln, ...
*
* NArray Reduction::Max(NArray m, size_t dim);
* NArray Reduction::Sum(NArray m, size_t dim);
*     // Those functions perform aggregation on the given dimension.
*     // For example:
*     // If m is a 3x5 matrix, m1 = Reduction::Max(m, 0); then m1 is a 1x5 matrix.
*
* NArray Reduction::MaxIndex(NArray m, size_t dim);
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
size_t num_train_samples = 60000;
size_t minibatch_size = 256;
size_t num_minibatches = 235;

//Fully layer
struct Layer {
  size_t length;
  NArray bias;
  Layer(size_t len): length(len) {
    bias = NArray::Constant({length, 1}, 0.0);
  }
};
vector<Layer> layers = {
  Layer(28*28), Layer(256), Layer(10)
};
vector<NArray> weights;
size_t num_layers = layers.size();

void InitNetwork() {
  for(size_t i = 0; i < num_layers - 1; ++i) {
    size_t row = layers[i + 1].length;
    size_t col = layers[i].length;
    float var = sqrt(4.0 / (row + col));
    weights.push_back(NArray::Randn({row, col}, 0.0, var));
  }
}

NArray Softmax(NArray m) {
  NArray maxval = m.Max(0);
  NArray centered = m - maxval.Tile({m.Size(0), 1});
  NArray class_normalizer = Elewise::Ln(Elewise::Exp(centered).Sum(0)) + maxval;
  return Elewise::Exp(m - class_normalizer.Tile({m.Size(0), 1}));
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
  DBLoader loader("./data/mnist/");

  for(int i = 0; i < num_epochs; i++) {
    for(size_t j = 0; j < num_minibatches; j++) {
      loader.LoadNext(minibatch_size);

      vector<NArray> acts, sens;
      acts.resize(num_layers);
      sens.resize(num_layers)

      acts[0] = loader.GetData();
      // FF
      for(size_t k = 1; k < num_layers; ++k) {
        acts[k] = weights[k] * acts[k-1] + layers[k].bias.Tile({1, num_minibatches});
        acts[k] = Elewise::Sigmoid(acts[k]);
      }
      // Error
      acts[num_layers-1] = Softmax(acts[num_layers-1]);
      NArray target = loader.GetLabel();
      PrintTrainingAccuracy(acts[num_layers-1], target);
      sens[num_layers-1] = target - acts[num_layers-1];
      // BP
      for(size_t k = num_layers - 2; k > 0; --k) {
        sens[k] = weights[k].Trans() * sens[k+1];
        sens[k] = Elewise::Mult(sens[k], 1 - sens[k]);
      }
      // Update bias
      for(size_t k = 1; k < num_layers; ++k) { // no input layer
        layers[k].bias -= epsB * sens[k].Sum(1) / minibatch_size;
      }
      // Update weight
      for(size_t k = 0; k < num_layers - 1; ++k) {
        weights[k] -= epsW * sens[k+1] * acts[k].Trans() / minibatch_size;
      }
    }
  }
}

int main(int argc, char** argv) {
  InitNetwork();
  TrainNetwork();
}
