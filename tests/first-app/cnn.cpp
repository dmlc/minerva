/**
* Function explanation:
* NArray NArray::Tile(NArray m, vector<size_t> reptimes);
*     // This function tiles the original matrix m by the times given on each dimension.
*     // For example:
*     // If m is a 3x5 matrix, m1 = NArray::Tile(m, {1, 2}); then m1 is a 3x10 matrix.
*
* NArray NArray::Randn(vector<size_t> dims, float mu, float var);
*     // This function generates a matrix of given dims. The value is filled by a normal distribution
*     // of parameter (mu, var)
*
* NArray Elewise::XXX(NArray m);
*     // All such functions are element-wise functions. The name is just its functionality, e.g. Exp, Ln, ...
*
* NArray Reduction::Max(NArray m, size_t dim);
* NArray Reduction::Sum(NArray m, size_t dim);
* NArray Reduction::Sum(NArray m, Vector dims);
*     // Those functions perform aggregation on the given dimensions.
*     // For example:
*     // If m is a 3x5 matrix, m1 = Reduction::Max(m, 0); then m1 is a 1x5 matrix.
*	  // If m is a 3x5x2 matrix, m1 = Reduction::Max(m, {0, 2}); then m1 is a 1x5x1 matrix.
*
* NArray Reduction::MaxIndex(NArray m, size_t dim);
*     // Similar to the reduction functions above, but the returned function contains the indices of the max values.
*
* NArray Reshape(NArray m, Vector dims);
*	  // This operation is to convert the shape of a matrix as long as the element size keeps unchanged
*	  // e.g, m is a 6*6 matrix, m1 = NArray::Reshape(m, {4, 9}), then m1 is a 4*9 matrix
*	  // Reshape can even change a m-dim matrix to a n-dim matrix, e.g, m is a 10*5*2*10 matrix,
*	  // m1 = NArray::Reshape(m, {10*5*2, 10}), then m1 is a 100*10 matrix, this operation will be used to 
*	  // convert a convolution matrix to a fully-connected matrix
*
* NArray Convolution::Conv(NArray a, NArray b, Connection conn);
*	  // This operation is to conduct convolution on NArray a using b, the Connection struct hold the detail
*	  // parameter of the convolution operation: receptive field size, stride, padding size
*
* NArray Convolution::GetGrad(NArray sen, NArray act, Connection conn);
*	  // In the bp algorithm, we get the gradient of a weight using the upper layer's error direvative(or sensitivity) and the lower layer's activation
*	  // of a certain weight. In convolution, this operation is conducted many times at different receptive field, then add up the gradient at different
* 	  // receptive field to serve as the gradient of the convolution weight. We note this operation as GetGrad
*/

#include <vector>
#include <cmath>
#include <iostream>

using namespace std;

//Renew params
M_FLOAT epsW = 0.001;
M_FLOAT epsB = 0.001;

int num_epochs = 100;

//dataset description
size_t num_train_samples = 60000;
size_t minibatch_size = 256;
size_t num_minibatches = 235;

/* The Vector class encapsulate a std::vector<int>. It provides some utilities to process the vector
 * Member functions:
 *  int Prod();
 * 		// Return the product of all elements in the vector
 *
 *  int operator[](size_t idx);
 *		// Element accessor
 *
 * Functions:
 *  Vector Vector::Concat(Vector s1, Vector s2);
 *		// Concat two scale vectors into one
 */
class Vector;

struct ConvInfo {
  int num_filters;
  Vector filtersize, stride, paddingsize;
};

//layer
Vector layer0 = {28, 28, 1};
Vector layer1 = {28, 28, 16};
Vector layer2 = {10};
ConvInfo convinfo{
  16, {5, 5, 1}, {1, 1, 1}, {2, 2, 0}
};
NArray w1, w2;
NArray b1 = NArray::Constant({1, 1, 16, 1}, 0.0);
NArray b2 = NArray::Constant({10, 1}, 0.0);

void InitNetwork() {
  // init convolution weight
  int row1 = convinfo.num_filters, col1 = convinfo.filtersize.Prod();
  float var1 = sqrt(4.0 / (row1 + col1));
  Vector w1dim = Vector::Concat(convinfo.num_filters, convinfo.filtersize);
  w1 = NArray::Randn(w1dim, 0.0, var1);
  // init fully weight
  int row2 = layer2.Prod(), col2 = layer1.Prod();
  float var2 = sqrt(4.0 / (row2 + col2));
  w2 = NArray::Randn({row2, col2}, 0.0, var2);
}

NArray Softmax(NArray m) {
  NArray maxval = Reduction::Max(m, 0);
  NArray centered = m - maxval.RepMat({m.Size(0), 1});
  NArray class_normalizer = Elewise::Ln(Reduction::Sum(Elewise::Exp(centered), 0)) + maxval;
  return Elewise::Exp(m - class_normalizer.RepMat({m.Size(0), 1}));
}

void PrintTrainingAccuracy(NArray o, NArray t) {
  //get predict
  NArray predict = Reduction::MaxIndex(o, 0);
  //get groundtruth
  NArray groundtruth = Reduction::MaxIndex(t, 0);
  float correct = Reduction::CountZero(predict - groundtruth).Get();
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
      NArray data = loader.GetData();

      size_t realmbsize = data.Size(1);
      Vector b1_tile_times = {layer1[0], layer1[1], 1, realmbsize};
      Vector b2_tile_times = {1, realmbsize};
      Vector l0_data_size = Vector::Concat(layer0, realmbsize);
      Vector l1_data_size_conv = Vector::Concat(layer1, realmbsize);
      Vector l1_data_size_fully = {layer1.Prod(), realmbsize};
      Vector l2_data_size = Vector::Concat(layer2, realmbsize);

      NArray a0, a1, a2, s1, s2;
      a0 = data.Reshape(l0_data_size);

      // FF-conv
      NArray y1 = Convolution::ConvFF(w1, a0, convinfo) + b1.Tile(b1_tile_times);
      a1 = Elewise::Sigmoid(y1);
      // FF-fully
      NArray y2 = w2 * a1.Reshape(l1_data_size_fully) + b2.Tile(b2_tile_times);
      a2 = Elewise::Sigmoid(y2);
      // Error
      NArray target = loader.GetLabel();
      NArray predict = Softmax(a2);
      PrintTrainingAccuracy(predict, target);
      s2 = target - predict;
      // BP-fully
      s1 = w2.Trans() * s2;
      s1 = Elewise::Mult(s1, 1- s1);
      // NO BP-conv
      // Update bias
      b2 -= epsB * Reduction::Sum(s2, 1) / realmbsize;
      b1 -= epsB * Reduction::Sum(s1.Reshape(l1_data_size_conv), {0, 1, 3}) / realmbsize;
      // Update weight
      w2 -= epsW * s2 * a1.Reshape(l1_data_size_fully).Trans() / realmbsize;
      w1 -= epsW * Convolution::GetGrad(s1.Reshape(l1_data_size_conv), a0, convinfo) / realmbsize;
    }
  }
}

int main(int argc, char** argv) {
  InitNetwork();
  TrainNetwork();
  return 0;
}
