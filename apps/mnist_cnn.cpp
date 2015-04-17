#include <minerva.h>
#include <fstream>
#include <gflags/gflags.h>
#include <iomanip>
#include "mnist_common.h"

int main(int argc, char** argv) {
  const auto& param = InitMnistApps(argc, argv);
  cout << param << endl;
  size_t data_length = 28 * 28 * param.mb_size; // img size = 28x28
  size_t label_length = 10 * param.mb_size; // 10 classes

  MnistCnnAlgo cnn_algo(param);
  cnn_algo.Init();

  ifstream data_file_in(param.train_data_file.c_str(), ios::binary);
  ifstream label_file_in(param.train_label_file.c_str(), ios::binary);

  cout << "Start training:" << endl;
  for (int epoch = 0; epoch < param.num_epochs; ++epoch) {
    data_file_in.ignore(2 * sizeof(int));
    label_file_in.ignore(2 * sizeof(int));
    cout << "Epoch #" << epoch << endl;
    for (int mb = 0; mb < param.num_mb; ++mb) {
      shared_ptr<float> data_ptr, label_ptr;
      tie(data_ptr, label_ptr) = GetNextBatch(data_file_in, label_file_in, data_length, label_length);
      NArray predict = cnn_algo.FF(data_ptr, false);
      NArray label = cnn_algo.BP(label_ptr);
      if (mb % 20 == 0) {
        PrintTrainingAccuracy(predict, label, param);
      }
      cnn_algo.Update();
    }
    data_file_in.clear();
    data_file_in.seekg(0, ios::beg);
    label_file_in.clear();
    label_file_in.seekg(0, ios::beg);
  }
  MinervaSystem::Finalize();
  cout << "Training finished" << endl;
  return 0;
}
