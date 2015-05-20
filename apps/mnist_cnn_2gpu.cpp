#include <minerva.h>
#include <fstream>
#include <gflags/gflags.h>
#include <iomanip>
#include "mnist_common.h"

int main(int argc, char** argv) {
  auto param = InitMnistApps(argc, argv);
  param.num_gpus = 2;
  int num_gpu = param.num_gpus;
  MinervaSystem& ms = MinervaSystem::Instance();
  cout << param << endl;
  size_t train_data_len = 28 * 28 * param.mb_size / num_gpu; // img size = 28x28
  size_t train_label_len = 10 * param.mb_size / num_gpu; // 10 classes
  size_t test_data_len = 28 * 28 * param.num_tests; // img size = 28x28
  size_t test_label_len = 10 * param.num_tests; // 10 classes

  vector<uint64_t> gpus;
  for(int i = 0; i < num_gpu; ++i) {
    gpus.push_back(ms.CreateGpuDevice(i));
  }

  MnistMlpAlgo cnn_algo(param);
  cnn_algo.Init();

  ifstream train_data_in(param.train_data_file.c_str(), ios::binary);
  ifstream train_label_in(param.train_label_file.c_str(), ios::binary);
  ifstream test_data_in(param.test_data_file.c_str(), ios::binary);
  ifstream test_label_in(param.test_label_file.c_str(), ios::binary);

  cout << "Start training:" << endl;
  for (int epoch = 0; epoch < param.num_epochs; ++epoch) {
    train_data_in.clear();
    train_data_in.seekg(2 * sizeof(int), ios::beg);
    train_label_in.clear();
    train_label_in.seekg(2 * sizeof(int), ios::beg);
    cout << "Epoch #" << epoch << endl;
    for (int mb = 0; mb < param.num_mb; ++mb) {
      for (int i = 0; i < num_gpu; ++i) {
        ms.SetDevice(gpus[i]); // switch GPU
        shared_ptr<float> data_ptr, label_ptr;
        tie(data_ptr, label_ptr) = GetNextBatch(train_data_in, train_label_in, train_data_len, train_label_len);
        NArray predict = cnn_algo.FF(data_ptr, false);
        NArray label = cnn_algo.BP(label_ptr, false);
        if (mb % 20 == 0) {
          cout << "GPU #" << i << " ";
          PrintAccuracy(predict, label, param);
        }
      }
      ms.SetDevice(gpus[0]); // update is on GPU #0
      cnn_algo.Update();
    }
    // Testing
    ms.SetDevice(gpus[0]); // test is on GPU #0
    cout << "Testing:" << endl;
    test_data_in.clear();
    test_data_in.seekg(2 * sizeof(int), ios::beg);
    test_label_in.clear();
    test_label_in.seekg(2 * sizeof(int), ios::beg);
    shared_ptr<float> data_ptr, label_ptr;
    tie(data_ptr, label_ptr) = GetNextBatch(test_data_in, test_label_in, test_data_len, test_label_len);
    NArray predict = cnn_algo.FF(data_ptr, true);
    NArray label = cnn_algo.BP(label_ptr, true);
    PrintAccuracy(predict, label, param, true);
  }
  cout << "Training finished" << endl;
  return 0;
}
