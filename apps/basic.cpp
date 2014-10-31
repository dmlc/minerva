#include <iostream>
#include <minerva.h>
#include <cublas_v2.h>

using namespace std;
using namespace minerva;

uint64_t cpu_device, gpu_device;

void Train() {
  auto& ms = MinervaSystem::Instance();
  ms.current_device_id_ = cpu_device;
  Scale img_size{8, 8, 1, 1};
  auto img = NArray::Randn(img_size, 0, 1);
  Scale filter_size{3, 3, 1, 1};
  auto filter1 = NArray::Randn(filter_size, 0, 1);
  shared_ptr<float> filter2_ptr(new float[filter_size.Prod()], [](float* ptr) { delete[] ptr; });
  auto filter1_ptr = filter1.Get();
  for (int i = 0; i < filter_size.Prod(); ++i) {
    filter2_ptr.get()[i] = filter1_ptr.get()[i];
  }
  filter2_ptr.get()[0] += 0.0001;
  auto filter2 = NArray::MakeNArray(filter_size, filter2_ptr);
  auto bias = NArray::Zeros({1});

  ms.current_device_id_ = gpu_device;
  ConvInfo conv_info{0, 0, 1, 1};
  auto top1 = Convolution::ConvForward(img, filter1, bias, conv_info);
  auto top2 = Convolution::ConvForward(img, filter2, bias, conv_info);
  auto top_diff = top1 - top2;
  auto filter_diff = Convolution::ConvBackwardFilter(top_diff, img, conv_info);
  auto top_diff_ptr = filter_diff.Get();
  for (int i = 0; i < filter_diff.Size().Prod(); ++i) {
    cout << top_diff_ptr.get()[i] << ' ';
  }
  cout << endl;
}

int main(int argc, char** argv) {
  auto& ms = MinervaSystem::Instance();
  ms.Initialize(&argc, &argv);
  cpu_device = ms.CreateCpuDevice();
  gpu_device = ms.CreateGpuDevice(1);
  Train();
  ms.dag_scheduler().GCNodes();
  cout << ms.device_manager().GetDevice(cpu_device)->GetMemUsage() << endl;
  cout << ms.device_manager().GetDevice(gpu_device)->GetMemUsage() << endl;
  ms.Finalize();
  return 0;
}
