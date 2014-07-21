#include <glog/logging.h>
#include <thread>

bool start = false;

void SayHello(int thrid) {
  while(!start);
  for(int i = 0; i < 100; ++i) {
    LOG(INFO) << "Thread#" << thrid << " say hello #" << i << " times.";
  }
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  std::thread t0(&SayHello, 0);
  std::thread t1(&SayHello, 1);
  start = true;
  t0.join();
  t1.join();
  return 0;
}
