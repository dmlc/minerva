#include <minerva.h>
#include <gtest/gtest.h>

using namespace minerva;

class MinervaTestEnvironment : public testing::Environment {
 public:
  MinervaTestEnvironment(int* argc, char*** argv): argc(argc), argv(argv) {
  }
  void SetUp() {
    MinervaSystem::Instance().Initialize(argc, argv);
  }
  void TearDown() {
    MinervaSystem::Instance().Finalize();
  }
 private:
  int* argc;
  char*** argv;
};

int main(int argc, char** argv) {
  testing::AddGlobalTestEnvironment(new MinervaTestEnvironment(&argc, &argv));
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
