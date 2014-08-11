#include <minerva.h>

using namespace std;
using namespace minerva;

int main(int argc, char** argv) {
  MinervaSystem::Instance().Initialize(&argc, &argv);
  return 0;
}
