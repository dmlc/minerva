#include <glog/logging.h>
//#include <fstream>

#include "minerva_system.h"
#include "op/impl/basic.h"

using namespace std;

namespace minerva {

void MinervaSystem::Initialize(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
}
void MinervaSystem::Finalize() {
}

void MinervaSystem::Eval(NArray narr) {
  std::vector<uint64_t> id_to_eval = {narr.data_node_->node_id_};
  expand_engine_.Process(logical_dag_, id_to_eval);
  auto physical_nodes = expand_engine_.GetPhysicalNodes(narr.data_node_->node_id_);
  physical_engine_.Process(physical_dag_, physical_nodes.ToVector());
}

float* MinervaSystem::GetValue(NArray narr) {
  NVector<uint64_t> phy_nid = expand_engine_.GetPhysicalNodes(narr.data_node_->node_id_);
  NVector<DataShard> data_shards = phy_nid.Map<DataShard>(
      [&] (uint64_t id) { return DataShard(physical_dag_.GetDataNode(id)->data_); }
    );
  float* rst_ptr = new float[narr.Size().Prod()];
  basic::Assemble(data_shards, rst_ptr, narr.Size());
  return rst_ptr;
}

}
