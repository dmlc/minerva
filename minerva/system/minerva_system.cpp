#include <glog/logging.h>
//#include <fstream>

#include "minerva_system.h"
#include "op/impl/basic.h"

using namespace std;

namespace minerva {

void MinervaSystem::Initialize(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  LoadBuiltinDagMonitors();
}
void MinervaSystem::Finalize() {
}

MinervaSystem::MinervaSystem(): expand_engine_(lnode_states_), physical_engine_(pnode_states_) {
}

void MinervaSystem::LoadBuiltinDagMonitors() {
  logical_dag_.RegisterMonitor(&lnode_states_);
  logical_dag_.RegisterMonitor(&expand_engine_);
  physical_dag_.RegisterMonitor(&pnode_states_);
  physical_dag_.RegisterMonitor(&physical_engine_);
}

void MinervaSystem::Eval(NArray& narr) {
  std::vector<uint64_t> id_to_eval = {narr.data_node_->node_id()};
  // convert logical dag to physical dag
  expand_engine_.Process(logical_dag_, id_to_eval);
  // do real computation
  auto physical_nodes = expand_engine_.GetPhysicalNodes(narr.data_node_->node_id());
  physical_engine_.Process(physical_dag_, physical_nodes.ToVector());
}

float* MinervaSystem::GetValue(NArray& narr) {
  NVector<uint64_t> phy_nid = expand_engine_.GetPhysicalNodes(narr.data_node_->node_id());
  NVector<DataShard> data_shards = phy_nid.Map<DataShard>(
      [&] (uint64_t id) { return DataShard(physical_dag_.GetDataNode(id)->data_); }
    );
  float* rst_ptr = new float[narr.Size().Prod()];
  basic::Assemble(data_shards, rst_ptr, narr.Size());
  return rst_ptr;
}

void MinervaSystem::IncrExternRC(LogicalDag::DNode* dnode, int amount) {
  dnode->data_.extern_rc += amount;
  uint64_t ldnode_id = dnode->node_id();
  if(expand_engine_.IsExpanded(ldnode_id)) {
    // this means the node's physical data has been created
    // incr the extern_rc of its physical_node, and change the rc in data_store
    for(uint64_t pnode_id : expand_engine_.GetPhysicalNodes(ldnode_id)) {
      PhysicalDataNode* pnode = physical_dag_.GetDataNode(pnode_id);
      pnode->data_.extern_rc += amount;
    }
  }
}
  
} // end of namespace minerva
