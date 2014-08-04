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

void MinervaSystem::LoadBuiltinDagMonitors() {
  logical_dag_.RegisterMonitor(&expand_engine_);
  physical_dag_.RegisterMonitor(&physical_engine_);
  physical_dag_.RegisterMonitor(this);
}

void MinervaSystem::Eval(NArray& narr) {
  std::vector<uint64_t> id_to_eval = {narr.data_node_->node_id()};
  // convert logical dag to physical dag
  expand_engine_.Process(logical_dag_, id_to_eval);
  // clean useless nodes in dag of last evaluation.
  GCDag();
  // do real computation
  auto physical_nodes = expand_engine_.GetPhysicalNodes(narr.data_node_->node_id());
  physical_engine_.Process(physical_dag_, physical_nodes.ToVector());
  // update alive node set
  const auto& expanded_set = expand_engine_.last_expanded_nodes();
  lnodes_pending_gc.insert(expanded_set.begin(), expanded_set.end());
  const auto& executed_set = physical_engine_.last_executed_nodes();
  pnodes_pending_gc.insert(executed_set.begin(), executed_set.end());
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

template<class M, class K, class V>
void AddValueInMap(M& m, const K& k, const V& v, const V& init) {
  if(m.find(k) == m.end()) {
    m[k] = init;
  }
  m[k] += v;
}

void MinervaSystem::OnCreateEdge(DagNode* pnode_from, DagNode* pnode_to) {
  uint64_t pnode_id = pnode_from->node_id();
  if(pnodes_pending_gc.find(pnode_id) != pnodes_pending_gc.end()) {
    // an existing nodes are used
    AddValueInMap(pdnode_rc_delta_, pnode_id, 1, 0);
  }
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
      AddValueInMap(pdnode_rc_delta_, pnode_id, amount, 0);
    }
  }
}
  
void MinervaSystem::GCDag() {
  cout << "alive_lnodes: " << lnodes_pending_gc << endl;
  //cout << "alive_pnodes: " << pnodes_pending_gc << endl;

  // GC logical dag
  std::vector<uint64_t> dead_lnodes;
  for(uint64_t lnode_id : lnodes_pending_gc) {
    auto lnode = logical_dag_.GetNode(lnode_id);
    if(lnode->Type() == DagNode::OP_NODE) { // op node should be GCed
      logical_dag_.DeleteNode(lnode_id);
      dead_lnodes.push_back(lnode_id);
    } else {
      auto ldnode = logical_dag_.GetDataNode(lnode_id);
      if(ldnode->data_.extern_rc == 0) { // GC data node when extern_rc == 0
        logical_dag_.DeleteNode(lnode_id);
        dead_lnodes.push_back(lnode_id);
      }
    }
  }
  // remove those dead nodes in alive set
  for(uint64_t id : dead_lnodes) {
    lnodes_pending_gc.erase(id);
  }

  // GC physical dag
  std::vector<uint64_t> dead_pnodes;
  for(uint64_t pnode_id : pnodes_pending_gc) {
    auto pnode = physical_dag_.GetNode(pnode_id);
    if(pnode->Type() == DagNode::OP_NODE) { // op node should be GCed
      physical_dag_.DeleteNode(pnode_id);
      dead_pnodes.push_back(pnode_id);
    } else {
      auto pdnode = physical_dag_.GetDataNode(pnode_id);
      int data_id = pdnode->data_.data_id;
      if(pdnode_rc_delta_.find(pnode_id) != pdnode_rc_delta_.end()) {
        // if rc is changed due to new operations
        int rc_delta = pdnode_rc_delta_[pnode_id];
        data_store_.IncrReferenceCount(data_id, rc_delta);
      }
      if(!data_store_.ExistData(data_id)) {
        // if the data is deleted, this means this node is also useless
        physical_dag_.DeleteNode(pnode_id);
        dead_pnodes.push_back(pnode_id);
      }
    }
  } 
  // remove dead nodes in alive set
  for(uint64_t id : dead_pnodes) {
    pnodes_pending_gc.erase(id);
  }
  
  // clear rc deltas
  pdnode_rc_delta_.clear();
}

} // end of namespace minerva
