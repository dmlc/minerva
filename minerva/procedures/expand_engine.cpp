#include <iostream>
#include <vector>
#include <tuple>
#include <glog/logging.h>

#include "expand_engine.h"
#include "system/minerva_system.h"

using namespace std;

namespace minerva {

void ExpandEngine::Process(LogicalDag& dag, const std::vector<uint64_t>& nodes) {
  GCNodes(dag);
  for(uint64_t nid : nodes) {
    ExpandNode(dag, nid);
  }
  cout << dag.PrintDag() << endl;
  GCNodes(dag);
  cout << dag.PrintDag() << endl;
}

bool ExpandEngine::IsExpanded(uint64_t lnode_id) const {
  return lnode_to_pnode_.find(lnode_id) != lnode_to_pnode_.end();
}

const NVector<uint64_t>& ExpandEngine::GetPhysicalNodes(uint64_t id) const {
  CHECK(IsExpanded(id)) << "invalid physical nid: " << id;
  return lnode_to_pnode_.find(id)->second;
}
  
void ExpandEngine::OnDeleteDataNode(LogicalDataNode* ldnode) {
  lnode_to_pnode_.erase(ldnode->node_id());
}

void ExpandEngine::GCNodes(LogicalDag& dag) {
  for(uint64_t nid : node_states_.GetNodesOfState(NodeState::kCompleted)) {
    DagNode* node = dag.GetNode(nid);
    switch(node->Type()) {
    case DagNode::OP_NODE:
      node_states_.ChangeState(nid, NodeState::kDead);// op nodes are just GCed
      break;
    case DagNode::DATA_NODE:
      LogicalDataNode* dnode = dynamic_cast<LogicalDataNode*>(node);
      int dep_count = dnode->data_.extern_rc;
      for(DagNode* succ : node->successors_) {
        NodeState succ_state = node_states_.GetState(succ->node_id());
        if(succ_state == NodeState::kBirth || succ_state == NodeState::kReady) {
          ++dep_count;
        }
      }
      if(dep_count == 0) {
        node_states_.ChangeState(nid, NodeState::kDead);
      }
      break;
    }
  }
  // delete node of kDead state 
  for(uint64_t nid : node_states_.GetNodesOfState(NodeState::kDead)) {
    dag.DeleteNode(nid);
  }
}

void ExpandEngine::ExpandNode(LogicalDag& dag, uint64_t lnid) {
  if(!IsExpanded(lnid)) { // haven't been expanded yet
    CHECK_EQ(node_states_.GetState(lnid), NodeState::kBirth);
    node_states_.ChangeState(lnid, NodeState::kReady);

    DagNode* curnode = dag.GetNode(lnid);
    //cout << "Try expand nodeid=" << lnid << " " << curnode->Type() << endl;
    for(DagNode* pred : curnode->predecessors_) {
      ExpandNode(dag, pred->node_id());
    }
    if(curnode->Type() == DagNode::DATA_NODE) { // data node
      LogicalDag::DNode* dnode = dynamic_cast<LogicalDag::DNode*>(curnode);
      // call expand function to generate data
      LogicalDataGenFn* fn = dnode->data_.data_gen_fn;
      if(fn != nullptr) {
        LOG(INFO) << "Expand logical datagen function: " << fn->Name();
        NVector<Scale> partsizes = dnode->data_.partitions.Map<Scale>(
            [] (const PartInfo& pi) { return pi.size; }
          );
        NVector<Chunk> chunks = fn->Expand(partsizes);
        MakeMapping(dnode, chunks);
      }
    }
    else { // op node
      LogicalDag::ONode* onode = dynamic_cast<LogicalDag::ONode*>(curnode);
      LogicalComputeFn* fn = onode->op_.compute_fn;
      CHECK_NOTNULL(fn);
      // make input chunks
      std::vector<NVector<Chunk>> in_chunks;
      for(LogicalDag::DNode* dn : onode->inputs_) {
        NVector<uint64_t> mapped_pnode_ids = lnode_to_pnode_[dn->node_id()];
        in_chunks.push_back(
          mapped_pnode_ids.Map<Chunk>(
            [] (const uint64_t& nid) {
              PhysicalDag::DNode* pnode = MinervaSystem::Instance().physical_dag().GetDataNode(nid);
              return Chunk(pnode);
            }
          )
        );
      }
      // call expand function
      LOG(INFO) << "Expand logical compute function: " << fn->Name();
      std::vector<NVector<Chunk>> rst_chunks = fn->Expand(in_chunks);
      // check output validity
      CHECK_EQ(rst_chunks.size(), onode->outputs_.size()) 
        << "Expand function error: #output unmatched. Function name: " << fn->Name();
      for(size_t i = 0; i < rst_chunks.size(); ++i) {
        MakeMapping(onode->outputs_[i], rst_chunks[i]);
      }
    }
    node_states_.ChangeState(lnid, NodeState::kCompleted);
  }
}

void ExpandEngine::MakeMapping(LogicalDag::DNode* ldnode, const NVector<Chunk>& chunks) {
  Scale numparts = chunks.Size();
  size_t numdims = numparts.NumDims();
  // check size
  NVector<Scale> chunk_sizes = chunks.Map<Scale>( [] (const Chunk& ch) { return ch.Size(); } );
  Scale merged_size = Scale::Merge(chunk_sizes);
  CHECK_EQ(ldnode->data_.size, merged_size)
    << "Expand function error: partition size unmatched!\n"
    << "Expected: " << ldnode->data_.size << "\n"
    << "Got: " << merged_size;
  // offset & offset_index
  Scale pos = Scale::Origin(numdims);
  chunks[pos].data_node()->data_.offset = pos; // the offset of first chunk is zero
  do {
    auto& phy_data = chunks[pos].data_node()->data_;
    phy_data.offset_index = pos;
    Scale upleftpos = pos.Map([] (int x) { return max(x - 1, 0); });
    auto& upleft_phy_data = chunks[upleftpos].data_node()->data_;
    phy_data.offset = upleft_phy_data.offset + upleft_phy_data.size;
    for(size_t i = 0; i < numdims; ++i) {
      if(pos[i] == 0) { // if the index of this dimension is 0, then so does the offset
        phy_data.offset[i] = 0;
      }
    }
    phy_data.extern_rc = ldnode->data_.extern_rc; // set external rc
  } while(Scale::IncrOne(pos, numparts));
  // insert mapping
  lnode_to_pnode_[ldnode->node_id()] = chunks.Map<uint64_t>(
      [&] (const Chunk& ch) {
        return ch.data_node()->node_id();
      }
    );
}

} // end of namespace minerva
