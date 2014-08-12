#include <iostream>
#include <vector>
#include <tuple>
#include <glog/logging.h>

#include "expand_engine.h"
#include "system/minerva_system.h"

using namespace std;

namespace minerva {

void ExpandEngine::Process(LogicalDag& dag, NodeStateMap<LogicalDag>& node_states,
    const std::vector<uint64_t>& nodes) {
  for(uint64_t nid : nodes) {
    ExpandNode(dag, node_states, nid);
  }
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

void ExpandEngine::GCNodes(LogicalDag& dag, NodeStateMap<LogicalDag>& node_states) {
  vector<uint64_t> dead_nodes;
  for(uint64_t nid : node_states.GetNodesOfState(NodeState::kDead)) {
    dead_nodes.push_back(nid);
  }
  // delete node of kDead state 
  for(uint64_t nid : dead_nodes) {
    dag.DeleteNode(nid);
  }
}

void ExpandEngine::ExpandNode(LogicalDag& dag, NodeStateMap<LogicalDag>& node_states, 
    uint64_t lnid) {
  DagNode* curnode = dag.GetNode(lnid);
  if(!IsExpanded(lnid)) { // haven't been expanded yet
    CHECK_EQ(node_states.GetState(lnid), NodeState::kBirth);
    node_states.ChangeState(lnid, NodeState::kReady);
    for(DagNode* pred : curnode->predecessors_) {
      ExpandNode(dag, node_states, pred->node_id());
    }
    if(curnode->Type() == DagNode::DATA_NODE) { // data node
      LogicalDag::DNode* dnode = dynamic_cast<LogicalDag::DNode*>(curnode);
      // call expand function to generate data
      LogicalDataGenFn* fn = dnode->data_.data_gen_fn;
      if(fn != nullptr) {
        LOG(INFO) << "Expand logical datagen function: " << fn->Name();
        NVector<Chunk> chunks = fn->Expand(dnode->data_.partitions);
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
  }

  // change states
  if(curnode->Type() == DagNode::OP_NODE) {
    node_states.ChangeState(lnid, NodeState::kDead); // the op node is expanded thus could be GCed
  } else {
    LogicalDag::DNode* dnode = dynamic_cast<LogicalDag::DNode*>(curnode);
    if(dnode->data_.extern_rc != 0) {
      // the data node is expanded but with external dependencies
      node_states.ChangeState(lnid, NodeState::kCompleted);
    } else {
      // the data node could be GCed
      node_states.ChangeState(lnid, NodeState::kDead);
    }
  }
}

void ExpandEngine::MakeMapping(LogicalDag::DNode* ldnode, NVector<Chunk>& chunks) {
  Scale numparts = chunks.Size();
  // check size & set offset, offset_index
  // TODO how to make this faster ?
  Scale merged_size = Chunk::ComputeOffset(chunks);
  //CHECK_EQ(ldnode->data_.size, merged_size)
    //<< "Expand function error: partition size unmatched!\n"
    //<< "Expected: " << ldnode->data_.size << "\n"
    //<< "Got: " << merged_size;
  // set external rc
  for(auto ch : chunks) {
    ch.data_node()->data_.extern_rc = ldnode->data_.extern_rc;
  }
  // insert mapping
  lnode_to_pnode_[ldnode->node_id()] = chunks.Map<uint64_t>(
      [&] (const Chunk& ch) {
        return ch.data_node()->node_id();
      }
    );
}

} // end of namespace minerva
