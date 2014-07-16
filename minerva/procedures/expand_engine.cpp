#include "expand_engine.h"
#include "system/minerva_system.h"
#include <iostream>

using namespace std;

namespace minerva {

void ExpandEngine::Process(LogicalDag& dag, std::vector<uint64_t>& nodes) {
  for(uint64_t nid : nodes) {
    ExpandNode(dag, nid);
  }
}
void ExpandEngine::ExpandNode(LogicalDag& dag, uint64_t lnid) {
  if(lnode_to_pnode_.find(lnid) == lnode_to_pnode_.end()) { // haven't been expanded yet
    DagNode* curnode = dag.GetNode(lnid);
    //cout << "Try expand nodeid=" << lnid << " " << curnode->Type() << endl;
    for(DagNode* pred : curnode->predecessors_) {
      ExpandNode(dag, pred->node_id_);
    }
    if(curnode->Type() == DagNode::DATA_NODE) { // data node
      LogicalDag::DNode* dnode = dynamic_cast<LogicalDag::DNode*>(curnode);
      LogicalDataGenFn* fn = dnode->data_.data_gen_fn;
      if(fn != NULL) {
        cout << "Logical datagen function: " << fn->Name() << endl;
        NVector<Chunk> chunks = fn->Expand(dnode->data_.size);
        MakeMapping(dnode, chunks);
      }
    }
    else { // op node
      LogicalDag::ONode* onode = dynamic_cast<LogicalDag::ONode*>(curnode);
      LogicalComputeFn* fn = onode->op_.compute_fn;
      // make input chunks
      std::vector<NVector<Chunk>> in_chunks;
      for(LogicalDag::DNode* dn : onode->inputs_) {
        NVector<uint64_t> mapped_pnode_ids = lnode_to_pnode_[dn->node_id_];
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
      cout << "Logical compute function: " << fn->Name() << endl;
      std::vector<NVector<Chunk>> rst_chunks = fn->Expand(in_chunks);
      // check output validity
      if(rst_chunks.size() != onode->outputs_.size()) {
        cout << "Expand function error: Wrong numbers of results" << endl;
        cout << "Expected: " << onode->outputs_.size() << endl;
        cout << "Returns: " << rst_chunks.size() << endl;
        cout << "Function name: " << fn->Name() << endl;
        assert(false);
      }
      for(size_t i = 0; i < rst_chunks.size(); ++i) {
        MakeMapping(onode->outputs_[i], rst_chunks[i]);
      }
    }
  }
}
void ExpandEngine::MakeMapping(LogicalDag::DNode* ldnode, const NVector<Chunk>& chunks) {
  // check size
  NVector<Scale> chunk_sizes = chunks.Map<Scale>( [] (const Chunk& ch) { return ch.Size(); } );
  Scale merged_size = Scale::Merge(chunk_sizes);
  if(ldnode->data_.size != merged_size) {
    cout << "Expand function error: Unmatched return size" << endl;
    cout << "Expected: " << ldnode->data_.size << endl;
    cout << "Returned: " << merged_size << endl;
    assert(false);
  }
  // offset & offset_index 
  // TODO
  // insert mapping
  lnode_to_pnode_[ldnode->node_id_] = chunks.Map<uint64_t>(
      [] (const Chunk& ch) {
        return ch.data_node()->node_id_;
      }
    );
}

} // end of namespace minerva
