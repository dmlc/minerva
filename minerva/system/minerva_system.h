#pragma once

#include "common/singleton.h"
#include "dag/logical_dag.h"
#include "dag/physical_dag.h"
#include "system/data_store.h"
#include "procedures/expand_engine.h"
#include "procedures/physical_engine.h"
#include "narray/narray.h"

namespace minerva {

class MinervaSystem :
  public Singleton<MinervaSystem>, public PhysicalDagMonitor {
  friend class NArray;
 public:
  MinervaSystem() {}
  void Initialize(int argc, char** argv);
  void Finalize();
  LogicalDag& logical_dag() { return logical_dag_; }
  PhysicalDag& physical_dag() { return physical_dag_; }
  DataStore& data_store() { return data_store_; }
  PhysicalEngine& physical_engine() { return physical_engine_; }

  void Eval(NArray& narr);
  float* GetValue(NArray& narr);

 private:
  void LoadBuiltinDagMonitors();
  void IncrExternRC(LogicalDag::DNode* , int amount = 1);
  void GCDag();
  void OnCreateEdge(DagNode* pnode_from, DagNode* pnode_to);

 private:
  LogicalDag logical_dag_;
  PhysicalDag physical_dag_;
  DataStore data_store_;
  ExpandEngine expand_engine_;
  PhysicalEngine physical_engine_;

  // nodes pending for gc after last evaluation
  // a dag node should include these states
  // birth -> no need -> ready -> completed -> pending gc -> dead
  // The condition for a logical node to change from "pending gc" -> "dead":
  //    for_all succ of the node, they are in "pending gc" or "dead" states
  //    AND extern_rc == 0
  // The condition for a physical node to change from "pending gc" -> "dead":
  //    for_all succ of the node, they are in "pending gc" or "dead" states
  //    AND extern_rc == 0
  std::set<uint64_t> lnodes_pending_gc, pnodes_pending_gc;
  std::unordered_map<uint64_t, int> pdnode_rc_delta_;
};

} // end of namespace minerva
