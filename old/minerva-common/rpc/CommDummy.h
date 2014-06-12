#ifndef MINERVA_RPC_COMM_DUMMY_H
#define MINERVA_RPC_COMM_DUMMY_H
#pragma once

#include "CommBase.h"

namespace minerva
{
namespace rpc
{
	class CommDummy : public CommBase
	{
	public:
		void Finalize() { }
		void Send(SendEvent& evt) { }
		void Barrier() { }
		int NumNodes() { return 1; }
		ProcId NodeId() { return 0; }
		void StartPolling() {}
		void ClosePolling() {}
	protected:
		void PollingFunction() {}
	};
} // end of namespace rpc
} // end of namespace minerva

#endif
