#ifndef MINERVA_RPC_COMM_H
#define MINERVA_RPC_COMM_H
#pragma once

#include <stdint.h>

#include <minerva/options/MinervaOptions.h>
#include <minerva/rpc/RPCTypes.h>
#include <minerva/util/BlockingQueue.h>
#include <vector>

namespace minerva
{
namespace rpc 
{

struct RecvEvent
{
	int group, rank;
	std::string id;
	MessagePtr msg;
};

struct SendEvent
{
	int group, rank;
	std::string id;
	MessagePtr msg;
};

class IComm // interface class for communicator
{
public:
	typedef boost::function<void(RecvEvent)> RecvEventCallback;
	//virtual void Init(int * argc, char *** argv) = 0;
	//virtual void Init(MinervaOptions& options) = 0;
	virtual void Finalize() = 0;
	virtual void AddRecvCallback(RecvEventCallback cb) = 0;
	virtual void Send(SendEvent& ) = 0;
	virtual void Barrier() = 0;
	// start polling for messages
	virtual void StartPolling() = 0;
	virtual void ClosePolling() = 0;
	virtual int NumNodes() = 0;
	virtual ProcId NodeId() = 0;
};

class CommBase:
	public IComm,
	public Optionable<CommBase>
{
public:
	//typedef utils::BlockingQueue<MessageBuffer*> RecvMessageQueue;
	static const int NUM_POLLING_THREADS = 1;
public:
	CommBase();
	virtual ~CommBase() {}

	static MinervaOptions GetOptions();
	virtual void SetOptions(const MinervaOptions& options);

	//void SetGroup(int g) { group = g; }
	int GroupId() const { return group; }
	//void SetNodeId(int r) { rank = r; }
	ProcId NodeId() { return rank; }
	//void SetNumNodes(int n) { numnodes = n; }
	int NumNodes() { return numnodes; }

	void AddRecvCallback(RecvEventCallback cb) { callbacks.push_back(cb); }

	// abstract functions
	virtual void Finalize() = 0;
	virtual void Send(SendEvent& ) = 0;
	virtual void Barrier() = 0;

	// start polling for messages
	virtual void StartPolling();
	virtual void ClosePolling();

protected:
	virtual void PollingFunction() = 0;
	bool IsTerminating() const;
	void TriggerRecvCallbacks(RecvEvent& );

protected:
	// thread
	ThreadPtr _polling_thread;
	//RecvMessageQueue& _received_messages;
	std::vector<RecvEventCallback> callbacks;
	int rank;
	int numnodes;
	int group;
private:
	// terminate mechanism
	bool _terminating;
};

}
}
#endif
