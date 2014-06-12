#ifndef MINERVA_RPC_STUB_H
#define MINERVA_RPC_STUB_H
#pragma once

#include <vector>
#include <iostream>
#include <map>
#include <list>

#include <cstddef>
#include <stdint.h>
#include <cassert>

#include <minerva-common/options/Options.h>
#include <minerva-common/util/Singleton.h>

#include "RPCTypes.h"
#include "RPCProtocol.h"
#include "CommBase.h"

namespace minerva
{

// forward-declaration
class Allocator;

namespace rpc
{

enum TinyErrorCode
{
	SUCCESS = 0
};

class MessageBuffer;

class RPCStub:// : public utils::Singleton<RPCStub>
	public Optionable<RPCStub>
{
public:
	RPCStub();
	~RPCStub();
	//void SetComm(CommBase* comm);
	// Initialize communication, etc.
	//void Init(Options& options);
	//void Init(int argc, char** argv);
	static Options GetOptions();
	void SetOptions(const Options& options);
	// Destroy this rpc stub
	void Destroy();
	// start processing messages
	void StartServing();

	// calls a remote function
	uint32_t RemoteCallOutGroup(const std::string& id, ProtocolBase & protocol);
	uint32_t RemoteCallAsyncOutGroup(const std::string& id, ProtocolBase & protocol);
	uint32_t RemoteCall(ProcId who, ProtocolBase & protocol);
	uint32_t RemoteCallAsync(ProcId who, ProtocolBase & protocol);
	template<typename Iterator> uint32_t RemoteCall(Iterator begin, Iterator end, ProtocolBase & protocol);
	template<typename Iterator> uint32_t RemoteCallAsync(Iterator begin, Iterator end, ProtocolBase & protocol);

	// inline functions
	inline void Barrier() {_comm->Barrier();}
	inline int NumNodes() {return _comm->NumNodes();}
	inline int NodeId() {return _comm->NodeId();}
	template<class T> void RegisterProtocol(void* closure = NULL);

private:
	int64_t MarshallProtocol(MessageBuffer& os, ProtocolBase& protocol, int);
	void HandleRecvEvent(RecvEvent& evt);
	int64_t GetNewSeqNum();
	// main function of worker threads. Looping on handling messages till terminated.
	void WorkerFunction(int threadid);
	// handle messages, called by WorkerFunction
	void HandleMessage(RecvEvent&);
	// return whether this RPC layer is terminating
	bool IsTerminating() const;

private:
	class RequestFactoryBase
	{
	public:
		virtual ProtocolBase * create_protocol()=0;
		virtual ~RequestFactoryBase() {}
	};

	template<class T>
	class RequestFactory : public RequestFactoryBase
	{
	public:
		virtual T * create_protocol(){return new T;};
	};

	template<class Response>
	class SleepingList
	{
	public:
		void wait_for_response(int64_t event, Response * r)
		{
			ScopedUniqueLock l(_lock);
			// if no response yet
			if (_list.find(event) == _list.end())
			{
				_list[event] = std::make_pair(false, r);
				_response_ptr_cond.notify_all();
			}
			while(_list[event].first == false)
				_wake_cond.wait(l);
			_list.erase(event);
		}
		Response * get_response_ptr(int64_t event)
		{
			ScopedUniqueLock l(_lock);
			while (_list.find(event) == _list.end())
				_response_ptr_cond.wait(l);
			return _list[event].second;
		}
		void signal(int64_t event)
		{
			ScopedUniqueLock l(_lock);
			_list[event].first = true;
			_wake_cond.notify_all();
		}
	private:
		std::map<int64_t, std::pair<bool, Response*> > _list;
		Lock _lock;
		CondVar _wake_cond;
		CondVar _response_ptr_cond;
	};

	typedef std::map<uint32_t, std::pair<RequestFactoryBase *, void*> > RequestFactories;

private:
	static const uint32_t ASYNC_RPC_CALL = 1;
	static const uint32_t SYNC_RPC_CALL = 0;
	CommBase* _comm;
	// for request handling
	RequestFactories _protocol_factory;
	// threads
	size_t _num_worker_threads;
	ThreadGroup _worker_threads;
	bool _terminating;
	// sequence number
	int64_t _seq_num;
	Lock _seq_lock;
	// waiting list for synchronous remote call
	SleepingList<ProtocolBase> _sleeping_list;
	// Recv queues
	typedef utils::BlockingQueue<RecvEvent> RecvEventBuffer;
	std::vector<RecvEventBuffer* > _received_events;
};

/*inline void RPCStub::SetComm(CommBase* comm)
{
	_comm = comm;
	_comm->AddRecvCallback(boost::bind(&RPCStub::HandleRecvEvent, this, _1));
}*/

template<class T>
void RPCStub::RegisterProtocol(void* closure)
{
	T * t = new T;
	uint32_t id = t->ID();
	delete t;
	if (_protocol_factory.find(id) != _protocol_factory.end())
	{
		// ID() should be unique, and should not be re-registered
		assert(0);
	}
	_protocol_factory[id] = std::make_pair(new RequestFactory<T>(), closure);
}

template<typename Iterator>
uint32_t RPCStub::RemoteCall(Iterator begin, Iterator end, ProtocolBase & protocol)
{
	MessagePtr message(new MessageBuffer);
	int64_t seq = MarshallProtocol(*message, protocol, RPCStub::SYNC_RPC_CALL);
	SendEvent evt;
	for(Iterator iter = begin; iter != end; ++iter)
	{
		evt.group = _comm->GroupId();
		evt.rank = *iter;
		evt.msg = message;
		_comm->Send(evt);
		_sleeping_list.wait_for_response(seq, &protocol);
	}
	return SUCCESS;
}
template<typename Iterator>
uint32_t RPCStub::RemoteCallAsync(Iterator begin, Iterator end, ProtocolBase & protocol)
{
	MessagePtr message(new MessageBuffer);
	MarshallProtocol(*message, protocol, RPCStub::ASYNC_RPC_CALL);
	SendEvent evt;
	for(Iterator iter = begin; iter != end; ++iter)
	{
		evt.group = _comm->GroupId();
		evt.rank = *iter;
		evt.msg = message;
		_comm->Send(evt);
	}
	return SUCCESS;
}

}// end of namespace rpc
}// end of namespace minerva

#endif // end of head guard
