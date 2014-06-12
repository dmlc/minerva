#include "RPCStub.h"
#include "Message.h"
#include "CommZMQ.h"
#include "CommDummy.h"

#include <minerva-common/logger/log.h>
#include <minerva-common/macro_def.h>

DEF_LOG_MODULE(RPCStub)
ENABLE_LOG_MODULE(RPCStub)
#define _TRACE LOG_TRACE(RPCStub)
#define _DEBUG LOG_DEBUG(RPCStub)
#define _INFO LOG_INFO(RPCStub)
#define _ERROR LOG_ERROR(RPCStub)

namespace minerva
{
namespace rpc 
{

//template<> RPCStub* utils::Singleton<RPCStub>::_instance = NULL;
//static const int NUM_WORKER_THREADS = 1;

RPCStub::RPCStub(): _comm(NULL), _terminating(true) {}
RPCStub::~RPCStub() { Destroy(); }

Options RPCStub::GetOptions()
{
	Options rpcopt("RPC options");
	rpcopt.AddOption<std::string>("rpc.comm", "Communicator used by RPC communication (dummy,zmq)", "dummy");
	rpcopt.AddOption<size_t>("rpc.numthreads", "Number of threads used to handle RPC callbacks", 1);
	rpcopt.AddOptions(CommBase::GetOptions());
	rpcopt.AddOptions(CommGroupZMQ::GetOptions());
	return rpcopt;
}
void RPCStub::SetOptions(const Options& options)
{
	std::string commtype = options.Get<std::string>("rpc.comm");
	_num_worker_threads = options.Get<size_t>("rpc.numthreads");
	assert(_terminating);
	// create communicator and set options
	if(commtype == "zmq")
		_comm = new CommGroupZMQ();
	else
		_comm = new CommDummy();
	_comm->SetOptions(options);
	_comm->AddRecvCallback(boost::bind(&RPCStub::HandleRecvEvent, this, _1));
}

void RPCStub::StartServing()
{
	if(_terminating)
	{
		_seq_num = 1;
		_terminating = false;
		for(size_t i = 0; i < _num_worker_threads; ++i)
			_received_events.push_back(new RecvEventBuffer());
		// start polling
		assert(_comm);
		_comm->StartPolling();
		// start threads
		for(size_t i=0; i < _num_worker_threads; i++)
		{
			_worker_threads.add_thread(new Thread(boost::bind(&RPCStub::WorkerFunction, this, i)));
		}
	}
}
	
void RPCStub::Destroy()
{
	if(!_terminating)
	{
		// stop communicator
		_comm->ClosePolling();
		_comm->Finalize();
		delete _comm;
		_TRACE << "Comm closed polling";
		// stop RPC callback threads
		_terminating = true;
		for(size_t i = 0; i < _received_events.size(); ++i)
			_received_events[i]->ForceStop(); // stop receiving queue after close Communicator
		_TRACE << "recv queue stopped";
		// wait for all threads to exit
		_worker_threads.join_all();
		_TRACE << "All worker threads exited";
		// clear all receiving buffers
		for(size_t i = 0; i < _received_events.size(); ++i)
			delete _received_events[i];
		_received_events.clear();
	}
}

void RPCStub::HandleRecvEvent(RecvEvent& evt)
{
	// TODO not thread safe
	typedef std::pair<int, int> IdPair;
	static int thrptr = 0;
	static std::map<IdPair, int> recvids;
	IdPair id(evt.group, evt.rank);
	if(recvids.find(id) == recvids.end())
		recvids[id] = (thrptr++) % _received_events.size();
	_received_events[recvids[id]]->Push(evt);
	_TRACE << "receive queue length: " << _received_events[recvids[id]]->Size();
}

void RPCStub::WorkerFunction(int threadid)
{
	RecvEvent evt;
	while(1)
	{
		_received_events[threadid]->Pop(evt); // (Blocking) pop received messages out of message queue
		if(IsTerminating() && _received_events[threadid]->Empty())
			break;
		try
		{
			HandleMessage(evt);
		}
		catch(const char* ex)
		{
			std::cout << "Catched exception in handling messages" << std::endl;
			assert(false);
		}
		DeleteMessagePtr(evt.msg);
	}
	//ScopedUniqueLock ul(_term_lock);
	//assert(_terminating);
	//--_num_running_workers;
	//_term_cond.notify_all();
}
bool RPCStub::IsTerminating() const
{
	//ScopedUniqueLock ul(_term_lock);
	return _terminating;
}

int64_t RPCStub::GetNewSeqNum()
{
	ScopedUniqueLock ul(_seq_lock);
	if (_seq_num >= LONG_MAX - 1)
		_seq_num = 1;
	return _seq_num++;
}

//---------------------------------
// message format:
//      int64_t seq_num:		positive means request, negtive means response
//		uint32_t protocol_id:	protocol id, must be registered in both client and server
//	for request
//		uint32_t async:			0 means sync call, 1 means async call
//		char * buf:				request of the protocol
//	for response
//		char * buf:				response of the protocol

void RPCStub::HandleMessage(RecvEvent& rcvevt)
{
	MessageBuffer& buf = *(rcvevt.msg);
	//std::cout << "handle message buf.gsize=" << buf->gsize() << std::endl;
	// get seq number, test if request or response
	int64_t seq;
	buf >> seq;
	//std::cout << "handle message seq=" << seq << std::endl;
	// get the protocol handle
	uint32_t protocol_id;
	buf >> protocol_id;
	//std::cout << "handle message protocol_id=" << protocol_id << std::endl;

	if (seq < 0)
	{
		// a response
		seq = -seq;
		// unmarshal response
		ProtocolBase * protocol = _sleeping_list.get_response_ptr(seq);
		protocol->UnmarshallResponse(buf);
		// wake up waiting thread
		_sleeping_list.signal(seq);
	}
	else
	{
		if (_protocol_factory.find(protocol_id) == _protocol_factory.end() )
		{
			std::cout << "Unsupported protocol_id=" << protocol_id << std::endl;
			assert(false);
			// unsupported call, register the func with the server please!
			//int remote = buf->get_remote_rank();
			//TinyLog(LOG_ERROR, "Unsupported call from %d, request ID=%d", remote, protocol_id);
			return;
		}
		ProtocolBase * protocol = _protocol_factory[protocol_id].first->create_protocol();
		// a request
		uint32_t is_async;
		buf >> is_async;
		// handle request
		protocol->UnmarshallRequest(buf);
		protocol->SetSender(rcvevt.id);
		protocol->HandleRequest(_protocol_factory[protocol_id].second);
		// send response if sync call
		if (!is_async)
		{
			SendEvent sndevt;
			MessagePtr out_buffer(new MessageBuffer);
			(*out_buffer) << -seq << protocol_id;
			protocol->MarshallResponse(*out_buffer);	
			//TinyLog(LOG_NORMAL, "responding to %d with seq=%d, protocol_id=%d\n", buf->get_remote_rank(), -seq, protocol_id);
			sndevt.group = rcvevt.group;
			sndevt.rank = rcvevt.rank;
			sndevt.id = rcvevt.id;
			sndevt.msg = out_buffer;
			_comm->Send(sndevt);
		}
		delete protocol;
	}
}

int64_t RPCStub::MarshallProtocol(MessageBuffer& os, ProtocolBase& protocol, int sync)
{
	int64_t seq = GetNewSeqNum();
	os << seq << protocol.ID() << sync;
	protocol.MarshallRequest(os);
	return seq;
}

uint32_t RPCStub::RemoteCallOutGroup(const std::string& id, ProtocolBase & protocol)
{
	MessagePtr message(new MessageBuffer);
	int64_t seq = MarshallProtocol(*message, protocol, SYNC_RPC_CALL);
	_TRACE << "After marshall message size=" << message->gsize();
	// send message
	SendEvent sndevt;
	sndevt.group = -1; sndevt.rank = -1;
	sndevt.id = id;
	sndevt.msg = message;
	_comm->Send(sndevt);
	// wait for signal
	_sleeping_list.wait_for_response(seq, &protocol);
	return SUCCESS;
}

uint32_t RPCStub::RemoteCallAsyncOutGroup(const std::string& id, ProtocolBase & protocol)
{
	MessagePtr message(new MessageBuffer);
	MarshallProtocol(*message, protocol, ASYNC_RPC_CALL);
	_TRACE << "After marshall message size=" << message->gsize();
	// send message
	SendEvent sndevt;
	sndevt.group = -1; sndevt.rank = -1;
	sndevt.id = id;
	sndevt.msg = message;
	_comm->Send(sndevt);
	return SUCCESS;
}

uint32_t RPCStub::RemoteCall(ProcId who, ProtocolBase & protocol)
{
	MessagePtr message(new MessageBuffer);
	int64_t seq = MarshallProtocol(*message, protocol, SYNC_RPC_CALL);
	_TRACE << "After marshall message size=" << message->gsize();
	// send message
	SendEvent sndevt;
	sndevt.group = _comm->GroupId();
	sndevt.rank = who;
	sndevt.msg = message;
	_comm->Send(sndevt);
	// wait for signal
	_sleeping_list.wait_for_response(seq, &protocol);
	return SUCCESS;
}

uint32_t RPCStub::RemoteCallAsync(ProcId who, ProtocolBase & protocol)
{
	MessagePtr message(new MessageBuffer);
	MarshallProtocol(*message, protocol, ASYNC_RPC_CALL);
	_TRACE << "After marshall message size=" << message->gsize();
	// send message
	SendEvent sndevt;
	sndevt.group = _comm->GroupId();
	sndevt.rank = who;
	sndevt.msg = message;
	_comm->Send(sndevt);
	return SUCCESS;
}

} // end of namespace rpc
} // end of namespace minerva
