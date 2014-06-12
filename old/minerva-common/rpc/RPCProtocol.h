#ifndef MINERVA_RPC_PROTOCOL_H
#define MINERVA_RPC_PROTOCOL_H
#pragma once

#include "Serialize.h"
#include "Message.h"

namespace minerva
{
namespace rpc 
{

enum RemoteProtocol
{
	// Worker protocol
	W_REMOTE_TRIGGER = 40000,
	W_REMOTE_ASK_FINISH,
	W_REMOTE_FETCH,
	// AETable protocol
	AE_REG_VAL_PROTO,
	AE_REG_TABLE_PROTO,
	AE_RECV_VERSION_PROTO,
	AE_RECV_UPDATES_PROTO,
	AE_PULL_UPDATES_PROTO,
	AE_HEARTBEAT_PROTO,
	// AEService protocol
	PS_REG_VAL,
	PS_PUSH_DELTA,
	PS_PULL_REQUEST,
	PS_PULL_RESPONSE,
	NUM_PROTOS
};

	// Basic Protocol class
class ProtocolBase
{
public:
	// Serialize request arguments
	virtual MessageBuffer & MarshallRequest(MessageBuffer & os)=0;
	// Deserialize request arguments
	virtual MessageBuffer & UnmarshallRequest(MessageBuffer & is)=0; 
	// Perform function call
	virtual void HandleRequest(void * closure) =0;
	// Serialize response arguments
	virtual MessageBuffer & MarshallResponse(MessageBuffer & os) =0;
	// Deserialize response arguments
	virtual MessageBuffer & UnmarshallResponse(MessageBuffer & is)=0;
	// Unique id to identify this RPC call
	virtual uint32_t ID() = 0;

	virtual ~ProtocolBase() {}

	std::string GetSender() const { return _sender; }
	void SetSender(const std::string& sid) { _sender = sid; }

private:
	std::string _sender;
};

	///////////////////////////// Protocol templates for easy usage //////////////////////////////
class ProtocolNoResponse : public virtual ProtocolBase
{
public:
	// Serialize response arguments
	virtual MessageBuffer & MarshallResponse(MessageBuffer & os) { return os; }
	// Deserialize response arguments
	virtual MessageBuffer & UnmarshallResponse(MessageBuffer & is) { return is; }
};

class ProtocolNoRequest : public virtual ProtocolBase
{
public:
	// Serialize response arguments
	virtual MessageBuffer & MarshallRequest(MessageBuffer & os) { return os; }
	// Deserialize response arguments
	virtual MessageBuffer & UnmarshallRequest(MessageBuffer & is) { return is; }
};

class ProtocolNoData : public ProtocolNoRequest, public ProtocolNoResponse { };

template<class RequestT, class ResponseT>
class ProtocolTemplate : public ProtocolBase
{
public:
	virtual MessageBuffer & MarshallRequest(MessageBuffer & os)
	{
		return os << request;
	}
	virtual MessageBuffer & UnmarshallRequest(MessageBuffer & is)
	{
		return is >> request;
	}
	virtual MessageBuffer & MarshallResponse(MessageBuffer & os)
	{
		return os << response;
	}
	virtual MessageBuffer & UnmarshallResponse(MessageBuffer & is)
	{
		return is >> response;
	}
public:
	RequestT request;
	ResponseT response;
};

template<class Req1, class Req2, class ResponseT>
class ProtocolTemplate2 : public ProtocolBase
{
public:
	virtual MessageBuffer & MarshallRequest(MessageBuffer & os)
	{
		return os << req1 << req2;
	}
	virtual MessageBuffer & UnmarshallRequest(MessageBuffer & is)
	{
		return is >> req1 >> req2;
	}
	virtual MessageBuffer & MarshallResponse(MessageBuffer & os)
	{
		return os << response;
	}
	virtual MessageBuffer & UnmarshallResponse(MessageBuffer & is)
	{
		return is >> response;
	}
public:
	Req1 req1;
	Req2 req2;
	ResponseT response;
};

template<class RequestT>
class ProtocolTemplateNoResponse : public ProtocolNoResponse
{
public:
	virtual MessageBuffer & MarshallRequest(MessageBuffer & os)
	{
		return os << request;
	}
	virtual MessageBuffer & UnmarshallRequest(MessageBuffer & is)
	{
		return is >> request;
	}
public:
	RequestT request;
};

template<class Req1, class Req2>
class ProtocolTemplate2NoResponse : public ProtocolNoResponse
{
public:
	virtual MessageBuffer & MarshallRequest(MessageBuffer & os)
	{
		return os << req1 << req2;
	}
	virtual MessageBuffer & UnmarshallRequest(MessageBuffer & is)
	{
		return is >> req1 >> req2;
	}
public:
	Req1 req1;
	Req2 req2;
};

template<class R1, class R2, class R3>
class ProtocolTemplate3NoResponse : public ProtocolNoResponse
{
public:
	virtual MessageBuffer & MarshallRequest(MessageBuffer & os)
	{
		return os << req1 << req2 << req3;
	}
	virtual MessageBuffer & UnmarshallRequest(MessageBuffer & is)
	{
		return is >> req1 >> req2 >> req3;
	}
public:
	R1 req1;
	R2 req2;
	R3 req3;
};

template<class R1, class R2, class R3, class R4>
class ProtocolTemplate4NoResponse : public ProtocolNoResponse
{
public:
	virtual MessageBuffer & MarshallRequest(MessageBuffer & os)
	{
		return os << req1 << req2 << req3 << req4;
	}
	virtual MessageBuffer & UnmarshallRequest(MessageBuffer & is)
	{
		return is >> req1 >> req2 >> req3 >> req4;
	}
public:
	R1 req1;
	R2 req2;
	R3 req3;
	R4 req4;
};

} // end of namespace rpc
} // end of namespace minerva

#endif // end of head guard
