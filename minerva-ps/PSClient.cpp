#include <minerva/rpc/RPCStub.h>
#include <minerva/rpc/RPCProtocol.h>
#include <minerva/ps-minjie/PSClient.h>

namespace minerva
{
	class PSRegValProto : public rpc::ProtocolTemplate2NoResponse<AEKey, AEData>
	{
	public:
		void HandleRequest(void * closure)
		{
			assert(false);
		}
		uint32_t ID()
		{
			return rpc::PS_REG_VAL;
		}
	};
	class PSPushDeltaProto : public rpc::ProtocolTemplate2NoResponse<AEKey, AEData>
	{
	public:
		void HandleRequest(void * closure)
		{
			assert(false);
		}
		uint32_t ID()
		{
			return rpc::PS_PUSH_DELTA;
		}
	};
	class PSPullRequestProto : public rpc::ProtocolTemplateNoResponse<AEKey>
	{
	public:
		void HandleRequest(void * closure)
		{
			assert(false);
		}
		uint32_t ID()
		{
			return rpc::PS_PULL_REQUEST;
		}
	};
	class PSPullResponseProto : public rpc::ProtocolTemplate2NoResponse<AEKey, AEData>
	{
	public:
		void HandleRequest(void * closure)
		{
			static_cast<PSClient*>(closure)->NotifyPullResponse(req1, req2);
		}
		uint32_t ID()
		{
			return rpc::PS_PULL_RESPONSE;
		}
	};

	PSClient::PSClient(rpc::RPCStub& rpcstub):rpcstub(rpcstub)
	{
		rpcstub.RegisterProtocol<PSRegValProto>(this);
		rpcstub.RegisterProtocol<PSPushDeltaProto>(this);
		rpcstub.RegisterProtocol<PSPullRequestProto>(this);
		rpcstub.RegisterProtocol<PSPullResponseProto>(this);
	}
	void PSClient::Register(const AEKey& key, const AEData& data)
	{
		PSRegValProto regproto;
		regproto.req1 = key;
		regproto.req2 = data;
		rpcstub.RemoteCallAsyncOutGroup("ps", regproto);
		pullkv[key] = AEData();
	}
	void PSClient::PutDelta(const AEKey& key, const AEData& data)
	{
		PSPushDeltaProto pushproto;
		pushproto.req1 = key;
		pushproto.req2 = data;
		rpcstub.RemoteCallAsyncOutGroup("ps", pushproto);
	}
	AEData PSClient::Pull(const AEKey& key)
	{
		PSPullRequestProto pullproto;
		pullproto.request = key;
		rpcstub.RemoteCallAsyncOutGroup("ps", pullproto);
		return WaitForPullResponse(key);
	}
	AEData PSClient::WaitForPullResponse(const AEKey& key)
	{
		boost::unique_lock<boost::mutex> ul(pullmut);
		while(!(pullkv[key].data))
		{
			pullcond.wait(ul);
		}
		AEData rst = pullkv[key];
		pullkv[key] = AEData();
		return rst;
	}
	void PSClient::NotifyPullResponse(const AEKey& key, const AEData& data)
	{
		boost::unique_lock<boost::mutex> ul(pullmut);
		pullkv[key] = data;
		pullcond.notify_all();
	}
}
