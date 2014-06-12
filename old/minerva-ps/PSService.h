#pragma once
#include <minerva/ps-minjie/AEStructs.h>
#include <minerva/rpc/RPCProtocol.h>

namespace minerva
{
	class PSRegValProto;
	class PSPushDeltaProto;
	class PSPullRequestProto;
	class PSPullResponseProto;

	class IPSService
	{
	public:
		virtual void Register(const AEKey& key, const AEData& data, const std::string& from) = 0;
		virtual void PutDelta(const AEKey& key, const AEData& data, const std::string& from) = 0;
		virtual void PullRequest(const AEKey& key, const std::string& from) = 0;
	};
	class PSRegValProto : public rpc::ProtocolTemplate2NoResponse<AEKey, AEData>
	{
	public:
		void HandleRequest(void * closure)
		{
			IPSService * service = (IPSService*)closure;
			service->Register(req1, req2, GetSender());
			req2.FreeSpace();
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
			IPSService * service = (IPSService*)closure;
			service->PutDelta(req1, req2, GetSender());
			req2.FreeSpace();
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
			IPSService * service = (IPSService*)closure;
			service->PullRequest(request, GetSender());
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
			assert(false); // this won't be called on server side
		}
		uint32_t ID()
		{
			return rpc::PS_PULL_RESPONSE;
		}
	};

	
}
