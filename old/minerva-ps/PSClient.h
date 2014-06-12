#pragma once

#include <boost/thread/locks.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>

#include <minerva/ps-minjie/AEStructs.h>

namespace minerva
{
	class PSRegValProto;
	class PSPushDeltaProto;
	class PSPullRequestProto;
	class PSPullResponseProto;
	namespace rpc { class RPCStub; }

	class PSClient
	{
		friend class PSRegValProto;
		friend class PSPushDeltaProto;
		friend class PSPullRequestProto;
		friend class PSPullResponseProto;
	public:
		PSClient(rpc::RPCStub& rpcstub);
		void Register(const AEKey& key, const AEData& data);
		void PutDelta(const AEKey& key, const AEData& data);
		AEData Pull(const AEKey& key);
	private:
		AEData WaitForPullResponse(const AEKey& key);
		void NotifyPullResponse(const AEKey& key, const AEData& data);

		rpc::RPCStub& rpcstub;

		boost::mutex pullmut;
		boost::condition_variable pullcond;
		std::map<AEKey, AEData> pullkv;
	};

} // end of namespace minerva
