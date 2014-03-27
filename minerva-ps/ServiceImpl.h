#pragma once

#include <minerva/ps-minjie/PSService.h>
#include <minerva/ps-minjie/AETable.h>

namespace minerva
{
	class MinervaOptions;
	namespace rpc { class RPCStub; }
	class PSServiceBounded : public IPSService
	{
	public:
		PSServiceBounded(MinervaOptions& option, rpc::RPCStub* rpcstub);
		~PSServiceBounded();
		virtual void Register(const AEKey& key, const AEData& data, const std::string& from);
		virtual void PutDelta(const AEKey& key, const AEData& data, const std::string& from);
		virtual void PullRequest(const AEKey& key, const std::string& from);
		AETable& GetTable() { return table; }
	protected:
		void TrySendPullResponse(const AEKey& key, const AEData& data, 
				unsigned int curpv, unsigned int minpv, const std::string& to);
		void OnValueUpdate(const AEKey& key);
		unsigned int UpdateVersionVector(const AEKey& key); // return min version
	protected:
		typedef std::map<AEKey, int> PullVersionMap;
		//PullVersionMap vLocalPullMin;
		//std::map<std::string, PullVersionMap> vWorkerPull;
		//std::map<std::string, PullVersionMap> vWorkerPush;
		std::map<AEKey, std::set<std::string> > pendingpulls;
		std::set<std::string> workers;

		const unsigned int bound;
		rpc::RPCStub* rpcstub;
		AETable table;
		AEMonitor monitor;
	};
}
