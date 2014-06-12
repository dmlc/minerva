#include <boost/bind.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include <minerva/options/MinervaOptions.h>
#include <minerva/ps-minjie/ServiceImpl.h>
#include <minerva/rpc/RPCStub.h>
#include <minerva/logger/log.h>
#include <minerva/macro_def.h>

DEF_LOG_MODULE(PSBounded)
ENABLE_LOG_MODULE(PSBounded)
#define _INFO LOG_INFO(PSBounded)
#define _DEBUG LOG_DEBUG(PSBounded)
#define _ERROR LOG_ERROR(PSBounded)

//#define USE_BOUND

namespace minerva
{
	PSServiceBounded::PSServiceBounded(MinervaOptions& option, rpc::RPCStub* rpcstub): 
		bound(5), rpcstub(rpcstub), table(option, rpcstub, 5000), monitor(table) // 1000 milliseconds
	{
		rpcstub->RegisterProtocol<PSRegValProto>(this);
		rpcstub->RegisterProtocol<PSPushDeltaProto>(this);
		rpcstub->RegisterProtocol<PSPullRequestProto>(this);
		rpcstub->RegisterProtocol<PSPullResponseProto>(this);
		table.AddUpdateListener(boost::bind(&PSServiceBounded::OnValueUpdate, this, _1));
		//monitor.Start();
	}
	PSServiceBounded::~PSServiceBounded()
	{
		//monitor.Stop();
	}
	void PSServiceBounded::Register(const AEKey& key, const AEData& data, const std::string& from)
	{
		if(pendingpulls.find(key) == pendingpulls.end()) // new data to register
		{
			// Register initial data
			table.Register(key, data, 0);
#ifdef USE_BOUND
			// Register initial pull version
			int * pvinit = new int[rpcstub->NumNodes()];
			for(int i = 0; i < rpcstub->NumNodes(); ++i) pvinit[i] = 0;
			AEKey pvkey("pv_", key);
			IntegralMeta<int> pvmeta(rpcstub->NumNodes());
			table.Register(pvkey, AEData(&pvmeta, pvinit), 0);
			delete [] pvinit;
#endif
			pendingpulls[key] = std::set<std::string>();
		}
		// Record the worker
		if(workers.find(from) == workers.end()) // new worker appears
		{
			_INFO << "Hear from new worker: " << from;
			workers.insert(from);
		}
	}
	void PSServiceBounded::PutDelta(const AEKey& key, const AEData& data, const std::string& from)
	{
		table.PutDelta(key, data);
#ifdef USE_BOUND
		UpdateVersionVector(key);
#endif
	}
	void PSServiceBounded::PullRequest(const AEKey& key, const std::string& from)
	{
		AEData senddata;
		senddata.meta = table.GetMeta(key);
		// get current version
		unsigned int curpv = 0;
		table.Get(key, &senddata.data, &curpv);
		// get minimum version
#ifdef USE_BOUND
		unsigned int minpv = UpdateVersionVector(key);
#else
		unsigned int minpv = curpv;
#endif
		// try send pull response
		TrySendPullResponse(key, senddata, curpv, minpv, from);
		senddata.FreeSpace();
	}
	unsigned int PSServiceBounded::UpdateVersionVector(const AEKey& key)
	{
#ifdef USE_BOUND
		AEKey pvkey("pv_", key);
		int * serverpv;
		table.Get(pvkey, &serverpv);
		int minpv = INT_MAX;
		for(int i = 0; i < rpcstub->NumNodes(); ++i) minpv = std::min(minpv, serverpv[i]);
		int oldpv = serverpv[rpcstub->NodeId()];
		unsigned int curpv = 0;
		table.Get(key, NULL, &curpv);
		assert(curpv >= oldpv);
		if(curpv > oldpv)
		{
			memset(serverpv, 0, sizeof(int) * rpcstub->NumNodes());
			serverpv[rpcstub->NodeId()] = curpv - oldpv;
			table.PutDelta(pvkey, serverpv);
		}
		delete [] serverpv;
		return minpv;
#else
		assert(false);
		return 0;
#endif
	}
	void PSServiceBounded::TrySendPullResponse(const AEKey& key, const AEData& data, 
			unsigned int curpv, unsigned int minpv, const std::string& to)
	{
#ifdef USE_BOUND
		assert(curpv >= minpv);
		if(curpv - minpv > bound)
		{
			_INFO << "pull key=" << key << " curpv=" << curpv << " minpv=" << minpv;
			_INFO << "Response to pull key=" << key << " to " << to << " need to be suspended";
			pendingpulls[key].insert(to);
		}
		else
#endif
		{
			_INFO << "Response to pull key=" << key << " to " << to << " curpv=" << curpv << " data=" << data.ToString();
			PSPullResponseProto proto;
			proto.req1 = key;
			proto.req2 = data;
			rpcstub->RemoteCallAsyncOutGroup(to, proto);
		}
	}
	void PSServiceBounded::OnValueUpdate(const AEKey& key)
	{
#ifdef USE_BOUND
		if(boost::starts_with(key.ToString(), "pv_")) // hear from version update
		{
			size_t keylen = key.ToString().length();
			AEKey datakey(key.ToString().substr(3, keylen - 3));
			_INFO << "Hear version update key=" << datakey << " #pending=" << pendingpulls[datakey].size();
			unsigned int minpv = UpdateVersionVector(datakey);
			if(!pendingpulls[datakey].empty())
			{
				std::set<std::string> pendingworkers = pendingpulls[datakey];
				pendingpulls[datakey].clear();

				AEData senddata;
				senddata.meta = table.GetMeta(datakey);
				unsigned int ver = 0;
				table.Get(datakey, &senddata.data, &ver);
				foreach(std::string wname, pendingworkers)
				{
					TrySendPullResponse(datakey, senddata, ver, minpv, wname);
				}
				senddata.FreeSpace();
			}
		}
		else // hear from data update
		{
			UpdateVersionVector(key);
		}
#endif
	}
}
