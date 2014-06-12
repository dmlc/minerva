#pragma once

#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/thread.hpp>
#include <minerva/rpc/Message.h>
#include <minerva/options/MinervaOptions.h>
#include <minerva/ps-minjie/AEStructs.h>

namespace minerva
{

	namespace rpc { class RPCStub; }

	class AETable;
	class AEMonitor;
	struct AEKeyWithVersion;
	struct AEKVDelta;

	class AETable
	{
		friend class AERegValProto;
		friend class AERegTableProto;
		friend class AERemoteVersionProtocol;
		friend class AERemotePushProtocol;
		friend class AEPullUpdatesProtocol;
		friend class AEMonitor;
	public:
		AETable(MinervaOptions& option, rpc::RPCStub* rpcstub, unsigned long syncinterval);
		~AETable();
		void RegisterTable(const std::string& id);
		void Register(const AEKey& key, const AEData& data, unsigned int threshold);
		void PutDelta(const AEKey& key, const AEData& data);
		void PutDelta(const AEKey& key, void* );
		template<class T> void Get(const AEKey& key, T** val, unsigned int* ver = NULL);
		void Get(const AEKey& key, void** val, unsigned int* ver = NULL);
		AEMeta* GetMeta(const AEKey& key);
		//unsigned int GetVersion(const AEKey& key);
		bool ExistKey(const AEKey& key);

		typedef boost::function<void(const AEKey&)> UpdateListener;
		void AddUpdateListener(UpdateListener listener) { updatelisteners.push_back(listener); }

	private:
	
		// remote call handlers
		void OnRegisterValue(const AEKey& key, const AEData& data, unsigned int threshold, const std::string& from = "na");
		void OnRegisterTable(const std::string& psname);
		void OnReceiveRemoteVersion(const std::vector<AEKeyWithVersion>& keyversions, const std::string& sender);
		void OnReceiveRemoteUpdates(const std::vector<AEKVDelta>& deltas, const std::string& sender);
		void OnPullUpdates(const std::vector<AEKeyWithVersion>& keyversions, const std::string& sender);

		// sync loop
		void SyncLoop();

	private:
		// rpc call stub
		rpc::RPCStub* rpcstub;

		// all states
		typedef std::map<AEKey, void*> DataMap;
		typedef std::pair<AEKey, void*> DataEntry;
		typedef std::map<AEKey, AEMeta*> MetaMap;
		typedef std::pair<AEKey, AEMeta*> MetaEntry;
		typedef std::map<AEKey, unsigned int> VersionMap;
		typedef std::pair<AEKey, unsigned int> VersionEntry;
		typedef std::map<std::string, DataMap> ProcDataMap;
		typedef std::map<std::string, VersionMap> ProcVersionMap;

		MetaMap mAll;
		DataMap dInit;
		DataMap dAll;
		VersionMap vAll;
		VersionMap vLocal;
		ProcDataMap dFrom;
		ProcDataMap dSend;
		ProcDataMap dDone;
		ProcDataMap dRecv;
		ProcVersionMap vFrom;
		ProcVersionMap sHigh;
		ProcVersionMap sLow;
		ProcVersionMap rLow;
		ProcVersionMap vRequestGap;
		std::vector<std::string> psnames;

		// thread control members
		typedef boost::shared_mutex RWLock;
		typedef boost::shared_lock<RWLock> ReadLock;
		typedef boost::upgrade_lock<RWLock> UpgradeLock;
		typedef boost::unique_lock<RWLock> WriteLock;
		typedef boost::upgrade_to_unique_lock<RWLock> UpgradeWriteLock;
		typedef std::map<AEKey, RWLock*> LockMap;

		RWLock rwlock;
		LockMap dLocks;

		// update event callbacks
		std::vector<UpdateListener> updatelisteners;

		// sync
		const unsigned long syncinterval;
		//std::vector<ProcId> synctargets;
		bool syncrunning;
		typedef boost::thread Thread;
		typedef boost::shared_ptr<Thread> ThreadPtr;
		ThreadPtr syncthread;
	};
		
	template<class T>
	void AETable::Get(const AEKey& key, T** val, unsigned int* ver)
	{
		Get(key, (void**)val, ver);
	}

	class AEMonitor
	{
	public:
		AEMonitor(AETable& table);
		~AEMonitor();
		void Start();
		void Stop();
	private:
		void MonitorLoop();
	private:
		typedef boost::thread Thread;
		typedef boost::shared_ptr<Thread> ThreadPtr;
		AETable& table;
		bool running;
		ThreadPtr monitorthread;
	};
}
