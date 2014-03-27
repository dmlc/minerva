#include <boost/date_time/posix_time/posix_time.hpp>

#include <minerva/ps-minjie/AETable.h>
#include <minerva/rpc/RPCProtocol.h>
#include <minerva/rpc/RPCStub.h>
#include <minerva/rpc/CommZMQ.h>
#include <minerva/logger/log.h>
#include <minerva/macro_def.h>

DEF_LOG_MODULE(AETable)
ENABLE_LOG_MODULE(AETable)
#define _INFO LOG_INFO(AETable)
#define _DEBUG LOG_DEBUG(AETable)
#define _ERROR LOG_ERROR(AETable)

namespace minerva
{

	using namespace rpc;
//#define THRESHOLD 5

	///////////////////////>>>>>>>>>>>>>>>>> AEStructs
	std::ostream& operator << (std::ostream& os, const AEKey& key)
	{
		return os << key.str;
	}

	template<class M>
	struct MetaClassUnMarshaller
	{
		static M* Exec(MessageBuffer& is)
		{
			M* rst = new M();
			is >> *rst;
			return rst;
		}
	};

	MessageBuffer& AEData::Marshall(MessageBuffer& os) const
	{
		os << meta->Id() << *meta;
		os.write(reinterpret_cast<const char*>(data), meta->Bytes());
		return os;
	}
	MessageBuffer& AEData::UnMarshall(MessageBuffer& is)
	{
		MetaClassId id;
		is >> id;
		switch(id)
		{
		case NA: std::cout << "Recv NA type meta" << std::endl; assert(false); break;
		case INT: meta = MetaClassUnMarshaller<IntegralMeta<int> >::Exec(is); break;
		case UINT: meta = MetaClassUnMarshaller<IntegralMeta<unsigned int> >::Exec(is); break;
		case FLOAT: meta = MetaClassUnMarshaller<IntegralMeta<float> >::Exec(is); break;
		case DOUBLE: meta = MetaClassUnMarshaller<IntegralMeta<double> >::Exec(is); break;
		};
		data = new char[meta->Bytes()];
		is.read(reinterpret_cast<char*>(data), meta->Bytes());
		return is;
	}
	std::string AEData::ToString() const
	{
		if(meta)
			return meta->ToString(data);
		else
			return "";
	}
	void AEData::FreeSpace()
	{
		if(meta)
		{
			meta->Free(data);
			delete meta;
		}
	}
	///////////////////////<<<<<<<<<<<<<<<<< AEStructs
	
	///////////////////////>>>>>>>>>>>>>>>>> RPC calls
	// rpc structs
	struct AEKeyWithVersion
	{
		AEKey key;
		unsigned int version;
		AEKeyWithVersion() {}
		AEKeyWithVersion(const AEKey& key, unsigned int version):
			key(key), version(version) {}
		// marshall/unmarshall
		rpc::MessageBuffer& Marshall(rpc::MessageBuffer& os) const
		{
			return os << key << version;
		}
		rpc::MessageBuffer& UnMarshall(rpc::MessageBuffer& is)
		{
			return is >> key >> version;
		}
	};
	struct AEKVDelta
	{
		AEKey key;
		AEData data;
		unsigned int low;
		unsigned int high;
		AEKVDelta() {}
		AEKVDelta(const AEKey& key, const AEData& data, unsigned int low, unsigned int high):
			key(key), data(data), low(low), high(high) {}
		// marshall/unmarshall
		rpc::MessageBuffer& Marshall(rpc::MessageBuffer& os) const
		{
			return os << key << data << low << high;
		}
		rpc::MessageBuffer& UnMarshall(rpc::MessageBuffer& is)
		{
			return is >> key >> data >> low >> high;
		}
	};
	class AERegValProto : public rpc::ProtocolTemplate3NoResponse<AEKey, AEData, unsigned int>
	{
	public:
		void HandleRequest(void * closure);
		uint32_t ID()
		{
			return rpc::AE_REG_VAL_PROTO;
		}
	};
	class AERegTableProto : public rpc::ProtocolNoData
	{
	public:
		void HandleRequest(void* closure);
		uint32_t ID()
		{
			return rpc::AE_REG_TABLE_PROTO;
		}
	};
	class AERemoteVersionProtocol : public rpc::ProtocolTemplateNoResponse<std::vector<AEKeyWithVersion> >
	{
	public:
		void HandleRequest(void* closure);
		uint32_t ID()
		{
			return rpc::AE_RECV_VERSION_PROTO;
		}
	};
	class AERemotePushProtocol : public rpc::ProtocolTemplateNoResponse<std::vector<AEKVDelta> >
	{
	public:
		void HandleRequest(void* closure);
		uint32_t ID()
		{
			return rpc::AE_RECV_UPDATES_PROTO;
		}
	};
	class AEPullUpdatesProtocol : public rpc::ProtocolTemplateNoResponse<std::vector<AEKeyWithVersion> >
	{
	public:
		void HandleRequest(void* closure);
		uint32_t ID()
		{
			return rpc::AE_PULL_UPDATES_PROTO;
		}
	};
	void AERegValProto::HandleRequest(void * closure)
	{
		AETable *table = reinterpret_cast<AETable*>(closure);
		table->OnRegisterValue(req1, req2, req3, GetSender());
		req2.FreeSpace();
	}
	void AERegTableProto::HandleRequest(void * closure)
	{
		AETable *table = reinterpret_cast<AETable*>(closure);
		table->OnRegisterTable(GetSender());
	}
	void AERemoteVersionProtocol::HandleRequest(void * closure)
	{
		AETable *table = reinterpret_cast<AETable*>(closure);
		table->OnReceiveRemoteVersion(request, GetSender());
	}
	void AERemotePushProtocol::HandleRequest(void * closure)
	{
		AETable *table = reinterpret_cast<AETable*>(closure);
		table->OnReceiveRemoteUpdates(request, GetSender());
		for(size_t i = 0; i < request.size(); ++i)
			request[i].data.FreeSpace();
	}
	void AEPullUpdatesProtocol::HandleRequest(void * closure)
	{
		AETable *table = reinterpret_cast<AETable*>(closure);
		table->OnPullUpdates(request, GetSender());
	}
	///////////////////////<<<<<<<<<<<<<<<<<<<<< RPC calls
	
	AETable::AETable(MinervaOptions& option, RPCStub* rpcstub, unsigned long syncinterval): rpcstub(rpcstub), syncinterval(syncinterval), syncrunning(false)
	{
		rpcstub->RegisterProtocol<AERegValProto>(this);
		rpcstub->RegisterProtocol<AERegTableProto>(this);
		rpcstub->RegisterProtocol<AERemoteVersionProtocol>(this);
		rpcstub->RegisterProtocol<AERemotePushProtocol>(this);
		rpcstub->RegisterProtocol<AEPullUpdatesProtocol>(this);
		syncrunning = true;
		syncthread = ThreadPtr( new Thread(boost::bind(&AETable::SyncLoop, this)) );
	}
	AETable::~AETable()
	{
		syncrunning = false;
		syncthread->join();
	}
	void AETable::RegisterTable(const std::string& id)
	{
		OnRegisterTable(id);
		AERegTableProto regproto;
		rpcstub->RemoteCall(CommGroupZMQ::ParseSocketId(id).second, regproto);
	}
	void AETable::Register(const AEKey& key, const AEData& data, unsigned int threshold)
	{
		OnRegisterValue(key, data, threshold);
	}
	void AETable::PutDelta(const AEKey& key, const AEData& data)
	{
		PutDelta(key, data.data);
	}
	void AETable::PutDelta(const AEKey& key, void* data)
	{
		_DEBUG << "Receive local updates(" << key << ") data=" << mAll[key]->ToString(data);
		ReadLock rg(rwlock); // acquire function mutex
		UpgradeLock chug(*dLocks[key]); // acquire chunk upgrade lock
		{
		UpgradeWriteLock chwg(chug); // upgrade to chunk write lock
		mAll[key]->Add(dAll[key], dAll[key], data); // dAll += d;
		++vAll[key];  // vAll = vAll + 1
		++vLocal[key];  // vLocal = vLocal + 1
		_INFO << "After accumulate local: key=" << key << " dAll=" << mAll[key]->ToString(dAll[key])
			<< " vAll=" << vAll[key] << " vLocal=" << vLocal[key];
		}
	}
	void AETable::Get(const AEKey& key, void** val, unsigned int* ver)
	{
		_DEBUG << "Reading key: (" << key << ")";
		ReadLock rg(rwlock); // acquire function mutex
		ReadLock chrg(*dLocks[key]); // acquire chunk read lock
		AEMeta* meta = mAll[key];
		if(val)
		{
			*val = meta->CloneData(dInit[key]);
			meta->Add(*val, *val, dAll[key]);
		}
		if(ver)
			*ver = vAll[key];
	}
	AEMeta* AETable::GetMeta(const AEKey& key)
	{
		ReadLock rg(rwlock); // acquire function mutex
		ReadLock chrg(*dLocks[key]); // acquire chunk read lock
		return mAll[key]->Clone();
	}
	/*unsigned int AETable::GetVersion(const AEKey& key)
	{
		ReadLock rg(rwlock); // acquire function mutex
		ReadLock chrg(*dLocks[key]); // acquire chunk read lock
		return vAll[key];
	}*/
	bool AETable::ExistKey(const AEKey& key)
	{
		ReadLock rg(rwlock); // acquire function mutex
		ReadLock chrg(*dLocks[key]); // acquire chunk read lock
		return mAll.find(key) != mAll.end();
	}

	void AETable::OnRegisterValue(const AEKey& key, const AEData& data, unsigned int threshold, const std::string& sender)
	{
		WriteLock wg(rwlock);
		_INFO << "Register chunk(" << key << ") data=" << data.meta->ToString(data.data);
		if(dAll.find(key) == dAll.end())
		{
			mAll[key] = data.meta->Clone();
			dInit[key] = data.meta->CloneData(data.data);
			dAll[key] = data.meta->NewZero();
			vAll[key] = 0;
			vLocal[key] = 0;
			dLocks[key] = new RWLock();
			foreach(std::string name, psnames)
			{
				dFrom[name][key] = data.meta->NewZero();
				dSend[name][key] = data.meta->NewZero();
				dDone[name][key] = data.meta->NewZero();
				dRecv[name][key] = data.meta->NewZero();
				vFrom[name][key] = 0;
				sHigh[name][key] = 0;
				sLow[name][key] = 0;
				rLow[name][key] = 0;
				vRequestGap[name][key] = threshold;
			}

			// Register params to other PS
			for(size_t i = 0; i < psnames.size(); ++i)
			{
				if(psnames[i] != sender)
				{
					AERegValProto initproto;
					initproto.req1 = key;
					initproto.req2 = data;
					initproto.req3 = threshold;
					rpcstub->RemoteCallAsync(CommGroupZMQ::ParseSocketId(psnames[i]).second, initproto);
				}
			}
		}
	}
	void AETable::OnRegisterTable(const std::string& psname)
	{
		WriteLock wg(rwlock);
		_INFO << "Receiving register request from " << psname;
		psnames.push_back(psname);
		dFrom[psname] = DataMap();
		dSend[psname] = DataMap();
		dDone[psname] = DataMap();
		dRecv[psname] = DataMap();
		vFrom[psname] = VersionMap();
		sHigh[psname] = VersionMap();
		sLow[psname] = VersionMap();
		rLow[psname] = VersionMap();
		vRequestGap[psname] = VersionMap();
	}
	void AETable::OnReceiveRemoteVersion(const std::vector<AEKeyWithVersion>& keyversions, const std::string& sender)
	{
		ReadLock rg(rwlock); // acquire function mutex
		AEPullUpdatesProtocol pullproto;
		for(size_t i = 0; i < keyversions.size(); ++i)
		{
			const AEKey& key = keyversions[i].key;
			unsigned int version = keyversions[i].version;

			ReadLock chrg(*dLocks[key]); // acquire chunk read lock
			unsigned int vfrom = vFrom[sender][key];
			_DEBUG << "Receive version=" << version << " vFrom=" << vfrom;
			if (version - vfrom > vRequestGap[sender][key])
			{
				//vRequestGap[sender][key] += THRESHOLD;
				//_INFO << "Version gap is " << vRequestGap[sender][key];
				// Send pull request
				pullproto.request.push_back(AEKeyWithVersion(key, vfrom));
			}
		}
		if(!pullproto.request.empty())
		{
			_DEBUG << "Requesting remote updates to " << sender;
			rpcstub->RemoteCallAsync(CommGroupZMQ::ParseSocketId(sender).second, pullproto);
		}
	}
	void AETable::OnReceiveRemoteUpdates(const std::vector<AEKVDelta>& deltas, const std::string& from)
	{
		ReadLock rg(rwlock); // acquire function mutex
		for(size_t i = 0; i < deltas.size(); ++i)
		{
			const AEKey& key = deltas[i].key;
			const AEData& data = deltas[i].data;
			unsigned int low = deltas[i].low;
			unsigned int high = deltas[i].high;

			_DEBUG << "Receiving updates (" << key << ")" << " range=[" << low << "," << high << "] data=" 
				<< data.ToString() << " from " << from;
			UpgradeLock chug(*dLocks[key]); // acquire upgrade lock
			AEMeta* meta = mAll[key];
			if(high > vFrom[from][key]) // has new version to update
			{
				if(low == rLow[from][key])
				{
					// this is an extend update
					UpgradeWriteLock chwg(chug); // upgrade to write lock
					meta->AddSub(dAll[key], dAll[key], data.data, dRecv[from][key]); // dAll = dAll + d - dRecv[j]
					meta->AddSub(dFrom[from][key], dFrom[from][key], data.data, dRecv[from][key]); // dFrom[j] = dFrom[j] + d - dRecv[j]
				}
				else
				{
					// this is a skip update
					UpgradeWriteLock chwg(chug); // upgrade to write lock
					assert(low == vFrom[from][key]);
					rLow[from][key] = low; // move version anchor
					meta->Add(dAll[key], dAll[key], data.data); // dAll = dAll + d
					meta->Add(dFrom[from][key], dFrom[from][key], data.data); // dFrom[j] = dFrom[j] + d
				}
				vAll[key] += high - vFrom[from][key]; // vAll = vAll + H - vFrom[j]
				vFrom[from][key] = high;  // vFrom[j] = H
				meta->Copy(dRecv[from][key], data.data); // record last recv
				_INFO << "After accumulate remote: key=" << key << " dAll=" << meta->ToString(dAll[key]) << " vAll=" << vAll[key];
				for(size_t j = 0; j < updatelisteners.size(); ++j)
					updatelisteners[j](key);
			}
		}
	}
	void AETable::OnPullUpdates(const std::vector<AEKeyWithVersion>& keyversions, const std::string& sender)
	{
		ReadLock rg(rwlock); // acquire function mutex
		AERemotePushProtocol pushproto;
		for(size_t i = 0; i < keyversions.size(); ++i)
		{
			const AEKey& key = keyversions[i].key;
			unsigned int version_rqst = keyversions[i].version;

			_DEBUG << "Request updates(" << key << ") v_rqst=" << version_rqst << " from " << sender;
			ReadLock chrg(*dLocks[key]); // acquire chunk read lock
			AEMeta* meta = mAll[key];
			if(version_rqst < vAll[key] - vFrom[sender][key]) // has new version to send
			{
				if(version_rqst == sHigh[sender][key])
				{
					// last send succeeds, send a skip update
					sLow[sender][key] = sHigh[sender][key];
					meta->Add(dDone[sender][key], dDone[sender][key], dSend[sender][key]); // dDone[j] = dDone[j] + dSend[j]
				}
				// extend the upper bound of the update to send
				sHigh[sender][key] = vAll[key] - vFrom[sender][key];
				meta->Sub(dSend[sender][key], dAll[key], dFrom[sender][key], dDone[sender][key]); // dSend[j] = dAll - dFrom[j] - dDone[j]
				// send update
				AEKVDelta del(key, AEData(meta, dSend[sender][key]), sLow[sender][key], sHigh[sender][key]);
				_DEBUG << "Send updates(" << del.key << ") range=[" 
					<< del.low << "," << del.high << "] data="
					<< del.data.ToString();
				pushproto.request.push_back(del);
			}
		}
		if(!pushproto.request.empty())
		{
			rpcstub->RemoteCallAsync(CommGroupZMQ::ParseSocketId(sender).second, pushproto);
		}
	}

	void AETable::SyncLoop()
	{
		while(syncrunning)
		{
			boost::this_thread::sleep(boost::posix_time::milliseconds(syncinterval));

			ReadLock rg(rwlock); // acquire function mutex
			for(size_t i = 0; i < psnames.size(); ++i)
			{
				AERemoteVersionProtocol versionproto;
				foreach(VersionEntry vent, vAll)
				{
					AEKey& key = vent.first;
					ReadLock chrg(*dLocks[key]); // acquire key read lock
					unsigned int vall = vent.second;
					versionproto.request.push_back(AEKeyWithVersion(key, vall - vFrom[psnames[i]][key]));
				}
				if(!versionproto.request.empty())
				{
					_DEBUG << "Sync all versions to " << psnames[i];
					rpcstub->RemoteCallAsync(CommGroupZMQ::ParseSocketId(psnames[i]).second, versionproto);
				}
			}
			// lock will be released after the loop
		}
	}
}// end of namespace minerva

#include <time.h>
#include <boost/chrono/chrono.hpp>

#undef _INFO
#undef _DEBUG
#undef _ERROR
DEF_LOG_MODULE(AEMonitor)
ENABLE_LOG_MODULE(AEMonitor)
#define _INFO LOG_INFO(AEMonitor)
#define _DEBUG LOG_DEBUG(AEMonitor)
#define _ERROR LOG_ERROR(AEMonitor)

namespace minerva {

	AEMonitor::AEMonitor(AETable& table): table(table), running(false) { }
	AEMonitor::~AEMonitor()
	{
		Stop();
	}
	void AEMonitor::Start()
	{
		if(!running)
		{
			running = true;
			monitorthread = ThreadPtr( new Thread(boost::bind(&AEMonitor::MonitorLoop, this)) );
		}
	}
	void AEMonitor::Stop()
	{
		if(running)
		{
			running = false;
			monitorthread->join();
		}
	}
	void AEMonitor::MonitorLoop()
	{
		using namespace boost::chrono;
		time_t curtm;
		time(&curtm);
		while(running)
		{
			++curtm;
			time_point<system_clock> tp = system_clock::from_time_t(curtm + 1);	
			boost::this_thread::sleep_until(tp);

			AETable::ReadLock rg(table.rwlock); // acquire function mutex
			foreach(AETable::VersionEntry vent, table.vAll)
			{
				AEKey& key = vent.first;
				AETable::ReadLock chrg(*table.dLocks[key]); // acquire key read lock
				unsigned int vall = vent.second;
				unsigned int vlocal = table.vLocal[key];
				_INFO << "time=" << curtm << " key=" << key << " vall=" << vall << " vlocal=" << vlocal;
			}
		}
	}

} // end of namespace minerva
