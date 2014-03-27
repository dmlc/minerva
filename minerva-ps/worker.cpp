#include <mpi.h>
#include <sys/time.h>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>

#include <minerva/rpc/RPCStub.h>
#include <minerva/rpc/CommZMQ.h>
#include <minerva/rpc/CommDummy.h>
#include <minerva/logger/log.h>
#include <minerva/ps-minjie/PSClient.h>

DEF_LOG_MODULE(Worker)
ENABLE_LOG_MODULE(Worker)
#define _INFO LOG_INFO(Worker)
#define _DEBUG LOG_DEBUG(Worker)
#define _ERROR LOG_ERROR(Worker)

using namespace minerva;
using namespace minerva::rpc;

const size_t len = 3 * 1024 * 1024;
const int numiter = 50;
const int interval = 1000000; // us
const int numpush_per_iter = 1;
const int numpull_per_iter = 1;

int wid, psid;

struct WaitObject
{
	boost::mutex mut;
	boost::condition_variable cond;
	bool flag;
	WaitObject(): flag(false) {}
	void WaitOn()
	{
		boost::unique_lock<boost::mutex> ul(mut);
		while(!flag)
			cond.wait(ul);
		flag = false;
	}
	void Notify()
	{
		boost::unique_lock<boost::mutex> ul(mut);
		flag = true;
		cond.notify_one();
	}
};
std::map<AEKey, WaitObject*> semaphores;

void PSPullResponseProto::HandleRequest(void * closure)
{
	_INFO << "Worker #" << wid << " recv key=" << req1 << " data=" << req2.ToString();
	req2.FreeSpace();
	semaphores[req1]->Notify();
}

int main (int argc, char** argv)
{
	int rank, numprocs;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	std::stringstream logstm;
	logstm << "nfs/log_w_" << rank << ".txt";

	MinervaOptions options(argc, argv);
	ff::log<>::init(options.loglevel, logstm.str(), options.verbose);
	std::stringstream ss;
	wid = rank;

	std::ifstream fin("worker.conf");
	std::string line;
	int c = 0;
	do { std::getline(fin, line); } while ( c++ != rank );
	std::vector<std::string> parts;
	boost::split(parts, line, boost::is_any_of(" "), boost::token_compress_on);
	assert(parts.size() == 2);
	psid = boost::lexical_cast<int>(parts[1]);

	ss << "workers/worker" << wid << ".txt";
	std::stringstream ss1;
	ss1 << "ipc://localps" << psid;
	CommGroupZMQ* workercomm = new CommGroupZMQ(ss.str(), ss1.str());
	workercomm->SetGroup(wid + 1);
	workercomm->SetNodeId(0);
	RPCStub* stub = new RPCStub();
	stub->SetComm(workercomm);
	stub->Init(options);
	stub->RegisterProtocol<PSRegValProto>();
	stub->RegisterProtocol<PSPushDeltaProto>();
	stub->RegisterProtocol<PSPullRequestProto>();
	stub->RegisterProtocol<PSPullResponseProto>();
	stub->StartServing();
	_INFO << "Worker #" << wid << " started, connect to PS: " << ss1.str();
	MPI_Barrier(MPI_COMM_WORLD); // initial barrier

	AEKey key("w0", 0);
	AEMeta* meta = new IntegralMeta<float>(len);
	float* array = new float[len];
	for(size_t i = 0; i < len; ++i) array[i] = 1;
	PSRegValProto regproto;
	regproto.req1 = key;
	regproto.req2 = AEData(meta, array);
	_INFO << "Worker #" << wid << " register to PS: " << ss1.str();
	stub->RemoteCallOutGroup("ps", regproto);
	semaphores[key] = new WaitObject();
	MPI_Barrier(MPI_COMM_WORLD); // register barrier

	float * delta = new float[len];
	for(size_t i = 0 ; i < len; ++i) delta[i] = 1;

	for(int i = 0; i < numiter; ++i)
	{
		_INFO << "=========== Start iter #" << i;
		for(int j = 0; j < numpush_per_iter; ++j)
		{
			_INFO << "Worker #" << wid << " push delta to PS";
			PSPushDeltaProto pushproto;
			pushproto.req1 = key;
			pushproto.req2 = AEData(meta, delta);
			stub->RemoteCallAsyncOutGroup("ps", pushproto);
			usleep(interval); // sleep for a while
		}
		for(int j = 0; j < numpull_per_iter; ++j)
		{
			PSPullRequestProto pullreqproto;
			pullreqproto.request = key;
			stub->RemoteCallAsyncOutGroup("ps", pullreqproto);
			_INFO << "Worker #" << wid << " request pull to PS";
			semaphores[key]->WaitOn();
		}
		_INFO << "=========== End iter #" << i;
	}

	MPI_Finalize();
	return 0;
}
