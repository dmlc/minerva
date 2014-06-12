#include <stdio.h>
#include <stdlib.h>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include <mpi.h>

#include <minerva/options/MinervaOptions.h>
#include <minerva/rpc/RPCStub.h>
#include <minerva/rpc/CommZMQ.h>
#include <minerva/logger/MinervaLog.h>
#include <minerva/ps-minjie/AETable.h>
#include <minerva/ps-minjie/ServiceImpl.h>
#include <minerva/macro_def.h>

using namespace minerva;
using namespace minerva::rpc;

int main(int argc, char** argv)
{
	int rank, numprocs;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if(argc > 1)
		rank = std::atoi(argv[1]);

	std::stringstream logstm;
	logstm << "nfs/log_ps_" << rank << ".txt";
	std::stringstream addrstm;
	addrstm << "ipc://localps" << rank;

	MinervaOptions options;
	options.AddHelpOption();
	options.AddOptions(RPCStub::GetOptions());
	options.AddOptions(MinervaLog::GetOptions());
	options.Save("log.path", logstm.str());
	options.Save("rpc.comm", "zmq");
	options.Save("comm.group", 0);
	options.Save("comm.node", rank);
	options.Save("comm.numnodes", numprocs);
	options.Save("zmq.config", "ps.txt");
	options.ParseFromCmdLine(argc, argv);
	std::cout << "Configs:\n" << options.ToString() << std::endl;
	options.ExitIfHelp();

	MinervaLog::SetOptions(options);
	// rpcstub to master
	//rpc::CommGroupZMQ* comm = new rpc::CommGroupZMQ("ps.txt", ss.str(), true);
	//CommGroupZMQ* comm = new CommGroupZMQ();
	//comm->SetOptions(options);
	RPCStub *rpc = new RPCStub();
	rpc->SetOptions(options);

	PSServiceBounded service(options, rpc);
	rpc->StartServing();
	AETable& table = service.GetTable();

	std::ifstream fin("server.conf");
	std::string line;
	int c = 0;
	do { std::getline(fin, line); } while ( c++ != rank );
	std::vector<std::string> parts;
	boost::split(parts, line, boost::is_any_of(" "), boost::token_compress_on);
	foreach(std::string s, parts)
	{
		int id = boost::lexical_cast<int>(s);
		if(id == rank) continue;
		std::string tableid = CommGroupZMQ::MakeSocketId(0, id);
		std::cout << "Register table: " << tableid << std::endl;
		table.RegisterTable(tableid);
	}
	std::cout << "Slave PS #" << rank << " started" << std::endl;
	MPI_Barrier(MPI_COMM_WORLD);
	while(1);
//	getchar();
	// Register initial weight
	/*float* array = new float[600];
	float* delta = new float[600];
	for(int i = 0; i < 600; ++i) array[i] = delta[i] = 1;

	table.Register(key, AEData(meta, array));

	int numpushes = 15;
	for(int i = 0; i < numpushes; ++i)
	{
		std::cout << "Slave PS #" << options.nodeId << " push delta to table" << std::endl;
		table.PutDelta(key, delta);
	}*/

	/*AEKey key("w0", 0);
	AEMeta* meta = table.GetMeta(key);
	float* rst;
	table.Get(key, &rst);
	std::cout << "Slave PS #" << options.nodeId << " got " << meta->ToString(rst) << std::endl;
	int * vrst;
	table.Get(AEKey("pv_", key), &vrst);
	std::cout << "[" << vrst[0] << " " << vrst[1] << " " << vrst[2] << "]" << std::endl;
	*/
	MPI_Finalize();
	rpc->Destroy();
	return 0;
}
