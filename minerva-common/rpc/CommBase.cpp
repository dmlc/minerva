#include <iostream>

#include <minerva/rpc/CommBase.h>
#include <minerva/rpc/Message.h>

namespace minerva
{
namespace rpc
{

	////////////////////////////////////// CommBase //////////////////////////////////////// 
CommBase::CommBase(): group(-1), _terminating(false) {}

MinervaOptions CommBase::GetOptions()
{
	MinervaOptions commopt("Communicator Options");
	commopt.AddOption<int>("comm.group", "group id of this communication group", 1);
	commopt.AddOption<int>("comm.node", "node id of this process within its communication group", 0);
	commopt.AddOption<int>("comm.numnodes", "number of processes in this communication group", 1);
	return commopt;
}
void CommBase::SetOptions(const MinervaOptions& options)
{
	rank = options.Get<int>("comm.node");
	numnodes = options.Get<int>("comm.numnodes");
	group = options.Get<int>("comm.group");
}

void CommBase::StartPolling()
{
	_terminating = false;
	// start polling thread
	_polling_thread = ThreadPtr( new Thread(boost::bind(&CommBase::PollingFunction, this)) );
}
void CommBase::ClosePolling()
{
	_terminating = true;
	// wait for polling thread to complete
	//std::cout << "Waiting for polling thread to exit" << std::endl;
	//while(!_terminated) _term_cond.wait(ul);
	_polling_thread->join();
	//std::cout << "Calling finalize" << std::endl;
	//std::cout << "Finished close polling" << std::endl;
}
bool CommBase::IsTerminating() const
{
	return _terminating;
}
void CommBase::TriggerRecvCallbacks(RecvEvent& evt)
{
	for(size_t i = 0 ; i < callbacks.size(); ++i)
		callbacks[i](evt);
}

}// end of namespace rpc
}// end of namespace minerva
