#include <fstream>

#include <minerva-common/logger/log.h>
#include <minerva-common/util/FileParsers.h>
#include <minerva-common/macro_def.h>

#include "Message.h"
#include "CommZMQ.h"

DEF_LOG_MODULE(CommZMQ)
ENABLE_LOG_MODULE(CommZMQ)
#define _INFO LOG_INFO(CommZMQ)
#define _DEBUG LOG_DEBUG(CommZMQ)
#define _ERROR LOG_ERROR(CommZMQ)
#define _TRACE LOG_TRACE(CommZMQ)

namespace minerva
{
namespace rpc
{
	Options CommGroupZMQ::GetOptions()
	{
		Options zmqopt("ZMQ options");
		zmqopt.AddOption<std::string>("zmq.config", "configure file for ZMQ communication group", "./zmq_group.cfg");
		zmqopt.AddOption<int>("zmq.numthreads", "number of I/O threads used by ZMQ context", 1);
		zmqopt.AddOption<int>("zmq.linger", "linger time of ZMQ context", 5000);
		return zmqopt;
	}

	void CommGroupZMQ::SetOptions(const Options& options)
	{
		assert(context == NULL); // assert the context is not created

		CommBase::SetOptions(options);

		socketId = MakeSocketId(group, rank);
		std::cout << "socketId=" << socketId << std::endl;

		std::string cfgfilename = options.Get<std::string>("zmq.config");
		int nthrs = options.Get<int>("zmq.numthreads");
		int linger = options.Get<int>("zmq.linger");

		context = new zmq::context_t(nthrs);
		utils::TableParser parser(cfgfilename);
		assert(parser.Parse());
		assert(rank < (int)parser.NumRows());

		for(size_t r = 0; r < parser.NumRows(); ++r)
		{
			assert(parser.NumCols(r) >= 3);
			std::string peeraddr = parser.Get<std::string>(r, 0);
			std::string outaddr = parser.Get<std::string>(r, 1);
			bind = parser.Get<bool>(r, 2);

			zmq::socket_t* sck = NULL;
			if(r == (size_t)rank)
			{
				receiver = new zmq::socket_t(*context, ZMQ_ROUTER);
				receiver->setsockopt(ZMQ_IDENTITY, socketId.c_str(), socketId.length());
				receiver->setsockopt(ZMQ_LINGER, &linger, sizeof(linger));
				receiver->bind(peeraddr.c_str());
				std::cout << "Bind addr=" << peeraddr << std::endl;
				if(bind)
				{
					outgroup = new zmq::socket_t(*context, ZMQ_ROUTER);
					outgroup->setsockopt(ZMQ_IDENTITY, socketId.c_str(), socketId.length());
					outgroup->setsockopt(ZMQ_LINGER, &linger, sizeof(linger));
					outgroup->bind(outaddr.c_str());
					//std::cout << "Bind on " << outaddr << std::endl;
				}
				else
				{
					outgroup = new zmq::socket_t(*context, ZMQ_DEALER);
					outgroup->setsockopt(ZMQ_IDENTITY, socketId.c_str(), socketId.length());
					outgroup->setsockopt(ZMQ_LINGER, &linger, sizeof(linger));
					outgroup->connect(outaddr.c_str());
					//std::cout << "Connect to " << outaddr << std::endl;
				}
			}
			else
			{
				sck = new zmq::socket_t(*context, ZMQ_DEALER);
				sck->setsockopt(ZMQ_IDENTITY, socketId.c_str(), socketId.length());
				sck->setsockopt(ZMQ_LINGER, &linger, sizeof(linger));
				sck->connect(peeraddr.c_str());
				std::cout << "Connect addr=" << peeraddr << std::endl;
			}
			sender.push_back(sck);
		}
	}
	void CommGroupZMQ::Finalize()
	{
		//_TRACE << "Deleting socket";
		delete receiver;
		foreach(zmq::socket_t* sck, sender)
			if(sck)
				delete sck;
		if(outgroup)
			delete outgroup;
		//_TRACE << "Deleting context";
		delete context;
		//_TRACE << "Finalizing ZMQ worker comm complete";
	}
	
	/*struct MessagePtrWrap
	{
		MessagePtr ptr;
		MessagePtrWrap(MessagePtr ptr): ptr(ptr) {}
	};*/

	void FreeMessagePtr(void* data, void* arg)
	{
		MessagePtr msgptr = (MessagePtr) arg;
		delete msgptr;
		//MessagePtrWrap* wrapptr = (MessagePtrWrap*)arg;
		//delete wrapptr;
	}

	void SendBody(zmq::socket_t* sck, MessagePtr sendmsg)
	{			
		assert(sck);
		size_t numrawbufs = sendmsg->get_num_raw_buf();
		// 1. first message part is the proto buffer
		zmq::message_t mainbody(sendmsg->get_buf(), sendmsg->gsize(), &FreeMessagePtr, sendmsg);
		size_t r = sck->send(mainbody, numrawbufs > 0? ZMQ_SNDMORE : 0);
		assert(r);
		//std::cout << "Send retval=" << r << std::endl;
		// 2. [optional] raw buffers
		for(size_t i = 0; i < numrawbufs; ++i)
		{
			RawBuf& rawbuf = sendmsg->get_raw_bufs()[i];
			assert(rawbuf.type == RawBuf::RAW_BUF_CPU);
			zmq::message_t msg(rawbuf.buf, rawbuf.size, rawbuf.freefn, rawbuf.arg);
			sck->send(msg, i == numrawbufs - 1? 0 : ZMQ_SNDMORE);
		}
	}
	void RecvHead(zmq::socket_t* sck, zmq::message_t& msg, RecvEvent& evt)
	{
		std::string sckid((char*)msg.data(), msg.size());
		//msgbuf->set_sender_id(sckid);
		int remotegroup = 0, remoterank = 0;
		boost::tie(remotegroup, remoterank) = CommGroupZMQ::ParseSocketId(sckid);
		evt.group = remotegroup;
		evt.rank = remoterank;
		evt.id = sckid;
	}
	void RecvBody(zmq::socket_t* sck, zmq::message_t& msg, RecvEvent& evt)
	{
		MessagePtr msgbuf( new MessageBuffer() );
		// first message part gives the proto buffer
		char* buf = new char[msg.size()];
		memcpy(buf, msg.data(), msg.size());
		msgbuf->set_buf(buf, msg.size());
		// [optional] raw buffers
		while(msg.more())
		{
			msg.rebuild();
			assert(sck->recv(&msg));
			char* rawbuf = new char[msg.size()];
			memcpy(rawbuf, msg.data(), msg.size());
			msgbuf->push_raw_buf(rawbuf, msg.size(), rpc::RawBuf::RAW_BUF_CPU);
		}
		evt.msg = msgbuf;
	}
	void CommGroupZMQ::PollingFunction()
	{
		unsigned long long recvbytes = 0;
		zmq::message_t msg;
		SendEvent sndevt;
		while(!IsTerminating())
		{
			msg.rebuild();
			// Receiver intra-group message
			if(receiver->recv(&msg, ZMQ_DONTWAIT))
			{
				recvbytes += msg.size();
				RecvEvent evt;
				RecvHead(receiver, msg, evt);
				assert(evt.group == group);
				assert(msg.more());
				receiver->recv(&msg);
				recvbytes += msg.size();
				RecvBody(receiver, msg, evt);
				TriggerRecvCallbacks(evt);
				//std::cout << "Recv inner group event: " << evt.group << " " << evt.rank << " " << evt.id << std::endl;
				_TRACE << "Recv total bytes: " << recvbytes;
			}
			// Receiver inter-group message
			if(outgroup && outgroup->recv(&msg, ZMQ_DONTWAIT))
			{
				RecvEvent evt;
				if(bind)
				{
					RecvHead(outgroup, msg, evt);
					assert(msg.more());
					outgroup->recv(&msg);
				}
				else
				{
					evt.group = -1;
					evt.rank = -1;
					evt.id = "outgroup";
				}
				//std::cout << "Recv outgroup event: " << evt.group << " " << evt.rank << " " << evt.id << std::endl;
				RecvBody(outgroup, msg, evt);
				TriggerRecvCallbacks(evt);
			}
			// Do one send
			if(!sendingQueue.Empty())
			{
				sendingQueue.Pop(sndevt);
				//std::cout << "Send event: " << sndevt.group << " " << sndevt.rank << " " << sndevt.id << std::endl;
				if(sndevt.group == group) // intra-group message
				{
					SendBody(sender[sndevt.rank], sndevt.msg);
				}
				else // inter-group message
				{
					if(bind)
					{
						// sending through ROUTER, need to add an identifier message
						outgroup->send(sndevt.id.c_str(), sndevt.id.length(), ZMQ_SNDMORE);
					}
					SendBody(outgroup, sndevt.msg);
				}
			}
		}
	}

	std::string CommGroupZMQ::MakeSocketId(int group, int rank)
	{
		std::stringstream ss;
		ss << "x" << std::setw(2) << std::setfill('0') << group 
			<< std::setw(2) << std::setfill('0') << rank;
		return ss.str();
	}
	std::pair<int, int> CommGroupZMQ::ParseSocketId(const std::string& sid)
	{
		assert(sid.length() == 5);
		assert(sid[0] == 'x');
		int group = atoi(sid.substr(1, 2).c_str());
		int rank = atoi(sid.substr(3, 2).c_str());
		return std::make_pair(group, rank);
	}

} // end of namespace rpc
} // end of namespace minerva
