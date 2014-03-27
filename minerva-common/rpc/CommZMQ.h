#ifndef MINERVA_COMM_ZMQ_H
#define MINERVA_COMM_ZMQ_H
#pragma once

#include <minerva/rpc/CommBase.h>
#include <minerva/util/zmq.hpp>

namespace minerva
{
namespace rpc
{
	class CommGroupZMQ : public CommBase
	{
	public:
		static std::string MakeSocketId(int group, int rank);
		static std::pair<int, int> ParseSocketId(const std::string& sid);
	public:
		CommGroupZMQ(): context(NULL), receiver(NULL), outgroup(NULL) {}
		~CommGroupZMQ() {}

		static MinervaOptions GetOptions();
		virtual void SetOptions(const MinervaOptions& options);

		//void Init(MinervaOptions& options);
		void Send(SendEvent& sndevt) { sendingQueue.Push(sndevt); }
		virtual void Barrier() {}
		void Finalize();
		// other
		int NumNodes() { return sender.size(); }

	protected:
		void PollingFunction();

	private:
		std::string socketId;
		zmq::context_t* context;	// ZMQ context
		utils::BlockingQueue<SendEvent> sendingQueue;
		zmq::socket_t* receiver;    // ROUTER socket for recving
		std::vector<zmq::socket_t*> sender; // DEALER socket for sending

		bool bind;   // whether this outgroup socket is recv or send
		zmq::socket_t* outgroup;    // socket for send/recv from message out of groups
	};

} // end of namespace rpc
} // end of namespace minerva

#endif
