#ifndef MINERVA_RPC_TYPES_H
#define MINERVA_RPC_TYPES_H
#pragma once

#include <stdint.h>

#include <boost/thread/locks.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>

namespace minerva
{
	// Communication layer
	typedef int32_t ProcId;
	const std::string PARAM_SERVER_ID = "ps";

namespace rpc
{
	typedef boost::mutex Mutex;
	typedef Mutex Lock;
	typedef boost::condition_variable CondVar;
	typedef boost::unique_lock<Mutex> ScopedUniqueLock;
	typedef boost::thread Thread;
	typedef boost::shared_ptr<Thread> ThreadPtr;
	typedef boost::thread_group ThreadGroup;

	// Forward declaration
	class MessageBuffer;
	//typedef boost::shared_ptr<MessageBuffer> MessagePtr;
//#define DeleteMessagePtr(ptr)
	typedef MessageBuffer* MessagePtr;
#define DeleteMessagePtr(ptr) delete ptr;
}
}

#endif
