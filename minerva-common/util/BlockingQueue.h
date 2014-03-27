/**
 * Blocking queue data structure for multi-thread access based on boost thread library.
 * Please refer to http://maipianshuo.com/?p=135
 */
#ifndef MINERVA_BLOCKING_QUEUE_H
#define MINERVA_BLOCKING_QUEUE_H
#pragma once

#include <queue>

#include <boost/noncopyable.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>

namespace minerva
{
namespace utils
{
	template<class Ty>
	class BlockingQueue : boost::noncopyable
	{
	public:
		BlockingQueue():forcestop(false) {}

		void Push(const Ty & val)
		{
			m_oMutex.lock();
			m_oContainer.push(val);
			size_t s = m_oContainer.size();
			m_oMutex.unlock();

			if(s == 1)
			{
				m_oCond.notify_all();
			}
		}

		void Pop(Ty & val)
		{
			boost::unique_lock<boost::mutex> ul(m_oMutex);

			while(!forcestop && m_oContainer.empty())
			{
				m_oCond.wait(ul);
			}

			if(!m_oContainer.empty())
			{
				val = m_oContainer.front();
				m_oContainer.pop();
			}
		}

		size_t Size() const
		{
			boost::unique_lock<boost::mutex> ul(m_oMutex);
			return m_oContainer.size();
		}
		bool Empty() const
		{
			boost::unique_lock<boost::mutex> ul(m_oMutex);
			return m_oContainer.empty();
		}
		void ForceStop()
		{
			boost::unique_lock<boost::mutex> ul(m_oMutex);
			forcestop = true;
			m_oCond.notify_all();
		}
	protected:
		mutable boost::mutex m_oMutex;
		mutable boost::condition_variable	m_oCond;
		std::queue<Ty> m_oContainer;
		bool forcestop;
	};
}
}
#endif
