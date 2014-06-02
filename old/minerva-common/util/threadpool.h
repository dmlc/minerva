#ifndef THREAD_POOL_H_
#define THREAD_POOL_H_

//typedef unsigned int uintptr_t;
#include <boost/cstdint.hpp>
#include <boost/thread.hpp>
#include <boost/noncopyable.hpp>
#include <boost/function.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/thread/tss.hpp>
#include <boost/atomic.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/random.hpp>
#include <boost/container/deque.hpp>
#include <vector>

namespace ff{
template<size_t MIN, class T>
class mutex_stealing_queue : public boost::noncopyable
{
public:
    mutex_stealing_queue() { }

    void push_back(const T & val)
    {
        boost::unique_lock<boost::mutex> ul(m_oMutex);
        m_oContainer.push_back(val);
    }

    bool pop(T & val)
    {
        boost::unique_lock<boost::mutex> ul(m_oMutex);

        if(m_oContainer.empty())
            return false;
        if( m_oContainer.size() > MIN &&
                m_oContainer.size() < m_oContainer.max_size()/4)
            m_oContainer.shrink_to_fit();
        val = m_oContainer.front();
        m_oContainer.pop_front();
        return true;
    }

    bool steal(T &val)
    {
        boost::unique_lock<boost::mutex> ul(m_oMutex);

        if(m_oContainer.empty())
            return false;
        val = m_oContainer.back();
        m_oContainer.pop_back();
        return true;
    }

    size_t size() const
    {
        boost::unique_lock<boost::mutex> ul(m_oMutex);
        return m_oContainer.size();
    }
    bool empty() const
    {
        boost::unique_lock<boost::mutex> ul(m_oMutex);
        return m_oContainer.empty();
    }
protected:
    mutable boost::mutex  m_oMutex;
    boost::container::deque<T> m_oContainer;
};//end class mutex_stealing_queue



template <class T>
class threadpool : public boost::noncopyable
{
public:
    typedef T task_t;
    typedef boost::function<void (task_t &)> runner_t;
    typedef mutex_stealing_queue<256, T> stealing_queue_t;
    typedef boost::shared_ptr<boost::thread> thrd_ptr;
    typedef stealing_queue_t * stealing_queue_ptr;

    template <class task_runner_t>
    threadpool(int thrd_num, const task_runner_t & f)
        : m_thrd_num(thrd_num)
        , m_func(f)
        , m_threads() , m_task_queues()
        , m_to_quit(false), m_waiting_on_join(false), m_pending_task_count(0)
        , m_gen() , m_dist(0, thrd_num) , m_die(m_gen, m_dist)
	{
        for(int i = 0; i < m_thrd_num; i++)
        {
            m_task_queues.push_back(new stealing_queue_t());
        }

        for (int i = 0; i< m_thrd_num ; i++)
        {
            m_threads.push_back(
                make_thread(
                    boost::bind(&threadpool<T>::thrd_run, this, i)));
        }
	}
	
	~threadpool()
	{
        m_to_quit.store(true);
		for(size_t i = 0; i < m_threads.size(); ++i)
			if(m_threads[i]->joinable())
			{
				m_threads[i]->join();
				//std::cout << "thread #" << i << " joined" << std::endl;
			}
		assert(m_pending_task_count == 0);
        for(size_t i = 0; i < m_task_queues.size(); ++i)
        {
            delete m_task_queues[i];
        }
		m_task_queues.clear();
		m_threads.clear();
	}

    void schedule(const T & task)
    {
		{
		boost::unique_lock<boost::mutex> ul(m_join_mut);
		++m_pending_task_count;
		}
        int id = m_die()%m_thrd_num;
        m_task_queues[id]->push_back(task);
    }

    void join()
    {
		boost::unique_lock<boost::mutex> ul(m_join_mut);
		m_waiting_on_join = true;
		while(m_pending_task_count)
			m_join_cond.wait(ul);
		m_waiting_on_join = false;
    }

protected:
	void run_task(T& task)
	{
		try
		{
			m_func(task);
		} catch (const char* ex)
		{
			std::cout << "Error: running with exceptions: " << ex << std::endl;
			assert(false);
		}
		boost::unique_lock<boost::mutex> ul(m_join_mut);
		assert(m_pending_task_count > 0);
		--m_pending_task_count;
		if(m_waiting_on_join && !m_pending_task_count)
			m_join_cond.notify_all();
	}

    void thrd_run(int id)
    {
        stealing_queue_ptr q = m_task_queues[id];
        task_t task;
        while(!m_to_quit.load() || !is_task_queue_null())
        {
            bool b = q->pop(task);
            if(b)
            {
				run_task(task);
            }
            else
			{
                b = steal_and_run(id);
            }

            if(!b)
                boost::thread::yield();
        }
    }

    bool steal_and_run(int id)
    {
        bool b =false;
        task_t task;
        for(int i = (id +1)%m_thrd_num; i!= id; i=(i+1)%m_thrd_num)
        {
            stealing_queue_ptr q = m_task_queues[i];
            b = q->pop(task);
            if(b)
            {
				run_task(task);
                return true;
            }
        }
        return false;
    }

    bool is_task_queue_null()
    {
        for(int i = 0; i< m_thrd_num; i++)
        {
            if(!m_task_queues[i]->empty())
                return false;
        }
        return true;
    }

private:
    template <class Func_t>
    thrd_ptr make_thread(const Func_t & f)
    {
        return thrd_ptr(new boost::thread(f));
    }
protected:
    int m_thrd_num;
    boost::function<void (T &)> m_func;

	// threads & queues
    std::vector<thrd_ptr> m_threads;
    std::vector<stealing_queue_ptr> m_task_queues;
	// quit mechanism
    boost::atomic<bool>	m_to_quit;
	boost::mutex m_join_mut;
	boost::condition_variable	m_join_cond;
	bool m_waiting_on_join;
	int m_pending_task_count;

    boost::mt19937 m_gen;
    boost::uniform_int<> m_dist;
    boost::variate_generator<boost::mt19937&, boost::uniform_int<> >m_die;
};//end class threadpool

}//end namespace ff;


#endif
