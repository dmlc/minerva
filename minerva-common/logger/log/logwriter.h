/*
   The MIT License (MIT)
   Copyright (c) 2013 <AthrunArthur>
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
   */
#ifndef FFNET_COMMON_LOG_LOG_WRITER_H_
#define FFNET_COMMON_LOG_LOG_WRITER_H_
#include <memory>
#include <fstream>
#include <minerva/logger/blocking_queue.h> // By Jermaine
#include <string>
#include <minerva/logger/singlton.h>// By Jermaine

#if __cplusplus < 201103L
#include <boost/thread/mutex.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#else
#include <mutex>
#include <thread>
#include <functional>
#endif

#include <iostream>

namespace ff
{
namespace internal
{
	//Container
	template<class T = ff::blocking_queue<std::string> >
	class logwriter
	{
	protected:
		friend class ff::singleton<logwriter<T> > ;
		logwriter()
			: m_strFilePath()
			  , m_bRunning(true) {};
	public:
		virtual ~logwriter()
		{
			m_oMutex.lock();
			m_oQueue.push_back("End and quit log!");
			m_bRunning = false;
			m_oMutex.unlock();

			m_pIOThread->join();
		}
		T &	queue()
		{
			return m_oQueue;
		}
		void run(const char * filePath, bool verbose)
		{
			m_strFilePath = std::string(filePath);
			m_verbose = verbose;
			if(m_pIOThread)
				return;
#if __cplusplus < 201103L
			m_pIOThread = boost::shared_ptr<boost::thread>(new boost::thread(boost::bind(&logwriter<T>::actualRun, this)));
#else	
			m_pIOThread = std::make_shared<std::thread>([this](){
					this->actualRun();});
#endif
		}
		void flush(std::string str)
		{
			m_oFile<<str<<std::endl;
			if(m_verbose)
				std::cout << str << std::endl;
		}
	protected:
		void actualRun()
		{
			std::string str;
			m_oFile.open(m_strFilePath.c_str() );

			m_oMutex.lock();
			while(m_bRunning || !m_oQueue.empty())
			{
				m_oMutex.unlock();
				size_t t = m_oQueue.size();
				while(t!=0)
				{
					m_oQueue.pop(str);
					m_oFile<<str<<std::endl;
					if(m_verbose)
						std::cout << str << std::endl;
					t--;
				}
				m_oFile.flush();
				m_oMutex.lock();
			}
			m_oMutex.unlock();
			m_oFile.close();
		}
	protected:
		T		m_oQueue;
#if __cplusplus < 201103L
		boost::shared_ptr<boost::thread>		m_pIOThread;
		boost::mutex				m_oMutex;
#else
		std::shared_ptr<std::thread>		m_pIOThread;
		std::mutex					m_oMutex;
#endif
		std::string					m_strFilePath;
		bool						m_verbose;
		std::ofstream				m_oFile;
		bool					m_bRunning;
	};//end class LogWriter
}//end namespace internal
}//end namespace ffnet
#endif
