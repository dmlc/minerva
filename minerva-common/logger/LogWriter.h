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
#ifndef MINERVA_COMMON_LOG_WRITER_H
#define MINERVA_COMMON_LOG_WRITER_H

#include <memory>
#include <fstream>
#include <string>
#include <iostream>

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

#include <minerva-common/util/BlockingQueue.h>
#include <minerva-common/util/Singleton.h>

namespace minerva
{
namespace log
{
	//Container
	class LogWriter : public utils::Singleton<LogWriter>
	{
	protected:
		typedef utils::BlockingQueue<std::string> LogQueue;
		friend class utils::Singleton<LogWriter> ;
		LogWriter(): m_strFilePath(), m_bRunning(true) {}
	public:
		virtual ~LogWriter()
		{
			m_oMutex.lock();
			m_oQueue.Push("End and quit log!");
			m_bRunning = false;
			m_oMutex.unlock();

			m_pIOThread->join();
		}
		LogQueue & GetQueue()
		{
			return m_oQueue;
		}
		void Run(const char * filePath, bool verbose)
		{
			m_strFilePath = std::string(filePath);
			m_verbose = verbose;
			if(m_pIOThread)
				return;
#if __cplusplus < 201103L
			m_pIOThread = boost::shared_ptr<boost::thread>(
					new boost::thread(boost::bind(&LogWriter::ActualRun, this)));
#else
			m_pIOThread = std::make_shared<std::thread>([this](){
					this->ActualRun();});
#endif
		}
		void Flush(std::string str)
		{
			m_oFile << str << std::endl;
			if(m_verbose)
				std::cout << str << std::endl;
		}
	protected:
		void ActualRun()
		{
			std::string str;
			m_oFile.open(m_strFilePath.c_str() );

			m_oMutex.lock();
			while(m_bRunning || !m_oQueue.Empty())
			{
				m_oMutex.unlock();
				size_t t = m_oQueue.Size();
				while(t!=0)
				{
					m_oQueue.Pop(str);
					m_oFile << str << std::endl;
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
		LogQueue m_oQueue;
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

}//end namespace log
}//end namespace minerva
#endif
