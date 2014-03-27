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

#ifndef FF_COMMON_LOG_H_
#define FF_COMMON_LOG_H_
#include <minerva/logger/log/logger.h> // By Jermaine
#include <minerva/logger/log/logwriter.h> // By Jermaine
#include <minerva/logger/singlton.h> // By Jermaine

#define USING_FF_LOG 1
namespace ff
{
enum LogLevel
{
    TRACE_LEVEL,
    DEBUG_LEVEL,
    INFO_LEVEL,
    WARN_LEVEL,
    ERROR_LEVEL,
    FATAL_LEVEL,
    NUM_LOG_LEVELS
};

template<class T = LogLevel>
class log
{
public:
    inline static void init(const T & l, const char * logfile, bool verbose = true)
    {
		ll = l;
		singleton<internal::logwriter<blocking_queue<std::string> > >::instance().run(logfile, verbose);
    }
    inline static void init(const T & l, const std::string & logfile, bool verbose = true)
    {
		init(l, logfile.c_str(), verbose);
    }
    inline static void init(const std::string & lvlstr, const std::string & logfile, bool verbose = true)
	{
		LogLevel loglvl = INFO_LEVEL;
		if(lvlstr == "ERROR")
			loglvl = ERROR_LEVEL;
		else if(lvlstr == "INFO")
			loglvl = INFO_LEVEL;
		else if(lvlstr == "DEBUG")
			loglvl = DEBUG_LEVEL;
		else if(lvlstr == "TRACE")
			loglvl = TRACE_LEVEL;
		init(loglvl, logfile, verbose);
	}
public:
    static T ll;
};//end class log

template<class T>
T log<T>::ll = ERROR_LEVEL;

namespace llog
{
	template<class T>
	struct enable_traits{
		static const bool value = false;
	};//end class enable_traits;
}
}//end namespace ff


#define DEF_LOG_MODULE(module) struct log_ ##module{};

#define ENABLE_LOG_MODULE(module) \
	namespace ff { \
	namespace llog { \
		template<> struct enable_traits<log_ ##module> { \
			static const bool value = true; };  \
	}}

#undef LOG_TRACE
#undef LOG_DEBUG
#undef LOG_INFO
#undef LOG_WARN
#undef LOG_ERROR
#undef LOG_FATAL

#define FILE_NAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define LOG_LEVEL(module, level)\
	if(::ff::log<>::ll <= ::ff::level##_LEVEL) \
	ff::internal::Logger<ff::llog::enable_traits<log_ ## module>::value >()\
		<< #level << " | " << #module <<" | "\
		<< FILE_NAME <<":"<<__LINE__<<":"<<__FUNCTION__<<" |\t"

#define LOG_TRACE(module) LOG_LEVEL(module, TRACE)
#define LOG_DEBUG(module) LOG_LEVEL(module, DEBUG)
#define LOG_INFO(module)  LOG_LEVEL(module, INFO)
#define LOG_WARN(module)  LOG_LEVEL(module, WARN)
#define LOG_ERROR(module) LOG_LEVEL(module, ERROR)
#define LOG_FATAL(module) LOG_LEVEL(module, FATAL)

#define ASSERT(module, expr) while(!(expr)) LOG_FATAL(module)

#endif
