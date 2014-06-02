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

#ifndef MINERVA_COMMON_LOG_H
#define MINERVA_COMMON_LOG_H

#include "Logger.h"
#include "LogWriter.h"
#include <minerva-common/options/Options.h>

namespace minerva
{
namespace log
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

class Log
{
public:
    inline static void Init(const LogLevel & lvl, const char * logfile, bool verbose = true)
    {
		level = lvl;
		LogWriter::Instance().Run(logfile, verbose);
    }
    inline static void Init(const LogLevel & lvl, const std::string & logfile, bool verbose = true)
    {
		Init(lvl, logfile.c_str(), verbose);
    }
	static Options GetOptions()
	{
		Options opt("Log options");
		opt.AddOption<std::string>("log.level", "log level(ERROR, INFO, DEBUG, TRACE)", "INFO");
		opt.AddOption<std::string>("log.path", "log file path", "./log.txt");
		opt.AddOption("log.verbose", "print log in stdout");
		return opt;
	}
	static void SetOptions(const Options& options)
	{
		std::string lvlstr = options.Get<std::string>("log.level");
		std::string logpath = options.Get<std::string>("log.path");
		bool verbose = options.Exists("log.verbose");
		LogLevel loglvl = INFO_LEVEL;
		if(lvlstr == "ERROR")
			loglvl = ERROR_LEVEL;
		else if(lvlstr == "INFO")
			loglvl = INFO_LEVEL;
		else if(lvlstr == "DEBUG")
			loglvl = DEBUG_LEVEL;
		else if(lvlstr == "TRACE")
			loglvl = TRACE_LEVEL;
		Init(loglvl, logpath, verbose);
	}
public:
    static LogLevel level;
};//end class Log

LogLevel Log::level = ERROR_LEVEL;

namespace details 
{
	template<class T>
	struct enable_traits{
		static const bool value = false;
	};//end class enable_traits;
}// end of namespace details

}// end of namespace log
}// end of namespace minerva


#define DEF_LOG_MODULE(module) struct log_ ##module{};

#define ENABLE_LOG_MODULE(module) \
	namespace minerva { \
	namespace log {\
	namespace details { \
		template<> struct enable_traits<log_##module> { \
			static const bool value = true; };  \
	}}}

#define FILE_NAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define LOG_LEVEL(module, lvl)\
	if(minerva::log::Log::level <= minerva::log::lvl##_LEVEL) \
		minerva::log::Logger<minerva::log::details::enable_traits<log_##module>::value >()\
			<< #lvl << " | " << #module <<" | "\
			<< FILE_NAME <<":"<<__LINE__<<":"<<__FUNCTION__<<" |\t"

#undef LOG_TRACE
#undef LOG_DEBUG
#undef LOG_INFO
#undef LOG_WARN
#undef LOG_ERROR
#undef LOG_FATAL

#define LOG_TRACE(module) LOG_LEVEL(module, TRACE)
#define LOG_DEBUG(module) LOG_LEVEL(module, DEBUG)
#define LOG_INFO(module)  LOG_LEVEL(module, INFO)
#define LOG_WARN(module)  LOG_LEVEL(module, WARN)
#define LOG_ERROR(module) LOG_LEVEL(module, ERROR)
#define LOG_FATAL(module) LOG_LEVEL(module, FATAL)

#endif
