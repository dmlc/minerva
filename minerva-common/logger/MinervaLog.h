#pragma once
#include <minerva/logger/log.h>
#include <minerva/options/MinervaOptions.h>

namespace minerva
{
	class MinervaLog
	{
	public:
		static MinervaOptions GetOptions()
		{
			MinervaOptions opt("Log options");
			opt.AddOption<std::string>("log.level", "log level(ERROR, INFO, DEBUG, TRACE)", "INFO");
			opt.AddOption<std::string>("log.path", "log file path", "./log.txt");
			opt.AddOption("log.verbose", "print log in stdout");
			return opt;
		}

		static void SetOptions(const MinervaOptions& options)
		{
			std::string lvlstr = options.Get<std::string>("log.level");
			std::string logpath = options.Get<std::string>("log.path");
			bool verbose = options.Exists("log.verbose");
			ff::log<>::init(lvlstr, logpath, verbose);
		}
	};
}
