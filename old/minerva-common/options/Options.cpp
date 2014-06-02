#include <iostream>
#include <sstream>

#include "Options.h"
#include <minerva-common/macro_def.h>
	
namespace po = boost::program_options;

namespace minerva
{
	Options::Options(){ }
	Options::Options(const std::string& name): all(name) {}
	Options::Options(const Options& options)
	{
		AddOptions(options);
	}
	//Options::Options(const po::options_description& desc): all(desc)
	//{
	//}
	void Options::AddHelpOption()
	{
		AddOption("help,h", "print this help");
	}
	void Options::ExitIfHelp() const
	{
		if(Exists("help"))
		{
			std::cout << "All options:\n" << all << std::endl;
			exit(1);
		}
	}
	Options& Options::AddOption(const std::string& key, const std::string& desc)
	{
		all.add_options()(key.c_str(), desc.c_str());
		AddOptionHelper<void>(key.substr(0, key.find_first_of(',')));
		return *this;
	}
	Options& Options::AddOptions(const Options& options)
	{
		all.add(options.all);
		helpermap.insert(options.helpermap.begin(), options.helpermap.end());
		return *this;
	}
	void Options::ParseFromCmdLine(int argc, char** argv)
	{
		try
		{
			po::parsed_options parsed = po::command_line_parser(argc, argv).options(all)
				.allow_unregistered().run();
			po::store(parsed, vm);
			po::notify(vm);
		}
		catch( po::error err )
		{
			std::cout << "Option parsing error:\n\t" << err.what() << "\n" << std::endl;
			std::cout << all << std::endl;
		}
	}
	void Options::ParseFromConfigFile(const std::string& filename)
	{
		try
		{
			po::parsed_options parsed = po::parse_config_file<char>(filename.c_str(), all, true);
			po::store(parsed, vm);
			po::notify(vm);
		}
		catch( po::error err )
		{
			std::cout << "Option parsing error:\n\t" << err.what() << "\n" << std::endl;
			std::cout << all << std::endl;
		}
	}
	std::string Options::ToString() const
	{
		std::stringstream ss;
		foreach(OptionHelperEntry ent, helpermap)
		{
			if(Exists(ent.first))
				ss << ent.first << "=" << ent.second->ToString(ent.first, *this) << std::endl;
			else
				ss << ent.first << "=NOT_SET" << std::endl;
		}
		return ss.str();
	}
}
