#include <iostream>
#include <sstream>
#include <minerva/options/MinervaOptions.h>
#include <minerva/macro_def.h>
	
namespace po = boost::program_options;

namespace minerva
{
	MinervaOptions::MinervaOptions(){ }
	MinervaOptions::MinervaOptions(const std::string& name): all(name) {}
	MinervaOptions::MinervaOptions(const MinervaOptions& options)
	{
		AddOptions(options);
	}
	//MinervaOptions::MinervaOptions(const po::options_description& desc): all(desc)
	//{
	//}
	void MinervaOptions::AddHelpOption()
	{
		AddOption("help,h", "print this help");
	}
	void MinervaOptions::ExitIfHelp() const
	{
		if(Exists("help"))
		{
			std::cout << "All options:\n" << all << std::endl;
			exit(1);
		}
	}
	MinervaOptions& MinervaOptions::AddOption(const std::string& key, const std::string& desc)
	{
		all.add_options()(key.c_str(), desc.c_str());
		AddOptionHelper<void>(key.substr(0, key.find_first_of(',')));
		return *this;
	}
	MinervaOptions& MinervaOptions::AddOptions(const MinervaOptions& options)
	{
		all.add(options.all);
		helpermap.insert(options.helpermap.begin(), options.helpermap.end());
		return *this;
	}
	void MinervaOptions::ParseFromCmdLine(int argc, char** argv)
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
	void MinervaOptions::ParseFromConfigFile(const std::string& filename)
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
	std::string MinervaOptions::ToString() const
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
