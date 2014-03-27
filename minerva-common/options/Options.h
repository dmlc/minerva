#ifndef MINERVA_OPTIONS_H
#define MINERVA_OPTIONS_H

#include <cstring>
#include <fstream>
#include <map>

#include <boost/shared_ptr.hpp>
#include <boost/program_options.hpp>

namespace minerva
{
	class OptionHelper;
	class MinervaOptions
	{
	public:
		MinervaOptions();
		MinervaOptions(const std::string& name);
		MinervaOptions(const MinervaOptions& options);
		//MinervaOptions(const boost::program_options::options_description& desc);

		void AddHelpOption();
		void ExitIfHelp() const;
		MinervaOptions& AddOptions(const MinervaOptions& options);
		template<typename T> MinervaOptions& AddOption(const std::string& key, const std::string& desc, const T& defaultval);
		template<typename T> MinervaOptions& AddOption(const std::string& key, const std::string& desc);
		MinervaOptions& AddOption(const std::string& key, const std::string& desc);

		template<typename T> void Save(const std::string& key, const T& value);
		void Save(const std::string& key, const char * value);
		//void ParseFromKV(const std::string& key, const std::string& value);
		void ParseFromCmdLine(int argc, char** argv);
		void ParseFromConfigFile(const std::string& filename);

		template<typename T> const T& Get(const std::string& key) const;
		inline bool Exists(const std::string& key) const;
		std::string ToString() const;

	private:
		template<typename T> void AddOptionHelper(const std::string& key);

	private:
		boost::program_options::options_description all;
		boost::program_options::variables_map vm;
		std::map<std::string, boost::any> kvmap;
		typedef boost::shared_ptr<OptionHelper> OptionHelperPtr;
		typedef std::pair<std::string, OptionHelperPtr> OptionHelperEntry;
		std::map<std::string, OptionHelperPtr> helpermap;
	};

	class OptionHelper
	{
	public:
		virtual std::string ToString(const std::string& key, const MinervaOptions& options) const = 0;
		virtual ~OptionHelper() {}
	};

	template<typename T>
	class OptionHelperTemp : public OptionHelper
	{
	public:
		std::string ToString(const std::string& key, const MinervaOptions& options) const
		{
			std::stringstream ss;
			ss << options.Get<T>(key);
			return ss.str();
		}
	};
	
	template<>
	class OptionHelperTemp<void> : public OptionHelper
	{
	public:
		std::string ToString(const std::string& key, const MinervaOptions& options) const
		{
			std::stringstream ss;
			ss << "SET";
			return ss.str();
		}
	};

	template<typename T>
	MinervaOptions& MinervaOptions::AddOption(const std::string& key, const std::string& desc, const T& defaultval)
	{
		namespace po = boost::program_options;
		all.add_options()
			(key.c_str(), po::value<T>()->default_value(defaultval), desc.c_str());
		AddOptionHelper<T>(key.substr(0, key.find_first_of(',')));
		return *this;
	}
	template<typename T>
	MinervaOptions& MinervaOptions::AddOption(const std::string& key, const std::string& desc)
	{
		namespace po = boost::program_options;
		all.add_options()
			(key.c_str(), po::value<T>(), desc.c_str());
		AddOptionHelper<T>(key.substr(0, key.find_first_of(',')));
		return *this;
	}
	template<typename T>
	void MinervaOptions::AddOptionHelper(const std::string& key)
	{
		assert(helpermap.find(key) == helpermap.end());
		helpermap[key] = OptionHelperPtr(new OptionHelperTemp<T>);
	}

	template<typename T> void MinervaOptions::Save(const std::string& key, const T& value)
	{
		kvmap[key] = boost::any(value);
		helpermap[key] = OptionHelperPtr(new OptionHelperTemp<T>);
	}
	inline void MinervaOptions::Save(const std::string& key, const char * value)
	{
		std::stringstream ss;
		ss << value;
		Save(key, ss.str());
	}

	template<typename T>
	const T& MinervaOptions::Get(const std::string& key) const
	{
		if(kvmap.find(key) != kvmap.end())
		{
			return boost::any_cast<const T&>(kvmap.find(key)->second);
		}
		else
		{
			if(vm.count(key) == 0)
			{
				std::cout << "ERROR: \"" << key << "\" is not found in options." << std::endl;
				std::cout << all << std::endl;
				exit(1);
			}
			return vm[key].as<T>();
		}
	}
	inline bool MinervaOptions::Exists(const std::string& key) const
	{
		return vm.count(key) || kvmap.find(key) != kvmap.end();
	}

	template<class TClass>
	class Optionable
	{
	public:
		static MinervaOptions GetOptionableOptions() { return TClass::GetOptions(); }
		virtual void SetOptions(const MinervaOptions& options) = 0;
		virtual ~Optionable<TClass>() {}
	};

} // end of namespace minerva

#endif
