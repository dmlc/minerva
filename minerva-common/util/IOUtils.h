#pragma once

#include <cstring>
#include <sstream>
#include <iomanip>

namespace minerva
{
	inline std::string MakeIndexKey(int index)
	{
		std::stringstream ss;
		ss << std::setw(8) << std::setfill('0') << std::right << index;
		return ss.str();
	}

	inline int ParseIndexKey(const std::string& key)
	{
		int ret = 0;
		std::stringstream ss(key);
		ss >> ret;
		return ret;
	}

	inline std::string MakeLabelIndexKey(int index)
	{
		std::stringstream ss;
		ss << "label" << std::setw(8) << std::setfill('0') << std::right << index;
		return ss.str();
	}

	inline int ParseLabelIndexKey(const std::string& key)
	{
		int ret = 0;
		std::stringstream ss(key.substr(5, key.length() - 5));
		ss >> ret;
		return ret;
	}
}
