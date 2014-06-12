#pragma once

#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#define PRINT_ERROR(filename, errormsg)\
	std::cout << "Error occurs when parsing: " << (filename) << std::endl; \
	std::cout << "Error msg: " << (errormsg) << std::endl;

namespace minerva
{
namespace utils
{
	class TableParser
	{
	public:
		TableParser(const std::string& filename): filename(filename) {}
		bool Parse()
		{
			namespace fs = boost::filesystem;
			namespace alg = boost::algorithm;

			fs::path file(filename);
			try
			{
				if(!fs::is_regular_file(file) || !fs::exists(file))
				{
					PRINT_ERROR(filename, "file is not readable");
					return false;
				}
				std::ifstream fin(filename.c_str());
				std::string line;
				while(std::getline(fin, line))
				{
					line = line.substr(0, line.find_first_of('#')); // remove comments
					alg::trim(line); // remove white chars on both sides
					if(!line.empty())
					{
						tablevalues.push_back(std::vector<std::string>());
						std::istringstream iss(line);
						std::string tok;
						while(iss >> tok)
						{
							tablevalues.back().push_back(tok);
						}
					}
				}
			}
			catch (const fs::filesystem_error& ex)
			{
				PRINT_ERROR(filename, ex.what());
				return false;
			}

			return true;
		}

		template<typename T> T Get(size_t row, size_t col)
		{
			if(row >= tablevalues.size() || col >= tablevalues[row].size())
			{
				std::cout << "(" << row << "," << col << ") is not a valid index" << std::endl;
				return T();
			}
			else
			{
				try
				{
					T ret = boost::lexical_cast<T>(tablevalues[row][col]);
					return ret;
				}
				catch(boost::bad_lexical_cast&)
				{
					std::cout << "Error when cast \"" << tablevalues[row][col] << "\" to target type" << std::endl;
					exit(1);
				}
			}
		}
		size_t NumRows() const { return tablevalues.size(); }
		size_t NumCols(size_t row) const { return tablevalues[row].size(); }
	private:
		std::string filename;
		std::vector<std::vector<std::string> > tablevalues;
	};
} // end of namespace utils
} // end of namespace minerva
#undef PRINT_ERROR
