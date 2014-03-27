#ifndef MINERVA_SINGLETON_H
#define MINERVA_SINGLETON_H
#pragma once

namespace minerva
{
namespace utils 
{

template<typename T>
class Singleton 
{
public:
	Singleton() { };
	virtual ~Singleton() { };

	static T& Instance()
	{
		static T instance;
		return instance;
	}
private:
	Singleton(const Singleton&);
	Singleton& operator=(const Singleton&);
};

} // end of namespace utils
} // end of namespace minerva

#endif
