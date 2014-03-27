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

	static T* Instance() {
		if (!_instance)
		{
			_instance = new T();
		}
		return _instance;
	}

	static void DeleteInstance() {
		if(_instance)
			delete _instance;
		_instance = NULL;
	}
protected:
	static T * _instance;
private:
	Singleton(const Singleton&);
	Singleton& operator=(const Singleton&);
};

//template<typename T>
//T* Singleton<T>::_instance = NULL;

}
}

#endif
