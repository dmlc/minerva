#ifndef MINERVA_UTILS_H
#define MINERVA_UTILS_H
#pragma once

#include <map>
#include <vector>
#include <set>
#include <string>
#include <sstream>
#include <algorithm>
#include <functional>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <time.h>
#include <cassert>

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/functional/hash.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/mutex.hpp>

#include <minerva/common/CommonTypes.h>

namespace minerva
{
namespace utils
{
    template<typename T> std::string ToString(T a);
	template<typename T1, typename T2> bool ContainInMap(const std::map<T1, T2> &m, T1 key);
	template<typename T1, typename T2> T2 * FindInMap(std::map<T1, T2* > & m, T1 key);
	template<typename K1, typename K2, typename V> V& FindInPairMap(std::map<size_t, V>& m, K1 k1, K2 k2);
	template<typename K1, typename K2, typename V> const V& FindInPairMap(const std::map<size_t, V>& m, K1 k1, K2 k2);
	template<typename K1, typename K2, typename V> bool ContainInPairMap(const std::map<size_t, V>& m, K1 k1, K2 k2);
    template<typename T> bool ContainInSet(const std::set<T> & set, T key);
    template<typename T> std::string VectorToString(const std::vector<T>& v);
    template<typename T> std::string VectorToString(T * v, size_t length);

	// thread-safe (TODO, not done yet) id generator
	template<typename IdType> class IdGenerator;

	// Calculate Cantor pairing code for given two integers
	extern unsigned long long CantorPairingFunctionUnsigned(uint64_t x, uint64_t y);
	extern unsigned long long CantorPairingFunction(int64_t x, int64_t y);

	// Execute f1 and f2 in an sequential way
	extern void Sequencer(boost::function<void(void)> f1, boost::function<void(void)> f2);
	extern M_FLOAT PerformOp(M_FLOAT lhs, M_FLOAT rhs, OPTYPE op);

	// Read/Write proxies // From StackOverflow, http://webcache.googleusercontent.com/search?q=cache:RY4AEtScD6AJ:stackoverflow.com/questions/8464458/c-pthread-how-to-make-map-access-threadsafe+&cd=1&hl=en&ct=clnk
	template<typename Mutex> class ReadGuard;
	template<typename Mutex> class WriteGuard;	
	template<typename Item, typename Mutex> class ReaderProxy;
	template<typename Item, typename Mutex> class WriterProxy;	

	// Timer class
	//class Timer;

	////////////////////////////// Impl /////////////////////////////
    template<typename T>
    std::string ToString(T a)
    {
        std::ostringstream ss;
        ss << a;
        return ss.str();
    }

	template<typename T1, typename T2>
	bool ContainInMap(const std::map<T1, T2> &m, T1 key)
	{
		return m.find(key) != m.end();
	}
	

	template<typename  T1, typename  T2> 
	T2 * FindInMap(std::map<T1, T2* >& m, T1 key)
	{
		typename std::map<T1, T2* >::iterator itr = m.find(key);
		if (itr == m.end())
		{
			return NULL;
		}
		return itr->second;
	}

	template<typename K1, typename K2, typename V> 
	V& FindInPairMap(std::map<size_t, V>& m, K1 k1, K2 k2)
	{
		boost::hash<std::pair<K1, K2> > hashFunc;
		size_t hash = hashFunc(std::pair<K1, K2>(k1, k2));
		return m[hash];
	}

	template<typename K1, typename K2, typename V> 
	const V& FindInPairMap(const std::map<size_t, V>& m, K1 k1, K2 k2)
	{
		boost::hash<std::pair<K1, K2> > hashFunc;
		size_t hash = hashFunc(std::pair<K1, K2>(k1, k2));
		return m.find(hash)->second;
	}

	template<typename K1, typename K2, typename V> 
	bool ContainInPairMap(const std::map<size_t, V>& m, K1 k1, K2 k2)
	{
		boost::hash<std::pair<K1, K2> > hashFunc;
		size_t hash = hashFunc(std::pair<K1, K2>(k1, k2));
		return ContainInMap(m, hash);
	}

    template<typename T>
    bool ContainInSet(const std::set<T> & set, T key)
    {
        typename std::set<T>::iterator itr = set.find(key);
		return itr != set.end();
    }

    template<typename T>
    std::string VectorToString(const std::vector<T> & v)
    {
        std::ostringstream ss;
		ss << "[";
        for(size_t i = 0; i < v.size(); i++)
        {
            ss << v[i] << " ";
        }
		ss << "]";
        return ss.str();
    }

    template<typename T>
    std::string VectorToString(T * v, size_t length)
    {
        std::ostringstream ss;
        for(size_t i = 0; i < length; i++)
        {
            if (i > 0)
            {
                ss << ",";
            }
            ss << v[i];
        }
        return ss.str();
    }

	template<typename IdType>
	class IdGenerator
	{
	private:
		IdType id;
		boost::mutex mut;
	public:
		IdGenerator(IdType init): id(init) {}
		IdType NewId()
		{ 
			boost::unique_lock<boost::mutex> ul(mut);
			return id++;
		}
	};

	template<typename Item, typename Mutex>
	class ReaderProxy
	{
	public:
		ReaderProxy(Item& i, Mutex& m): lock(m), item(i) {}
		Item& operator*() { return item; }
		Item* operator->() { return &item; }
	private:
		boost::shared_lock<Mutex> lock;
		Item& item;
	};
	template<typename Item, typename Mutex>
	class WriterProxy
	{
	public:
		WriterProxy(Item&i, Mutex& m): uplock(m), lock(uplock), item(i) {}
		Item& operator*() { return item; }
		Item* operator->() { return &item; }
	private:
		boost::upgrade_lock<Mutex> uplock;
		boost::upgrade_to_unique_lock<Mutex> lock;
		Item& item;
	};
}
}
#endif
