#ifndef MINERVA_CONCURRENT_MAP_H
#define MINERVA_CONCURRENT_MAP_H
#pragma once

#include <map>
#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <minerva/util/utils.h>

namespace minerva
{
namespace utils
{
	template<typename K, typename V>
	class ConcurrentMap
	{
	public:
		void Put(const K& key, const V& value)
		{
			boost::upgrade_lock<MapMutex> uplock(mapMutex);
			boost::upgrade_to_unique_lock<MapMutex> wlock(uplock);

			mapContainer[key] = value;
		}
		V& Get(const K& key)
		{
			boost::shared_lock<MapMutex> rlock(mapMutex);
			return mapContainer.find(key)->second;
		}
		const V& Get(const K& key) const
		{
			boost::shared_lock<MapMutex> rlock(mapMutex);
			return mapContainer.find(key)->second;
		}
		void Erase(const K& key)
		{
			boost::upgrade_lock<MapMutex> uplock(mapMutex);
			boost::upgrade_to_unique_lock<MapMutex> wlock(uplock);
			int r = mapContainer.erase(key);
			assert( r == 1);
		}
		bool Contains(const K& key) const
		{
			boost::shared_lock<MapMutex> rlock(mapMutex);
			return mapContainer.find(key) != mapContainer.end();
		}
		bool Empty() const
		{
			boost::shared_lock<MapMutex> rlock(mapMutex);
			return mapContainer.empty();
		}
	private:
		typedef boost::shared_mutex MapMutex;
		mutable MapMutex mapMutex;
		std::map<K, V> mapContainer;
	};
}
}

#endif
