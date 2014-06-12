#ifndef MINERVA_PAIR_KEY_MAP_H
#define MINERVA_PAIR_KEY_MAP_H
#pragma once

#include <map>
#include <sstream>

namespace minerva
{
namespace utils
{

	template<typename K1, typename K2, typename V>
	class PairKeyMap
	{
	public:
		class Entry
		{
		public:
			Entry(K1 k1, K2 k2): k1(k1), k2(k2) {}

			K1& First() { return k1; }
			const K1& First() const { return k1; }
			K2& Second() { return k2; }
			const K2& Second() const { return k2; }

			bool operator < (const Entry& ent) const
			{
				return k1 < ent.k1 ? true : k1 == ent.k1 && k2 < ent.k2;
			}
		private:
			K1 k1;
			K2 k2;
		};
		void Clear()
		{
			mapContainer.clear();
		}
		void Put(const Entry& ent, const V& v)
		{
			mapContainer[ent] = v;
		}
		void Put(const K1& k1, const K2 k2, const V& v)
		{
			mapContainer[Entry(k1, k2)] = v;
		}
		V& Get(const Entry& ent)
		{
			return mapContainer.find(ent)->second;
		}
		const V& Get(const Entry& ent) const
		{
			return mapContainer.find(ent)->second;
		}
		V& Get(const K1& k1, const K2& k2)
		{
			return mapContainer.find(Entry(k1, k2))->second;
		}
		const V& Get(const K1& k1, const K2& k2) const
		{
			return mapContainer.find(Entry(k1, k2))->second;
		}
		bool Contains(const Entry& ent) const
		{
			return mapContainer.find(ent) != mapContainer.end();
		}
		bool Contains(const K1& k1, const K2& k2) const
		{
			return mapContainer.find(Entry(k1, k2)) != mapContainer.end();
		}
		std::string ToString() const
		{
			std::stringstream ss;
			typedef typename std::map<Entry, V>::const_iterator MapIter;
			for(MapIter it = mapContainer.begin(); it != mapContainer.end(); ++it)
			{
				ss << "(" << it->first.First() << "," << it->first.Second() << ") => "
					<< it->second << std::endl;
			}
			return ss.str();
		}
		bool Empty() const
		{
			return mapContainer.empty();
		}
		size_t Size() const
		{
			return mapContainer.size();
		}
	private:
		std::map<Entry, V> mapContainer;
	};

}
}

#endif
