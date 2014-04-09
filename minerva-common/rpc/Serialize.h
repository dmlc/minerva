#ifndef MINERVA_RPC_SERIALIZE_H
#define MINERVA_RPC_SERIALIZE_H
#pragma once

#include <map>
#include <set>
#include <deque>
#include <list>
#include <vector>

#include <boost/type_traits.hpp>
#include <boost/shared_ptr.hpp>

namespace minerva
{
namespace rpc
{
//----------------------------------------------
// Marshal/UnMarshal 

template<class OutArcType, class T, bool POD>
struct MarshallImpl
{
	static OutArcType& Exec(OutArcType& os, const T& val)
	{
		return val.Marshall(os);
	}
};

template<class InArcType, class T, bool POD>
struct UnMarshallImpl
{
	static InArcType& Exec(InArcType &is, T &val)
	{
		return val.UnMarshall(is);
	}
};

template<class OutArcType, class T>
struct MarshallImpl<OutArcType, T, true>
{
	static OutArcType& Exec(OutArcType &os, const T &val)
	{
		os.write((char *)&val, sizeof(T));
		return os;
	}
};

template<class InArcType, class T>
struct UnMarshallImpl<InArcType, T, true>
{
	static InArcType& Exec(InArcType &is, T &val)
	{
		is.read((char *)&val, sizeof(T));
		return is;
	}
};

template<class OutArcType, class T>
inline OutArcType& Marshall(OutArcType& os, const T& val)
{
	return MarshallImpl<OutArcType, T, boost::is_pod<T>::value>::Exec(os, val);
}

template<class InArcType, class T>
inline InArcType& UnMarshall(InArcType& is, T& val)
{
	return UnMarshallImpl<InArcType, T, boost::is_pod<T>::value>::Exec(is, val);
}

// serialization of pointers
template<class OutArcType, class T>
inline OutArcType& Marshall(OutArcType &os, T* val)
{
	Marshall(os, *val);
	return os;
}

template<class InArcType, class T>
inline InArcType& UnMarshall(InArcType &is, T* &val)
{
	val = new T;
    UnMarshall(is, *val);
	return is;
}

// serialization of boost shared pointers
template<class OutArcType, class T>
inline OutArcType& Marshall(OutArcType &os, boost::shared_ptr<T> val)
{
	Marshall(os, *val);
	return os;
}

template<class InArcType, class T>
inline InArcType& UnMarshall(InArcType &is, boost::shared_ptr<T> &val)
{
	val = boost::shared_ptr<T>(new T);
    UnMarshall(is, *val);
	return is;
}

//////////////////////////////////////// Serialize/Deserialize of containers ///////////////////////////////////////////
template<class OutArcType>
inline OutArcType& Marshall(OutArcType &os, const std::stringstream &val)
{
	std::string v = val.str();
	Marshall(os, (int)v.size());
	os.write((const char*)v.c_str(), v.size());
	return os;
}

template<class InArcType>
inline InArcType& UnMarshall(InArcType &is, std::stringstream &val)
{
	const int buffer_size = 512;
	int len;
	UnMarshall(is, len);

	char buffer[buffer_size];
	
	for (int i = len / buffer_size; i > 0; i--)
	{
		is.read((char*)buffer, buffer_size);
		val.write(buffer, buffer_size);
	}

	is.read((char*)buffer, len % buffer_size);
	val.write(buffer, len % buffer_size);
	return is;
}

template<class OutArcType>
inline OutArcType& Marshall(OutArcType &os, const std::string &str)
{
	Marshall(os, (uint32_t)str.length());
    if (str.length() != 0)
    {
    	os.write((char *)str.c_str(), (std::streamsize)str.length());
    }
	return os;
}

template<class InArcType>
inline InArcType& UnMarshall (InArcType &is, std::string &str)
{
	uint32_t size;
	UnMarshall(is, size);
    if (size == 0)
    {
        str = std::string("");
    }
    else
    {
		str.resize(size);
		is.read(&str[0], size);
    }
	return is;
}

template<class OutArcType, class _First, class _Second>
inline OutArcType & Marshall (OutArcType &os,const std::pair<_First, _Second> &data)
{
	Marshall(os, data.first);
	Marshall(os, data.second);

	return os;
}

template<class InArcType, class _First, class _Second>
inline InArcType & UnMarshall (InArcType &is, std::pair<_First, _Second> &data)
{
	UnMarshall(is, data.first);
	UnMarshall(is, data.second);

	return is;
}

template<class OutArcType, class _Ty>
inline OutArcType & Marshall (OutArcType &os,const std::vector<_Ty> &data)
{
	Marshall(os,(uint32_t)data.size());
	for(size_t i = 0; i < data.size(); i++)
	{
		Marshall(os, data[i]);
	}
	return os;
}

template<class InArcType, class _Ty>
inline InArcType & UnMarshall (InArcType &is, std::vector<_Ty> &data)
{
	uint32_t size;
	data.clear();
	UnMarshall(is, size);
	data.resize(size);
	try
	{
		for(uint32_t i = 0; i < size; i++)
		{
			UnMarshall(is, data[i]);
		}
	}
	catch(std::exception e)
	{

	}
	return is;
}

template<class OutArcType, class _Kty, class _Ty, class _Pr, class _Alloc>
inline OutArcType & Marshall (OutArcType &os,const std::map<_Kty, _Ty, _Pr, _Alloc> &data)
{
	Marshall(os, (uint32_t)data.size());
	typedef typename std::map<_Kty, _Ty, _Pr, _Alloc>::const_iterator map_iter;
	for(map_iter it = data.begin(); it != data.end(); ++it)
	{
		Marshall(os, it->first);
		Marshall(os, it->second);
	}
	return os;
}

template<class InArcType, class _Kty, class _Ty, class _Pr, class _Alloc>
inline InArcType & UnMarshall (InArcType &is, std::map<_Kty, _Ty, _Pr, _Alloc> &data)
{
	uint32_t size;

	data.clear();
	UnMarshall(is, size);
	for (uint32_t i = 0 ; i < size ; i++ )
	{
		_Kty k;
		_Ty t;
		UnMarshall(is, k);
		UnMarshall(is, t);
		data.insert(std::map<_Kty, _Ty, _Pr, _Alloc>::value_type(k,t));
	}
	return is;
}

template<class OutArcType, class _Kty, class _Pr, class _Alloc>
inline OutArcType & Marshall (OutArcType &os,const std::set<_Kty, _Pr, _Alloc> &data)
{
	Marshall(os, (uint32_t)data.size());
	typedef typename std::set<_Kty, _Pr, _Alloc>::const_iterator set_iter;
	for(set_iter it = data.begin(); it != data.end(); ++it)
	{
		Marshall(os, *it);		
	}
	return os;
}

template<class InArcType, class _Kty, class _Pr, class _Alloc>
inline InArcType & UnMarshall (InArcType &is, std::set<_Kty, _Pr, _Alloc> &data)
{
	uint32_t size;

	data.clear();
	UnMarshall(is, size);
	for (uint32_t i = 0 ; i < size ; i++ )
	{
		_Kty k;
		UnMarshall(is, k);
		data.insert(std::set<_Kty, _Pr, _Alloc>::value_type(k));
	}
	return is;
}

template<class OutArcType, class _Kty>
inline OutArcType & Marshall (OutArcType &os,const std::deque<_Kty>& data)
{
	Marshall(os, (uint32_t)data.size());
	typedef typename std::deque<_Kty>::const_iterator deque_iter;
	for (deque_iter it = data.begin(); it != data.end(); ++it)
	{
		Marshall(os, *it);		
	}
	return os;
}

template<class InArcType, class _Kty>
inline InArcType & UnMarshall (InArcType &is, std::deque<_Kty> &data)
{
	uint32_t size;

	data.clear();
	UnMarshall(is, size);
	for (uint32_t i = 0 ; i < size ; i++ )
	{
		_Kty k;
		UnMarshall(is, k);
		data.push_back(std::deque<_Kty>::value_type(k));
	}
	return is;
}

template<class OutArcType, class _Kty, class _Alloc>
inline OutArcType & Marshall (OutArcType &os,const std::list<_Kty, _Alloc>& data)
{
	Marshall(os, (uint32_t)data.size());
	typedef typename std::list<_Kty>::const_iterator list_iter;
	for (list_iter it = data.begin(); it != data.end(); ++it)
	{
		Marshall(os, *it);		
	}
	return os;
}

template<class InArcType, class _Kty, class _Alloc>
inline InArcType & UnMarshall (InArcType &is, std::list<_Kty, _Alloc> &data)
{
	uint32_t size;

	data.clear();
	UnMarshall(is, size);
	for (uint32_t i = 0 ; i < size ; i++ )
	{
		_Kty k;
		UnMarshall(is, k);
		data.push_back(std::list<_Kty, _Alloc>::value_type(k));
	}
	return is;
}

} // end of namespace rpc
} // end of namespace minerva
#endif
