#ifndef MINERVA_RPC_MESSAGE_H
#define MINERVA_RPC_MESSAGE_H
#pragma once

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <assert.h>
#include <stdint.h>
#include <list>
#include <utility>
#include <iostream>

#include <boost/function.hpp>

#include <minerva/rpc/Serialize.h>

namespace minerva
{
namespace rpc
{

class TinyStringStreamBuf : public std::streambuf
{
	static const size_t MIN_SIZE = 32;
	static const size_t THRINK_THRESHOLD = 32*1024;
public:
	TinyStringStreamBuf() : _buf(NULL), _ppos(0), _gpos(0), _buf_size(0){};
	~TinyStringStreamBuf() { free(_buf); }
	virtual std::streamsize xsputn(const char * b,	std::streamsize n)
	{
		//std::cout << "xsputn: n=" << n << " _buf_size=" << _buf_size << " _ppos=" << _ppos << std::endl;
		if ((std::streamsize)(_buf_size - _ppos) < n)
		{
			size_t least_size = _ppos + n;
			size_t new_size = (_buf_size*2) > MIN_SIZE ? (_buf_size*2) : MIN_SIZE;
			new_size = new_size > least_size ? new_size : least_size;
			char* tmp = (char*)realloc(_buf, new_size);
			if(!tmp)
			{
				std::cout << "Warning: realloc fail: new_size=" << new_size << std::endl;
				tmp = (char*)malloc(new_size);
				memcpy(tmp, _buf, _buf_size);
				free(_buf);
			}
			_buf = tmp;
			assert(_buf);
			_buf_size = new_size;
		}
		memcpy(_buf + _ppos, b, n);
		_ppos += n;
		return n;
	}
	virtual std::streamsize xsgetn(char * b,	std::streamsize n)
	{
		if (_gpos + n > _ppos)
		{
			return 0;
		}
		memcpy(b, _buf+_gpos, n);
		_gpos += n;
		// shrink if too much wasted space
		if (_gpos >= THRINK_THRESHOLD && _gpos >= _buf_size/2)
		{
			memmove(_buf, _buf + _gpos, _ppos - _gpos);
			_ppos -= _gpos;
			_buf_size -= _gpos;
			_gpos = 0;
			if (_buf_size != 0)
			{
				_buf = (char*)realloc(_buf, _buf_size);
				assert(_buf);
			}
			else
			{
				free(_buf);
				_buf = NULL;
			}
		}
		return n;
	}

	inline size_t gsize() const
	{
		return _ppos - _gpos;
	}

	inline size_t psize() const
	{
		return _ppos;
	}

	inline char * get_buf() const
	{
		return _buf + _gpos;
	}

	inline void set_buf(char * p, size_t s, bool is_write = false)
	{		
		if (p != _buf)
			free(_buf);

		_buf = p;
		_ppos = is_write ? 0 : s;
		_gpos = 0;
		_buf_size = s;
	}
private:
	char * _buf;
	size_t _ppos;
	size_t _gpos;
	size_t _buf_size;
};

class TinyStringStream : public std::iostream
{
	friend std::ostream & Marshall(std::ostream & os, const TinyStringStream & ss);
	friend std::istream & UnMarshall(std::istream & is, TinyStringStream & ss);
private:
	TinyStringStream(const TinyStringStream &);
	TinyStringStream operator = (const TinyStringStream &);
	template<class T> TinyStringStream & operator << (const T &);
	template<class T> TinyStringStream & operator >> (T &);
public:
	TinyStringStream() : std::iostream(&_sb){}
	~TinyStringStream(){}
	inline size_t gsize() const
	{
		return _sb.gsize();
	}
	inline size_t psize() const
	{
		return _sb.psize();
	}
	inline char * get_buf() const
	{
		return _sb.get_buf();
	}
	inline void set_buf(char * b, size_t n, bool is_write = false)
	{
		return _sb.set_buf(b, n, is_write);
	}
private:
	TinyStringStreamBuf _sb;
};


struct RawBuf
{
	typedef void(*FreeFn)(void* data, void* arg);
	enum RawBufType
	{
		RAW_BUF_CPU = 0,
		RAW_BUF_GPU
	};
	RawBuf(): buf(NULL), size(0), type(-1), freefn(NULL), arg(NULL) {}
	RawBuf(char * b, uint64_t s, int t, FreeFn fn, void* arg):
		buf(b), size(s), type(t), freefn(fn), arg(arg) {}
	char * buf;
	uint64_t size;
	int type;

	FreeFn freefn;
	void* arg;
};

class MessageBuffer : public TinyStringStream
{
public:
	template<class T> MessageBuffer & operator << (const T &val)
	{
		Marshall(*this, val);
		return *this;
	}
	template<class T> MessageBuffer & operator >> (T &val)
	{
		UnMarshall(*this, val);
		return *this;
	}
	void push_raw_buf(char * buf, uint64_t size, int type, RawBuf::FreeFn fn = NULL, void* arg = NULL)
	{
		_raw_bufs.push_back(RawBuf(buf, size, type, fn, arg));
	}
	void pop_raw_buf(char ** buf, uint64_t * size)
	{
		RawBuf b = pop_raw_buf();
		*buf = b.buf;
		*size = b.size;
	}
	RawBuf pop_raw_buf()
	{
		assert(!_raw_bufs.empty());
		RawBuf b = _raw_bufs.front();
		_raw_bufs.erase(_raw_bufs.begin());
		return b;
	}
	size_t get_num_raw_buf(){ return _raw_bufs.size(); }
	std::vector<RawBuf> & get_raw_bufs(){return _raw_bufs;}
private:
	std::vector<RawBuf> _raw_bufs;
};

} // end of namespace rpc
} // end of namespace minerva

#endif
