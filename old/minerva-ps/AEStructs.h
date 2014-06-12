#pragma once
#include <minerva/rpc/Message.h>

namespace minerva
{
	class AEKey;
	class AEMeta;
	struct AEData;

	extern std::ostream& operator << (std::ostream& os, const AEKey& key);
	class AEKey
	{
		friend std::ostream& operator << (std::ostream& os, const AEKey& key);
	public:
		AEKey() {}
		template<class A1> AEKey(const A1& a1)
		{
			std::stringstream ss;
			ss << a1;
			str = ss.str();
		}
		template<class A1, class A2> AEKey(const A1& a1, const A2& a2)
		{
			std::stringstream ss;
			ss << a1 << a2;
			str = ss.str();
		}
		template<class A1, class A2, class A3> AEKey(const A1& a1, const A2& a2, const A3& a3)
		{
			std::stringstream ss;
			ss << a1 << a2 << a3;
			str = ss.str();
		}
		bool operator < (const AEKey& other) const
		{
			return str < other.str;
		}
		rpc::MessageBuffer& Marshall(rpc::MessageBuffer& os) const
		{
			return os << str;
		}
		rpc::MessageBuffer& UnMarshall(rpc::MessageBuffer& is)
		{
			return is >> str;
		}
		std::string ToString() const { return str; }
	private:
		std::string str;
	};

	class AEMeta
	{
	public:
		AEMeta(): length(0) {}
		AEMeta(size_t length): length(length) {}
		AEMeta(const AEMeta& other): length(other.length) {}
		virtual ~AEMeta() {}
		size_t Length() const { return length; }

		virtual void Free(void* ptr) const = 0;
		virtual void* CloneData(const void* src) const
		{
			void * ptr = NewZero();
			Copy(ptr, src);
			return ptr;
		}
		virtual void Copy(void* dst, const void* src) const = 0;
		virtual void* NewZero() const = 0;
		virtual void Add(void* rst, const void* a1, const void* a2) const = 0; // rst = a1 + a2
		virtual void Sub(void* rst, const void* a1, const void* a2) const = 0; // rst = a1 - a2
		virtual void Sub(void* rst, const void* a1, const void* a2, const void* a3) const = 0; // rst = a1 - a2 - a3
		virtual void AddSub(void* rst, const void* a1, const void* a2, const void* a3) const = 0; // rst = a1 + a2 - a3

		virtual std::string ToString(void* ptr) const = 0;

		virtual AEMeta* Clone() const = 0;
		virtual int Id() const = 0;
		virtual size_t Bytes() const = 0;
		virtual rpc::MessageBuffer& Marshall(rpc::MessageBuffer& os) const { return os << length; }
		virtual rpc::MessageBuffer& UnMarshall(rpc::MessageBuffer& is) { return is >> length; }

	protected:
		size_t length;
	};

	struct AEData
	{
		AEData(): meta(NULL), data(NULL) {}
		AEData(AEMeta* meta, void* data): meta(meta), data(data) {}
		virtual rpc::MessageBuffer& Marshall(rpc::MessageBuffer& os) const;
		virtual rpc::MessageBuffer& UnMarshall(rpc::MessageBuffer& is);
		std::string ToString() const;
		void FreeSpace();

		AEMeta* meta;
		void* data;
	};

	///////////////////// Commonly used AEMeta class
	enum MetaClassId
	{
		NA,
		INT,
		UINT,
		FLOAT,
		DOUBLE
	};

	template<class M> struct MetaClassHash { static int Id() { return NA; } };

	template<class T>
	class IntegralMeta : public AEMeta
	{
	public:
		IntegralMeta(): AEMeta() {}
		IntegralMeta(size_t length): AEMeta(length) {}

		virtual AEMeta* Clone() const
		{
			return new IntegralMeta<T>(length);
		}
		virtual int Id() const
		{
			return MetaClassHash<IntegralMeta<T> >::Id();
		}
		virtual size_t Bytes() const
		{
			return length * sizeof(T);
		}

		virtual void Free(void* ptr) const
		{
			T* tptr = (T*)ptr;
			delete [] tptr;
		}
		virtual void Copy(void* dst, const void* src) const
		{
			memcpy(dst, src, sizeof(T) * length);
		}
		virtual void* NewZero() const
		{
			T* ptr = new T[length];
			memset(ptr, 0, sizeof(T) * length);
			return ptr;
		}
		
		virtual void Add(void* rst, const void* a1, const void* a2) const // rst = a1 + a2
		{
			T* rstptr = (T*)rst;
			const T* a1ptr = (const T*)a1;
			const T* a2ptr = (const T*)a2;
			for(size_t i = 0; i < length; ++i)
				rstptr[i] = a1ptr[i] + a2ptr[i];
		}
		virtual void Sub(void* rst, const void* a1, const void* a2) const // rst = a1 - a2
		{
			T* rstptr = (T*)rst;
			const T* a1ptr = (const T*)a1;
			const T* a2ptr = (const T*)a2;
			for(size_t i = 0; i < length; ++i)
				rstptr[i] = a1ptr[i] - a2ptr[i];
		}
		virtual void Sub(void* rst, const void* a1, const void* a2, const void* a3) const // rst = a1 - a2 - a3
		{
			T* rstptr = (T*)rst;
			const T* a1ptr = (const T*)a1;
			const T* a2ptr = (const T*)a2;
			const T* a3ptr = (const T*)a3;
			for(size_t i = 0; i < length; ++i)
				rstptr[i] = a1ptr[i] - a2ptr[i] - a3ptr[i];
		}
		virtual void AddSub(void* rst, const void* a1, const void* a2, const void* a3) const // rst = a1 + a2 - a3
		{
			T* rstptr = (T*)rst;
			const T* a1ptr = (const T*)a1;
			const T* a2ptr = (const T*)a2;
			const T* a3ptr = (const T*)a3;
			for(size_t i = 0; i < length; ++i)
				rstptr[i] = a1ptr[i] + a2ptr[i] - a3ptr[i];
		}
		virtual std::string ToString(void* ptr) const
		{
			std::stringstream ss;
			T* tptr = (T*)ptr;
			if(length <= 10)
			{
				ss << length << ":[";
				for(size_t i = 0; i < length; ++i)
					ss << tptr[i] << " ";
				ss << "]";
			}
			else
				ss << length << ":[" << tptr[0] << "," << tptr[length - 1] << "]";
			return ss.str();
		}
	};

	template<> 
	struct MetaClassHash<IntegralMeta<int> > { static int Id() { return INT; } };
	template<> 
	struct MetaClassHash<IntegralMeta<unsigned int> > { static int Id() { return UINT; } };
	template<> 
	struct MetaClassHash<IntegralMeta<double> > { static int Id() { return DOUBLE; } };
	template<> 
	struct MetaClassHash<IntegralMeta<float> > { static int Id() { return FLOAT; } };

}
