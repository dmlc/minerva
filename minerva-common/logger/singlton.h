/*
 The MIT License (MIT)
Copyright (c) 2013 <AthrunArthur>
Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
*/

#ifndef FF_SINGLETON_H
#define FF_SINGLETON_H
#if __cplusplus < 201103L
#include <boost/thread/mutex.hpp>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#else
#include <thread>
#include <mutex>
#include <memory>
#endif



namespace ff {

template<typename T>
class singleton
#if __cplusplus < 201103L
    : public boost::noncopyable
#endif
{
public:
    static T& instance()
    {
        //std::call_once(s_oOnce, singleton<T>::init);
        if(!s_pInstance)
            init();
        return *s_pInstance;
    }
    #if __cplusplus < 201103L
    static boost::shared_ptr<T> instance_ptr()
#else
    static std::shared_ptr<T> instance_ptr()
#endif
    {
        //std::call_once(s_oOnce, singleton<T>::init);
        if(!s_pInstance)
            init();
        return s_pInstance;
    }
protected:
    singleton() {}
#if __cplusplus >= 201103L    
    singleton(const singleton<T> & ) = delete;
#endif
private:
    static void init()
    {
#if __cplusplus < 201103L
        s_pInstance = boost::shared_ptr<T>(new T());
#else
        s_pInstance = std::shared_ptr<T>(new T());
#endif
    }

protected:
#if __cplusplus < 201103L
    static boost::shared_ptr<T>		s_pInstance;
#else
    static std::shared_ptr<T>		s_pInstance;
#endif
    //static std::once_flag			s_oOnce;
};
#if __cplusplus < 201103L
template <typename T>
boost::shared_ptr<T> singleton<T>::s_pInstance;
#else
template <typename T>
std::shared_ptr<T> singleton<T>::s_pInstance(nullptr);
#endif
//template <typename T>
//std::once_flag		singleton<T>::s_oOnce;
}//end namespace ff

#endif
