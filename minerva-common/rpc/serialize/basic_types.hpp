/*  
 * Copyright (c) 2009 Carnegie Mellon University. 
 *     All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an "AS
 *  IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *  express or implied.  See the License for the specific language
 *  governing permissions and limitations under the License.
 *
 * For more about this software visit:
 *
 *      http://www.graphlab.ml.cmu.edu
 *
 */


/*
   This files defines the serializer/deserializer for all basic types
   (as well as string and pair)  
*/
#ifndef ARCHIVE_BASIC_TYPES_HPP
#define ARCHIVE_BASIC_TYPES_HPP

#include <cassert>
#include <string>
#include <cxxabi.h>

namespace graphlab {
namespace archive_detail {

	/** Serializes POD types */
    template <typename OutArcType, typename T>
    struct serialize_impl<OutArcType, T, true> {
      static void exec(OutArcType& oarc, const T& t) {
		/*int status;
		char* realname;
		realname = abi::__cxa_demangle(typeid(T).name(), 0, 0, &status);
		std::cout << "serial type: " << realname << std::endl;
		free(realname);*/
        oarc.write(reinterpret_cast<const char*>(&t), sizeof(T));
      }
    };
	/** Deserializes POD types */
    template <typename InArcType, typename T>
    struct deserialize_impl<InArcType, T, true>{
      inline static void exec(InArcType& iarc, T &t) {
		/*int status;
		char* realname;
		realname = abi::__cxa_demangle(typeid(T).name(), 0, 0, &status);
		std::cout << "deserial type: " << realname << std::endl;
		free(realname);*/
        iarc.read(reinterpret_cast<char*>(&t), sizeof(T));
      }
    };


    /** Serialization of null terminated const char* strings.
     * This is necessary to serialize constant strings like
     * \code 
     * oarc << "hello world";
     * \endcode
     */
    template <typename OutArcType>
    struct serialize_impl<OutArcType, const char*, false> {
      static void exec(OutArcType& oarc, const char* const& s) {
        // save the length
        // ++ for the \0
        size_t length = strlen(s); length++;
		serialize(oarc, length);
        oarc.write(reinterpret_cast<const char*>(s), length);
        assert(!oarc.fail());
      }
    };


    /// Serialization of fixed length char arrays
    template <typename OutArcType, size_t len>
    struct serialize_impl<OutArcType, char [len], false> {
      static void exec(OutArcType& oarc, const char s[len] ) { 
        size_t length = len;
		serialize(oarc, length);
        oarc.write(reinterpret_cast<const char*>(s), length);
        assert(!oarc.fail());
      }
    };


    /// Serialization of null terminated char* strings
    template <typename OutArcType>
    struct serialize_impl<OutArcType, char*, false> {
      static void exec(OutArcType& oarc, char* const& s) {
        // save the length
        // ++ for the \0
        size_t length = strlen(s); length++;
		serialize(oarc, length);
        oarc.write(reinterpret_cast<const char*>(s), length);
        assert(!oarc.fail());
      }
    };

    /// Deserialization of null terminated char* strings
    template <typename InArcType>
    struct deserialize_impl<InArcType, char*, false> {
      static void exec(InArcType& iarc, char*& s) {
        // Save the length and check if lengths match
        size_t length;
		deserialize(iarc, length);
        s = new char[length];
        //operator>> the rest
        iarc.read(reinterpret_cast<char*>(s), length);
        assert(!iarc.fail());
      }
    };
  
    /// Deserialization of fixed length char arrays 
    template <typename InArcType, size_t len>
    struct deserialize_impl<InArcType, char [len], false> {
      static void exec(InArcType& iarc, char s[len]) { 
        size_t length;
		deserialize(iarc, length);
        //ASSERT_LE(length, len);
        assert(length <= len);
        iarc.read(reinterpret_cast<char*>(s), length);
        assert(!iarc.fail());
      }
    };



    /// Serialization of std::string
    template <typename OutArcType>
    struct serialize_impl<OutArcType, std::string, false> {
      static void exec(OutArcType& oarc, const std::string& s) {
        size_t length = s.length();
		serialize(oarc, length);
        oarc.write(reinterpret_cast<const char*>(s.c_str()), 
                   (std::streamsize)length);
        assert(!oarc.fail());
      }
    };


    /// Deserialization of std::string
    template <typename InArcType>
    struct deserialize_impl<InArcType, std::string, false> {
      static void exec(InArcType& iarc, std::string& s) {
        //read the length
        size_t length;
		deserialize(iarc, length);
        //resize the string and read the characters
        s.resize(length);
        iarc.read(const_cast<char*>(s.c_str()), (std::streamsize)length);
        assert(!iarc.fail());
      }
    };

    /// Serialization of std::pair
    template <typename OutArcType, typename T, typename U>
    struct serialize_impl<OutArcType, std::pair<T, U>, false > {
      static void exec(OutArcType& oarc, const std::pair<T, U>& s) {
		serialize(oarc, s.first);
		serialize(oarc, s.second);
      }
    };

    /// Deserialization of std::pair
    template <typename InArcType, typename T, typename U>
    struct deserialize_impl<InArcType, std::pair<T, U>, false > {
      static void exec(InArcType& iarc, std::pair<T, U>& s) {
		deserialize(iarc, s.first);
		deserialize(iarc, s.second);
      }
    };

	/** Serializes an arbitrary pointer + length to an archive */
	template<typename OutArcType>
	OutArcType& serialize(OutArcType& oarc, const void* str, const size_t length)
   	{
		// save the length
		serialize(oarc, length);
	   	oarc.write(reinterpret_cast<const char*>(str), (std::streamsize)length);
	   	assert(!oarc.fail());
	   	return oarc;
	}

	/** deserializes an arbitrary pointer + length from an archive */
	template<typename InArcType>
	InArcType& deserialize(InArcType& iarc, void* str, const size_t length) {
		// Save the length and check if lengths match
		size_t length2;
		deserialize(iarc, length2);
	   	assert(length == length2);
		//operator>> the rest
		iarc.read(reinterpret_cast<char*>(str), (std::streamsize)length);
	   	assert(!iarc.fail());
	   	return iarc;
   	}


  } // namespace archive_detail
} // namespace graphlab
 
#undef INT_SERIALIZE
#endif

