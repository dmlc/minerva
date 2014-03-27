/**  
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


#ifndef GRAPHLAB_SERIALIZE_HPP
#define GRAPHLAB_SERIALIZE_HPP

#include <boost/type_traits.hpp>

namespace graphlab {
namespace archive_detail {

	template <typename OutArcType, typename T, bool IsPOD>
	struct serialize_impl {
	   	static void exec(OutArcType& oarc, const T& t) {
		   	assert(false); // TODO we do not know how to serialize
	   	}
   	};

	template <typename InArcType, typename T, bool IsPOD>
	struct deserialize_impl {
	   	inline static void exec(InArcType& iarc, T& t) {
		   	assert(false); // TODO we do not know how to deserialize
	   	}
   	};

	template <typename OutArcType, typename T>
	void serialize(OutArcType& oarc, const T& t)
	{
		archive_detail::serialize_impl<OutArcType, T, boost::is_pod<T>::value>::exec(oarc, t);
	}

	template <typename InArcType, typename T>
	void deserialize(InArcType& iarc, T& t)
	{
		archive_detail::deserialize_impl<InArcType, T, boost::is_pod<T>::value>::exec(iarc, t);
	}

}
}

#include <minerva/rpc/serialize/basic_types.hpp>

#include <minerva/rpc/serialize/list.hpp>
#include <minerva/rpc/serialize/set.hpp>
#include <minerva/rpc/serialize/vector.hpp>
#include <minerva/rpc/serialize/map.hpp>
#include <minerva/rpc/serialize/unordered_map.hpp>
#include <minerva/rpc/serialize/unordered_set.hpp>

#endif 

