/*  =========================================================================
    zmsg.hpp

    Multipart message class for example applications.

    Follows the ZFL class conventions and is further developed as the ZFL
    zfl_msg class.  See http://zfl.zeromq.org for more details.

    -------------------------------------------------------------------------
    Copyright (c) 1991-2010 iMatix Corporation <www.imatix.com>
    Copyright other contributors as noted in the AUTHORS file.

    This file is part of the ZeroMQ Guide: http://zguide.zeromq.org

    This is free software; you can redistribute it and/or modify it under the
    terms of the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your option)
    any later version.

    This software is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABIL-
    ITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General
    Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
    =========================================================================

    Andreas Hoelzlwimmer <andreas.hoelzlwimmer@fh-hagenberg.at>
*/

#ifndef __ZMSG_H_INCLUDED__
#define __ZMSG_H_INCLUDED__

//#include "zhelpers.hpp"
#include <zmq.hpp>

#include <list>
#include <cstring>
#include <iostream>
//#include <stdarg.h>

namespace zmq
{

	typedef message_t frame_t;

	class zmsg {

	public:
		typedef std::list<frame_t*>::iterator frame_iterator;

		zmsg() { }

		//  Copy Constructor, equivalent to zmsg_dup
		zmsg(zmsg &msg) {
		   throw error_t(); // TODO not implemented
		}

		virtual ~zmsg() {
		  clear();
		}

		//  --------------------------------------------------------------------------
		//  Erases all messages
		void clear() {
		   while(!frames.empty())
		   {
			   frame_t* frame = frames.front();
			   frames.pop_front();
			   delete frame;
		   }
		}

		void push_back(frame_t* frame)
		{
		   frames.push_back(frame);
		}
		void push_front(frame_t* frame)
		{
		   frames.push_back(frame);
		}
		frame_t* pop_front()
		{
		   frame_t* front = frames.front();
		   frames.pop_front();
		   return front;
		}
		frame_t* pop_back()
		{
		   frame_t* back = frames.back();
		   frames.pop_back();
		   return back;
		}
		frame_t* front()
		{
		   return frames.front();
		}
		frame_t* back()
		{
		   return frames.back();
		}

		frame_iterator begin()
		{
		   return frames.begin();
		}
		frame_iterator end()
		{
		   return frames.end();
		}

		bool recv(socket_t & socket)
		{
		  clear();
		  while(1)
		  {
			 frame_t *frame = new frame_t();
			 try
			 {
				if (!socket.recv(frame, 0))
				{
					continue;
				}
			 }
			 catch (error_t error)
			 {
				assert(error.num() != 0);
				if(error.num() == EINTR)
					continue;
				else
				{
					std::cout << "[Recv] E: " << error.what() << std::endl;
					return false;
				}
			 }
			 frames.push_back(frame);
			 //std::cout << "x " << frame->size() << std::endl;
			 int64_t more = 0;
			 size_t more_size = sizeof(more);
			 socket.getsockopt(ZMQ_RCVMORE, &more, &more_size);
			 if (!more)
			 {
				break;
			 }
		  }
		  return true;
		}

		void send(zmq::socket_t & socket)
		{
		   while(!frames.empty())
		   {
			   frame_t* frame = pop_front();
			   try
			   {
				   //std::cout << frame->size() << std::endl;
				   assert(socket.send(*frame, frames.size()? ZMQ_SNDMORE : 0));
			   }
			   catch(error_t error)
			   {
				   assert(error.num() != 0);
				   std::cout << "[Send] E: " << error.what() << std::endl;
			   }
			   delete frame;
		   }
		}

		size_t num_frames()
		{
			return frames.size();
		}

	private:
		std::list<frame_t*> frames;
	};

	//  --------------------------------------------------------------------------
	//  Formats 17-byte UUID as 33-char string starting with '@'
	//  Lets us print UUIDs as C strings and use them as addresses
	//
	//char * zmsg_encode_uuid (unsigned char *data)
	//{
	//   static char
	//	   hex_char [] = "0123456789ABCDEF";

	//   assert (data [0] == 0);
	//   char *uuidstr = new char[34];
	//   uuidstr [0] = '@';
	//   int byte_nbr;
	//   for (byte_nbr = 0; byte_nbr < 16; byte_nbr++) {
	//	   uuidstr [byte_nbr * 2 + 1] = hex_char [data [byte_nbr + 1] >> 4];
	//	   uuidstr [byte_nbr * 2 + 2] = hex_char [data [byte_nbr + 1] & 15];
	//   }
	//   uuidstr [33] = 0;
	//   return (uuidstr);
	//}


	//static unsigned char * zmsg_decode_uuid (char *uuidstr)
	//{
	//   static char
	//	   hex_to_bin [128] = {
	//		  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, /* */
	//		  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, /* */
	//		  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, /* */
	//		   0, 1, 2, 3, 4, 5, 6, 7, 8, 9,-1,-1,-1,-1,-1,-1, /* 0..9 */
	//		  -1,10,11,12,13,14,15,-1,-1,-1,-1,-1,-1,-1,-1,-1, /* A..F */
	//		  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, /* */
	//		  -1,10,11,12,13,14,15,-1,-1,-1,-1,-1,-1,-1,-1,-1, /* a..f */
	//		  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 }; /* */

	//   assert (strlen (uuidstr) == 33);
	//   assert (uuidstr [0] == '@');
	//   unsigned char *data = new unsigned char[17];
	//   int byte_nbr;
	//   data [0] = 0;
	//   for (byte_nbr = 0; byte_nbr < 16; byte_nbr++)
	//	   data [byte_nbr + 1]
	//		   = (hex_to_bin [uuidstr [byte_nbr * 2 + 1] & 127] << 4)
	//		   + (hex_to_bin [uuidstr [byte_nbr * 2 + 2] & 127]);

	//   return (data);
	//}

}
#endif /* ZMSG_H_ */
