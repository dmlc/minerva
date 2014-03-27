#ifndef MINERVA_UTILS_TIMER_H
#define MINERVA_UTILS_TIMER_H

#include <boost/date_time/posix_time/posix_time.hpp>

namespace minerva
{
namespace utils
{
	class Timer
	{
		typedef boost::posix_time::microsec_clock ClockType;
	public:
		Timer()
		{
			Reset();
		}
		void Reset()
		{
			start = ClockType::local_time();
		}
		double ElapsedSecond()
		{
			boost::posix_time::time_duration diff = ClockType::local_time() - start;
			return ((double)diff.total_microseconds()) / 1000000.0;
		}
		double ElapsedMillisecond()
		{
			boost::posix_time::time_duration diff = ClockType::local_time() - start;
			return ((double)diff.total_microseconds()) / 1000.0;
		}
		double ElapsedMicrosecond()
		{
			boost::posix_time::time_duration diff = ClockType::local_time() - start;
			return (double)diff.total_microseconds();
		}
	private:
		boost::posix_time::ptime start;
	};
}
}

#endif
