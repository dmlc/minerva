#include <boost/chrono/chrono.hpp>
#include <boost/thread/thread.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <iostream>
#include <time.h>

using namespace boost;

int main()
{
	//time_point<system_clock>::duration dur = system_clock::now().time_since_epoch();
	//std::cout << dur.hours() << std::endl;
	//std::cout << dur.minutes() << std::endl;
	//std::cout << dur.seconds() << std::endl;
	//std::cout << duration_cast<seconds>(system_clock::now().time_since_epoch()).count() << std::endl;
	posix_time::ptime t = posix_time::second_clock::local_time();
	std::cout << t << std::endl;
	//std::cout << t + posix_time::seconds(1) << std::endl;
	time_t curtm;
	time(&curtm);
	for(int i = 0; i < 5; ++i)
	{
	std::cout << curtm << std::endl;
	++curtm;
	//struct tm * timeinfo = localtime(&curtm);
	//timeinfo->tm_sec += 5;
	//time_t ftm = mktime(timeinfo);
	//std::cout << ftm << std::endl;
	chrono::time_point<chrono::system_clock> tp = chrono::system_clock::from_time_t(curtm);	
	std::cout << tp.time_since_epoch().count() << std::endl;
	std::cout << chrono::system_clock::now().time_since_epoch().count() << std::endl;
	this_thread::sleep_until(tp);
	std::cout << posix_time::second_clock::local_time() << std::endl;
	std::cout << chrono::system_clock::now().time_since_epoch().count() << std::endl;
	}
	return 0;
}
