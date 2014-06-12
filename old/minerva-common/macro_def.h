
#include <boost/bind.hpp>
#include <boost/foreach.hpp>

#ifdef MINERVA_MACROS 
#error "Repeated include of <macros_def.h>. This probably means that macros_def.h was not the last include, or some header file failed to include <macros_undef.h>"
#endif

#define MINERVA_MACROS 

//#ifndef MACRO_FOREACH
//#define MACRO_FOREACH
//#pragma message("define macros")

#define foreach BOOST_FOREACH
#define rev_foreach BOOST_REVERSE_FOREACH

