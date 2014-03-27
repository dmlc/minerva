#include <minerva/util/utils.h>

namespace minerva
{

namespace utils
{

	unsigned long long CantorPairingFunctionUnsigned(uint64_t x, uint64_t y)
	{
		return (x + y) * (x + y + 1) / 2 + y;
	}

	unsigned long long CantorPairingFunction(int64_t x, int64_t y)
	{
		uint32_t ux = x >= 0 ? 2 * x : -2 * x - 1;
		uint32_t uy = y >= 0 ? 2 * y : -2 * y - 1;
		//return CantorPairingFunctionUnsigned(ux, uy);
		
		//2x is too big for uint32 and there is no x below 0
		return CantorPairingFunctionUnsigned(x, y);
	}

	void Sequencer(boost::function<void(void)> f1, boost::function<void(void)> f2)
	{
		f1();
		f2();
	}
	M_FLOAT PerformOp(M_FLOAT lhs, M_FLOAT rhs, OPTYPE op)
	{
		switch(op)
		{
		case ADD: return lhs + rhs;
		case MINUS: return lhs - rhs;
		case MULT: return lhs * rhs;
		case DIV: return lhs / rhs;
		};
		assert(false); // unknown op
		return 0.0;
	}
}

}
