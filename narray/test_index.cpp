#include "Index.h"
#include <vector>
#include <iostream>

using namespace std;
using namespace minerva::utils;

class MyCls {
public:
	MyCls(const Index& idx): idx_(idx) {}
	void Print() { cout << idx_ << endl; }
private:
	Index idx_;
};

int main() {
	vector<int> x = {1, 2, 3};	
	Index idx(x);
	cout << idx << endl;
	MyCls cls(x);
	cls.Print();
	MyCls cls1({2,3,4});
	cls1.Print();
	return 0;
}
