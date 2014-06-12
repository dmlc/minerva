#include "Index.h"
#include <vector>
#include <iostream>

using namespace std;
using namespace minerva;

class MyCls {
public:
	MyCls(const Index& idx): idx_(idx) {}
	void Print() { cout << idx_ << endl; }
	void operator [] (const initializer_list<int>& i) {
		cout << "OOOPS" << endl;
	}
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
	cls1[{3,4,5}];
	return 0;
}
