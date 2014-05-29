#include "dag.h"
#include <functional>
#include <cstdio>

using namespace std;

int main() {
    Dag d;
    d.NewDataNode()->AddParent(d.Root());
    d.NewDataNode()->AddParent(d.Root());
    d.NewDataNode()->AddParent(d.Root());
    d.NewDataNode()->AddParent(d.Root());
    d.TraverseAndRun();
    return 0;
}
