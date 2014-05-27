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
    auto f = [] (DagNode* n) {
        printf("Visiting %llu\n", (unsigned long long) n->ID());
        n->Runner()();
    };
    d.BreadthFirstSearch(f);
    return 0;
}
