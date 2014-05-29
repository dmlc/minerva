#pragma once
#include <atomic>

class BoolFlag {
private:
    BoolFlag();
    std::atomic<bool> flag;
public:
    BoolFlag(bool);
    ~BoolFlag();
    BoolFlag(const BoolFlag&);
    BoolFlag& operator=(const BoolFlag&);
    bool Read() const;
    void Write(bool);
};

