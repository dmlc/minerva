#pragma once
#include "bool_flag.h"
#include <list>
#include <mutex>
#include <condition_variable>

template <typename T> class ConcurrentBlockingQueue {
private:
    std::list<T> queue;
    std::mutex mutex;
    std::condition_variable cv;
    BoolFlag exitNow;
public:
    ConcurrentBlockingQueue(): exitNow(false) {
    }
    void Push(const T& e) {
        std::lock_guard<std::mutex> lock(mutex);
        queue.push_back(e);
        if (queue.size() == 1) {
            cv.notify_one();
        }
    }
    bool Pop(T& rv) {
        std::unique_lock<std::mutex> lock(mutex);
        while (queue.empty() && !exitNow.Read()) {
            cv.wait(lock);
        }
        if (!exitNow.Read()) {
            rv = queue.front();
            queue.pop_front();
            return false;
        } else {
            return true;
        }
    }
    std::list<T> PopAll() {
        std::lock_guard<std::mutex> lock(mutex);
        std::list<T> rv;
        rv.swap(queue);
        return rv;
    }
    void SignalForKill() {
        exitNow.Write(true);
        cv.notify_all();
    }
    bool Empty() {
        std::lock_guard<std::mutex> lock(mutex);
        return queue.empty();
    }
};

