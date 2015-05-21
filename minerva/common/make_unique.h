#pragma once
#include <memory>
#include <cstddef>

namespace minerva {
namespace common {

namespace {

template<typename T>
struct MakeUniqueHandler {
  using Type = std::unique_ptr<T>;

  template<typename... Args>
  static inline Type Make(Args&&... args) {
    return Type(new T(std::forward<Args>(args)...));
  }
};

template<typename T>
struct MakeUniqueHandler<T[]> {
  using Type = std::unique_ptr<T[]>;

  template<typename... Args>
  static inline Type Make(Args&&... args) {
    return Type(new T[sizeof...(Args)]{std::forward<Args>(args)...});
  }
};

template<typename T, std::size_t n>
struct MakeUniqueHandler<T[n]> {
  using Type = std::unique_ptr<T[]>;

  template<typename... Args>
  static inline Type Make(Args&&... args) {
    static_assert(sizeof...(Args) <= n,
        "for MakeUnique<T[n]>, n must be as large as the number of arguments");
    return Type(new T[n]{std::forward<Args>(args)...});
  }
};

}  // anonymous namespace

template<typename T, typename... Args>
inline typename MakeUniqueHandler<T>::Type
MakeUnique(Args&&... args) {
  return MakeUniqueHandler<T>::Make(std::forward<Args>(args)...);
}

}  // namespace common
}  // namespace minerva

