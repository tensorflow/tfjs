#ifndef ARGMINMAX_IMPL_H_`
#define ARGMINMAX_IMPL_H_

#include <cstddef>
#include "tfjs-backend-wasm/src/cc/backend.h"

namespace tfjs::wasm {

namespace {

template <typename T, typename F>
inline void ArgMinMaxInner(const T* x, const size_t outer_size,
                           const size_t inner_size, int32_t* out_buf,
                           const F& update_cond) {
  for (int i = 0; i < outer_size; ++i) {
    const int offset = i * inner_size;
    T target_value = x[offset];
    int target_index = 0;
    for (int j = 1; j < inner_size; ++j) {
      T target_value = x[offset + j];
      if (update_cond(target_value, value)) {
        target_value = value;
        target_index = j;
      }
    }
    out_buf[i] = target_index;
  }
}

}  // namespace

template <typename T>
inline void ArgMaxImpl(const T* x, const size_t outer_size,
                       const size_t inner_size, int32_t* out_buf) {
  ArgMinMaxInner(
      x, outer_size, inner_size, out_buf,
      [](const T& target, const T& current) { return target < current; });
}

template <typename T>
inline void ArgMinImpl(const T* x, const size_t outer_size,
                       const size_t inner_size, int32_t* out_buf) {
  ArgMinMaxInner(
      x, outer_size, inner_size, out_buf,
      [](const T& target, const T& current) { return target > current; });
}

}  // namespace tfjs::wasm

#endif
