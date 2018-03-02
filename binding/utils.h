/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

#ifndef TF_NODEJS_UTILS_H_
#define TF_NODEJS_UTILS_H_

#include <stdio.h>
#include <cstdlib>
#include <sstream>
#include <vector>
#include "../deps/tensorflow/include/tensorflow/c/c_api.h"
#include "tf_auto_status.h"

#define NAPI_STRING_SIZE 512

#define MAX_TENSOR_SHAPE 4

#define ARRAY_SIZE(array) (sizeof(array) / sizeof(array[0]))

#define DEBUG 0

#define DEBUG_LOG(message, file, lineNumber)                             \
  do {                                                                   \
    if (DEBUG)                                                           \
      fprintf(stderr, "** -%s:%lu\n-- %s\n", file, lineNumber, message); \
  } while (0)

namespace tfnodejs {

inline void NapiThrowError(napi_env env, const char* message, const char* file,
                           const size_t lineNumber) {
  DEBUG_LOG(message, file, lineNumber);
  napi_throw_error(env, nullptr, message);
}

#define ENSURE_NAPI_OK(env, status) \
  EnsureNapiOK(env, status, __FILE__, __LINE__)

inline void EnsureNapiOK(napi_env env, napi_status status, const char* file,
                         const size_t lineNumber) {
  if (status != napi_ok) {
    const napi_extended_error_info* error_info = 0;
    napi_get_last_error_info(env, &error_info);

    std::ostringstream oss;
    oss << "Invalid napi_status: " << error_info->error_message;
    NapiThrowError(env, oss.str().c_str(), file, lineNumber);
  }
}

#define ENSURE_TF_OK(env, status) EnsureTFOK(env, status, __FILE__, __LINE__)

inline void EnsureTFOK(napi_env env, TF_AutoStatus& status, const char* file,
                       const size_t lineNumber) {
  if (TF_GetCode(status.status) != TF_OK) {
    std::ostringstream oss;
    oss << "Invalid TF_Status: " << TF_GetCode(status.status);
    NapiThrowError(env, oss.str().c_str(), file, lineNumber);
  }
}

#define ENSURE_CONSTRUCTOR_CALL(env, nstatus) \
  EnsureConstructorCall(env, info, __FILE__, __LINE__)

inline void EnsureConstructorCall(napi_env env, napi_callback_info info,
                                  const char* file, const size_t lineNumber) {
  napi_value js_target;
  napi_status nstatus = napi_get_new_target(env, info, &js_target);
  ENSURE_NAPI_OK(env, nstatus);
  if (js_target == nullptr) {
    NapiThrowError(env, "Function not used as a constructor!", file,
                   lineNumber);
  }
}

#define ENSURE_VALUE_IS_ARRAY(env, value) \
  EnsureValueIsArray(env, value, __FILE__, __LINE__)

inline void EnsureValueIsArray(napi_env env, napi_value value, const char* file,
                               const size_t lineNumber) {
  bool is_array;
  ENSURE_NAPI_OK(env, napi_is_array(env, value, &is_array));
  if (!is_array) {
    NapiThrowError(env, "Argument is not an array!", file, lineNumber);
  }
}

#define ENSURE_VALUE_IS_TYPED_ARRAY(env, value) \
  EnsureValueIsTypedArray(env, value, __FILE__, __LINE__)

inline void EnsureValueIsTypedArray(napi_env env, napi_value value,
                                    const char* file, const size_t lineNumber) {
  bool is_array;
  ENSURE_NAPI_OK(env, napi_is_typedarray(env, value, &is_array));
  if (!is_array) {
    NapiThrowError(env, "Argument is not a typed-array!", file, lineNumber);
  }
}

#define ENSURE_VALUE_IS_LESS_THAN(env, value, max) \
  EnsureValueIsLessThan(env, value, max, __FILE__, __LINE__)

inline void EnsureValueIsLessThan(napi_env env, uint32_t value, uint32_t max,
                                  const char* file, const size_t lineNumber) {
  if (value > max) {
    std::ostringstream oss;
    oss << "Argument is greater than max: " << value << " > " << max;
    NapiThrowError(env, oss.str().c_str(), file, lineNumber);
  }
}

#define REPORT_UNKNOWN_TF_DATA_TYPE(env, type) \
  ReportUnknownTFDataType(env, type, __FILE__, __LINE__)

inline void ReportUnknownTFDataType(napi_env env, TF_DataType type,
                                    const char* file, const size_t lineNumber) {
  std::ostringstream oss;
  oss << "Unhandled TF_DataType: " << type;
  NapiThrowError(env, oss.str().c_str(), file, lineNumber);
}

#define REPORT_UNKNOWN_TF_ATTR_TYPE(env, type) \
  ReportUnknownTFAttrType(env, type, __FILE__, __LINE__)

inline void ReportUnknownTFAttrType(napi_env env, TF_AttrType type,
                                    const char* file, const size_t lineNumber) {
  std::ostringstream oss;
  oss << "Unhandled TF_AttrType: " << type;
  NapiThrowError(env, oss.str().c_str(), file, lineNumber);
}

#define REPORT_UNKNOWN_TYPED_ARRAY_TYPE(env, type) \
  ReportUnknownTypedArrayType(env, type, __FILE__, __LINE__)

inline void ReportUnknownTypedArrayType(napi_env env, napi_typedarray_type type,
                                        const char* file,
                                        const size_t lineNumber) {
  std::ostringstream oss;
  oss << "Unhandled napi typed_array_type: " << type;
  NapiThrowError(env, oss.str().c_str(), file, lineNumber);
}

// Returns a vector with the shape values of an array.
inline void ExtractArrayShape(napi_env env, napi_value array_value,
                              std::vector<int64_t>* result) {
  napi_status nstatus;

  uint32_t array_length;
  nstatus = napi_get_array_length(env, array_value, &array_length);
  ENSURE_NAPI_OK(env, nstatus);

  for (uint32_t i = 0; i < array_length; i++) {
    napi_value dimension_value;
    nstatus = napi_get_element(env, array_value, i, &dimension_value);
    ENSURE_NAPI_OK(env, nstatus);

    int64_t dimension;
    nstatus = napi_get_value_int64(env, dimension_value, &dimension);
    ENSURE_NAPI_OK(env, nstatus);

    result->push_back(dimension);
  }
}
}  // namespace tfnodejs

#endif  // TF_NODEJS_UTILS_H_
