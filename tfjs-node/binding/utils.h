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

#include <node_api.h>
#include <stdarg.h>
#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "tensorflow/c/c_api.h"
#include "tf_auto_status.h"

#define MAX_TENSOR_SHAPE 4

#define ARRAY_SIZE(array) (sizeof(array) / sizeof(array[0]))

#ifndef DEBUG
#define DEBUG 0
#endif

#define DEBUG_LOG(message, file, line_number)                             \
  do {                                                                    \
    if (DEBUG)                                                            \
      fprintf(stderr, "** -%s:%zu\n-- %s\n", file, line_number, message); \
  } while (0)

namespace tfnodejs {

#define NAPI_THROW_ERROR(env, message, ...) \
  NapiThrowError(env, __FILE__, __LINE__, message, ##__VA_ARGS__);

inline void NapiThrowError(napi_env env, const char *file,
                           const size_t line_number, const char *message, ...) {
  char buffer[500];
  va_list args;
  va_start(args, message);
  std::vsnprintf(buffer, 500, message, args);
  va_end(args);
  DEBUG_LOG(buffer, file, line_number);
  napi_throw_error(env, nullptr, buffer);
}

#define ENSURE_NAPI_OK(env, status) \
  if (!EnsureNapiOK(env, status, __FILE__, __LINE__)) return;
#define ENSURE_NAPI_OK_RETVAL(env, status, retval) \
  if (!EnsureNapiOK(env, status, __FILE__, __LINE__)) return retval;

inline bool EnsureNapiOK(napi_env env, napi_status status, const char *file,
                         const size_t line_number) {
  if (status != napi_ok) {
    const napi_extended_error_info *error_info = 0;
    napi_get_last_error_info(env, &error_info);
    NapiThrowError(
        env, file, line_number, "Invalid napi_status: %s\n",
        error_info->error_message ? error_info->error_message : "unknown");
  }
  return status == napi_ok;
}

#define ENSURE_TF_OK(env, status) \
  if (!EnsureTFOK(env, status, __FILE__, __LINE__)) return;
#define ENSURE_TF_OK_RETVAL(env, status, retval) \
  if (!EnsureTFOK(env, status, __FILE__, __LINE__)) return retval;

inline bool EnsureTFOK(napi_env env, TF_AutoStatus &status, const char *file,
                       const size_t line_number) {
  TF_Code tf_code = TF_GetCode(status.status);
  if (tf_code != TF_OK) {
    NapiThrowError(env, file, line_number, "Invalid TF_Status: %u\nMessage: %s",
                   TF_GetCode(status.status), TF_Message(status.status));
  }
  return tf_code == TF_OK;
}

#define ENSURE_CONSTRUCTOR_CALL(env, info) \
  if (!EnsureConstructorCall(env, info, __FILE__, __LINE__)) return;
#define ENSURE_CONSTRUCTOR_CALL_RETVAL(env, info, retval) \
  if (!EnsureConstructorCall(env, info, __FILE__, __LINE__)) return retval;

inline bool EnsureConstructorCall(napi_env env, napi_callback_info info,
                                  const char *file, const size_t line_number) {
  napi_value js_target;
  napi_status nstatus = napi_get_new_target(env, info, &js_target);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, false);
  bool is_target = js_target != nullptr;
  if (!is_target) {
    NapiThrowError(env, file, line_number,
                   "Function not used as a constructor!");
  }
  return is_target;
}

#define ENSURE_VALUE_IS_OBJECT(env, value) \
  if (!EnsureValueIsObject(env, value, __FILE__, __LINE__)) return;
#define ENSURE_VALUE_IS_OBJECT_RETVAL(env, value, retval) \
  if (!EnsureValueIsObject(env, value, __FILE__, __LINE__)) return retval;

inline bool EnsureValueIsObject(napi_env env, napi_value value,
                                const char *file, const size_t line_number) {
  napi_valuetype type;
  ENSURE_NAPI_OK_RETVAL(env, napi_typeof(env, value, &type), false);
  bool is_object = type == napi_object;
  if (!is_object) {
    NapiThrowError(env, file, line_number, "Argument is not an object!");
  }
  return is_object;
}

#define ENSURE_VALUE_IS_STRING(env, value) \
  if (!EnsureValueIsString(env, value, __FILE__, __LINE__)) return;
#define ENSURE_VALUE_IS_STRING_RETVAL(env, value, retval) \
  if (!EnsureValueIsString(env, value, __FILE__, __LINE__)) return retval;

inline bool EnsureValueIsString(napi_env env, napi_value value,
                                const char *file, const size_t line_number) {
  napi_valuetype type;
  ENSURE_NAPI_OK_RETVAL(env, napi_typeof(env, value, &type), false);
  bool is_string = type == napi_string;
  if (!is_string) {
    NapiThrowError(env, file, line_number, "Argument is not a string!");
  }
  return is_string;
}

#define ENSURE_VALUE_IS_NUMBER(env, value) \
  if (!EnsureValueIsNumber(env, value, __FILE__, __LINE__)) return;
#define ENSURE_VALUE_IS_NUMBER_RETVAL(env, value, retval) \
  if (!EnsureValueIsNumber(env, value, __FILE__, __LINE__)) return retval;

inline bool EnsureValueIsNumber(napi_env env, napi_value value,
                                const char *file, const size_t line_number) {
  napi_valuetype type;
  ENSURE_NAPI_OK_RETVAL(env, napi_typeof(env, value, &type), false);
  bool is_number = type == napi_number;
  if (!is_number) {
    NapiThrowError(env, file, line_number, "Argument is not a string!");
  }
  return is_number;
}

#define ENSURE_VALUE_IS_ARRAY(env, value) \
  if (!EnsureValueIsArray(env, value, __FILE__, __LINE__)) return;
#define ENSURE_VALUE_IS_ARRAY_RETVAL(env, value, retval) \
  if (!EnsureValueIsArray(env, value, __FILE__, __LINE__)) return retval;

inline bool EnsureValueIsArray(napi_env env, napi_value value, const char *file,
                               const size_t line_number) {
  bool is_array;
  ENSURE_NAPI_OK_RETVAL(env, napi_is_array(env, value, &is_array), false);
  if (!is_array) {
    NapiThrowError(env, file, line_number, "Argument is not an array!");
  }
  return is_array;
}

#define ENSURE_VALUE_IS_TYPED_ARRAY(env, value) \
  if (!EnsureValueIsTypedArray(env, value, __FILE__, __LINE__)) return;
#define ENSURE_VALUE_IS_TYPED_ARRAY_RETVAL(env, value, retval) \
  if (!EnsureValueIsTypedArray(env, value, __FILE__, __LINE__)) return retval;

inline bool EnsureValueIsTypedArray(napi_env env, napi_value value,
                                    const char *file,
                                    const size_t line_number) {
  bool is_array;
  ENSURE_NAPI_OK_RETVAL(env, napi_is_typedarray(env, value, &is_array), false);
  if (!is_array) {
    NapiThrowError(env, file, line_number, "Argument is not a typed-array!");
  }
  return is_array;
}

#define ENSURE_VALUE_IS_LESS_THAN(env, value, max) \
  if (!EnsureValueIsLessThan(env, value, max, __FILE__, __LINE__)) return;
#define ENSURE_VALUE_IS_LESS_THAN_RETVAL(env, value, max, retval)  \
  if (!EnsureValueIsLessThan(env, value, max, __FILE__, __LINE__)) \
    return retval;

inline bool EnsureValueIsLessThan(napi_env env, uint32_t value, uint32_t max,
                                  const char *file, const size_t line_number) {
  if (value > max) {
    NapiThrowError(env, file, line_number,
                   "Argument is greater than max: %u > %u", value, max);
    return false;
  } else {
    return true;
  }
}

#define REPORT_UNKNOWN_TF_DATA_TYPE(env, type) \
  ReportUnknownTFDataType(env, type, __FILE__, __LINE__)

inline void ReportUnknownTFDataType(napi_env env, TF_DataType type,
                                    const char *file,
                                    const size_t line_number) {
  NapiThrowError(env, file, line_number, "Unhandled TF_DataType: %u\n", type);
}

#define REPORT_UNKNOWN_TF_ATTR_TYPE(env, type) \
  ReportUnknownTFAttrType(env, type, __FILE__, __LINE__)

inline void ReportUnknownTFAttrType(napi_env env, TF_AttrType type,
                                    const char *file,
                                    const size_t line_number) {
  NapiThrowError(env, file, line_number, "Unhandled TF_AttrType: %u\n", type);
}

#define REPORT_UNKNOWN_TYPED_ARRAY_TYPE(env, type) \
  ReportUnknownTypedArrayType(env, type, __FILE__, __LINE__)

inline void ReportUnknownTypedArrayType(napi_env env, napi_typedarray_type type,
                                        const char *file,
                                        const size_t line_number) {
  NapiThrowError(env, file, line_number, "Unhandled napi typed_array_type: %u",
                 type);
}

// Returns a vector with the shape values of an array.
inline void ExtractArrayShape(napi_env env, napi_value array_value,
                              std::vector<int64_t> *result) {
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

inline bool IsExceptionPending(napi_env env) {
  bool has_exception = false;
  ENSURE_NAPI_OK_RETVAL(env, napi_is_exception_pending(env, &has_exception),
                        has_exception);
  return has_exception;
}

#define ENSURE_VALUE_IS_NOT_NULL(env, value) \
  if (!EnsureValueIsNotNull(env, value, __FILE__, __LINE__)) return;
#define ENSURE_VALUE_IS_NOT_NULL_RETVAL(env, value, retval) \
  if (!EnsureValueIsNotNull(env, value, __FILE__, __LINE__)) return retval;

inline bool EnsureValueIsNotNull(napi_env env, void *value, const char *file,
                                 const size_t line_number) {
  bool is_null = value == nullptr;
  if (is_null) {
    NapiThrowError(env, file, line_number, "Argument is null!");
  }
  return !is_null;
}

inline napi_status GetStringParam(napi_env env, napi_value string_value,
                                  std::string &string) {
  ENSURE_VALUE_IS_STRING_RETVAL(env, string_value, napi_invalid_arg);

  napi_status nstatus;

  size_t str_length;
  nstatus =
      napi_get_value_string_utf8(env, string_value, nullptr, 0, &str_length);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, nstatus);

  char *buffer = (char *)(malloc(sizeof(char) * (str_length + 1)));
  ENSURE_VALUE_IS_NOT_NULL_RETVAL(env, buffer, napi_generic_failure);

  nstatus = napi_get_value_string_utf8(env, string_value, buffer,
                                       str_length + 1, &str_length);
  ENSURE_NAPI_OK_RETVAL(env, nstatus, nstatus);

  string.assign(buffer, str_length);
  free(buffer);
  return napi_ok;
}

// Returns the number of elements in a Tensor.
inline size_t GetTensorNumElements(TF_Tensor *tensor) {
  size_t ret = 1;
  for (int i = 0; i < TF_NumDims(tensor); ++i) {
    ret *= TF_Dim(tensor, i);
  }
  return ret;
}

// Split a string into an array of characters array with `,` as delimiter.
inline std::vector<const char *> splitStringByComma(const std::string &str) {
  std::vector<const char *> tokens;
  size_t prev = 0, pos = 0;
  do {
    pos = str.find(',', prev);
    if (pos == std::string::npos) pos = str.length();
    std::string token = str.substr(prev, pos - prev);
    if (!token.empty()) {
      char *cstr = new char[str.length() + 1];
      std::strcpy(cstr, token.c_str());
      tokens.push_back(cstr);
    }
    prev = pos + 1;
  } while (pos < str.length() && prev < str.length());
  return tokens;
}

}  // namespace tfnodejs

#endif  // TF_NODEJS_UTILS_H_
