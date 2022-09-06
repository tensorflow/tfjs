/* Copyright 2021 Google LLC. All Rights Reserved.
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
 * ===========================================================================*/

#ifndef SIN_COS_WORKAROUND_H_
#define SIN_COS_WORKAROUND_H_

// Workaround a bug related to sin/cos with emscripten/webkit on iOS 11/12:
// https://github.com/emscripten-core/emscripten/issues/13130
namespace tfjs {
namespace sin_cos_workaround {

float sin_fixed(float x);

float cos_fixed(float x);

float tan_fixed(float x);

}  // namespace sin_cos_workaround
}  // namespace tfjs
#endif  // SIN_COS_WORKAROUND_H_
