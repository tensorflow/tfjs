/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

type GLSL = {
  defineSpecialNaN: string
};

export function getGlslDifferences(): GLSL {
  const defineSpecialNaN = `
      bool isnan_custom(float val) {
        // logical or has undefined behavior, https://bugs.chromium.org/p/tint/issues/detail?id=976.
        if (val > 0.0) {
          return false;
        }
        if (val < 0.0) {
          return false;
        }
        if (val == 0.0) {
          return false;
        }
        return true;
      }

      bvec4 isnan_custom(vec4 val) {
        return bvec4(isnan_custom(val.x),
          isnan_custom(val.y), isnan_custom(val.z), isnan_custom(val.w));
      }

      #define isnan(value) isnan_custom(value)
    `;

  return {defineSpecialNaN};
}
