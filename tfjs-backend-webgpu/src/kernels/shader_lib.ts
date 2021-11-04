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

export const idiv = `
  fn idiv(a: i32, b: i32, sign: f32) -> i32 {
    var res: i32 = a / b;
    let mod: i32 = a % b;
    if (sign < 0. && mod != 0) {
      res = res - 1;
    }
    return res;
  }`;

export const isNanCustom = `
  fn isNanCustom(val : f32) -> bool {
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
  }`;

export const isNanCustomVec4F32 = isNanCustom + `
  fn isNanCustomVec4F32(val : vec4<f32>) -> vec4<f32> {
    var res = vec4<f32> (0.0);
    for (var i = 0u; i < 4u; i = i + 1u) {
      if (isNanCustom(val[i])) {
        res[i] = 1.0;
      } else {
        res[i] = 0.0;
      }
    }
    return res;
  }`;

export const idivAndIsNanCustom = idiv + isNanCustomVec4F32;

export const getGlobalIndex = `
  // Only used when the y/z dimension of workgroup size is 1.
  fn getGlobalIndex(globalId : vec3<u32>, localId : vec3<u32>, numWorkgroups: vec3<u32>) -> i32 {
    if (numWorkgroups.y == 1u && numWorkgroups.z == 1u) {
      return i32(globalId.x);
    }

    let localInvocationIndex = localId.z * workGroupSizeX * workGroupSizeY +
        localId.y * workGroupSizeX + localId.x;
    let workGroupID = (globalId - localId)/vec3<u32>(
        workGroupSizeX, workGroupSizeY, workGroupSizeZ);

    return i32((workGroupID.z * numWorkgroups.x * numWorkgroups.y +
      workGroupID.y * numWorkgroups.x + workGroupID.x) *
      (workGroupSizeX * workGroupSizeY * workGroupSizeZ) +
      localInvocationIndex);
  }`;
