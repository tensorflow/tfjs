/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import {decodeString, encodeString} from '../util';

// Utilities needed by backend consumers of tf-core.
export * from '../ops/axis_util';
export * from '../ops/broadcast_util';
export * from '../ops/concat_util';
export * from '../ops/conv_util';
export * from '../ops/fused_util';
export * from '../ops/fused_types';
export * from '../ops/reduce_util';

import * as slice_util from '../ops/slice_util';
export {slice_util};

export {BackendValues, TypedArray, upcastType, PixelData} from '../types';
export {MemoryInfo, TimingInfo} from '../engine';
export * from '../ops/rotate_util';
export * from '../ops/array_ops_util';
export * from '../ops/gather_nd_util';
export * from '../ops/scatter_nd_util';
export * from '../ops/selu_util';
export * from '../ops/fused_util';
export * from '../ops/erf_util';
export * from '../log';
export * from '../backends/complex_util';
export * from '../backends/einsum_util';
export * from '../ops/split_util';

import * as segment_util from '../ops/segment_util';
export {segment_util};

export function fromUint8ToStringArray(vals: Uint8Array[]) {
  try {
    // Decode the bytes into string.
    return vals.map(val => decodeString(val));
  } catch (err) {
    throw new Error(
        `Failed to decode encoded string bytes into utf-8, error: ${err}`);
  }
}

export function fromStringArrayToUint8(strings: string[]) {
  return strings.map(s => encodeString(s));
}
