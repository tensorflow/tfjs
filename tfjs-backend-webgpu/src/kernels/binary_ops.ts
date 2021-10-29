/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import {backend_util, util} from '@tensorflow/tfjs-core';

import {BinaryOpSharedProgram} from './binary_op_shared_webgpu';
import {BinaryOpType} from './binary_op_util';
import {BinaryOpVec4Program} from './binary_op_vec4_webgpu';
import {BinaryOpProgram} from './binary_op_webgpu';

export function getBinaryProgram(
    op: BinaryOpType, aShape: number[], bShape: number[]) {
  const outputShape = backend_util.assertAndGetBroadcastShape(aShape, bShape);
  const useVec4 =
      util.arraysEqual(aShape, bShape) && util.sizeFromShape(aShape) % 4 === 0;
  if (useVec4) {
    return new BinaryOpVec4Program(op, outputShape);
  }
  const useSharedMemoryWithA =
      aShape.length === 1 && bShape.length > 1 && aShape[0] < 512;
  const useSharedMemoryWithB =
      bShape.length === 1 && aShape.length > 1 && bShape[0] < 512;
  if (useSharedMemoryWithA || useSharedMemoryWithB) {
    const isScater = useSharedMemoryWithB ? bShape[0] === 1 : aShape[0] === 1;
    return new BinaryOpSharedProgram(
        op, outputShape, useSharedMemoryWithB, isScater);
  } else {
    return new BinaryOpProgram(op, outputShape);
  }
}
