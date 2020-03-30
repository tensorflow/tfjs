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

import {env} from '../../../environment';
import {TensorInfo} from '../../../kernel_registry';
import {DataType, NumericDataType, TypedArray} from '../../../types';
import * as util from '../../../util';
import {MathBackendWebGL} from '../backend_webgl';
import {TransposeProgram} from '../transpose_gpu';
import {TransposePackedProgram} from '../transpose_packed_gpu';

export function transposeImpl(
    x: TensorInfo, perm: number[], backend: MathBackendWebGL): TensorInfo {
  const program = env().getBool('WEBGL_PACK_ARRAY_OPERATIONS') ?
      new TransposePackedProgram(x.shape, perm) :
      new TransposeProgram(x.shape, perm);
  return backend.runWebGLProgram(program, [x], x.dtype);
}

// todo(@yassogba) import this from cpu backend once that package is published.
export function transposeImplCPU(
    xVals: TypedArray, xShape: number[], dtype: DataType, perm: number[],
    newShape: number[]): TypedArray {
  const xSize = util.sizeFromShape(xShape);
  const xRank = xShape.length;
  const xStrides = util.computeStrides(xShape);
  const newStrides = util.computeStrides(newShape);

  const result = util.getTypedArrayFromDType(
      dtype as NumericDataType, util.sizeFromShape(newShape));

  for (let i = 0; i < xSize; ++i) {
    const loc = util.indexToLoc(i, xRank, xStrides);

    // Permute location.
    const newLoc: number[] = new Array(loc.length);
    for (let i = 0; i < newLoc.length; i++) {
      newLoc[i] = loc[perm[i]];
    }

    const newIndex = util.locToIndex(newLoc, xRank, newStrides);
    result[newIndex] = xVals[i];
  }
  return result;
}
