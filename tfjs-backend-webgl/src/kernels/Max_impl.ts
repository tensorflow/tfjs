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

import {TensorInfo, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {reduce} from '../kernel_utils/reduce';
import {reshape} from '../kernel_utils/reshape';

export function maxImpl(
    x: TensorInfo, reduceShape: number[], outShape: number[],
    backend: MathBackendWebGL): TensorInfo {
  const inSize = util.sizeFromShape(reduceShape);
  const xSize = util.sizeFromShape(x.shape);
  const batchSize = xSize / inSize;
  const reshapedInput = reshape(x, [batchSize, inSize], backend);
  const reduced = reduce(reshapedInput, x.dtype, 'max', backend);

  if (reshapedInput.dataId !== x.dataId) {
    // dispose the output of the packed reshape.
    backend.disposeData(reshapedInput.dataId);
  }

  return reshape(reduced, outShape, backend);
}
