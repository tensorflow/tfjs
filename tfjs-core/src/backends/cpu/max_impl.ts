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

import {TensorInfo} from '../../kernel_registry';
import * as axis_util from '../../ops/axis_util';
import {sizeFromShape} from '../../util';

export const max = (x: TensorInfo, axes: number[]) => {
  axis_util.assertAxesAreInnerMostDims('max', axes, x.shape.length);
  const [outShape, reduceShape] =
      axis_util.computeOutAndReduceShapes(x.shape, axes);

  const aVals = cpuBackend.data.get(x.dataId).values as Float32Array;
  const vals = new Float32Array(sizeFromShape(outShape));
  const reduceSize = sizeFromShape(reduceShape);

  for (let i = 0; i < vals.length; ++i) {
    const offset = i * reduceSize;
    let max = aVals[offset];
    for (let j = 0; j < reduceSize; ++j) {
      const value = aVals[offset + j];
      if (value > max) {
        max = value;
      }
    }
    vals[i] = max;
  }

  const dataId = cpuBackend.write(vals, outShape, x.dtype);
  return {dataId, shape: x.shape, dtype: x.dtype};
};
