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
import {Conv2DInfo} from '../../../ops/conv_util';
import {DataType, TypedArray} from '../../../types';
import {computeStrides} from '../../../util';
import {MathBackendCPU} from '../backend_cpu';

export function maxPoolWithArgmaxImpl(
    xValues: TypedArray, xShape: number[], dtype: DataType,
    includeBatchInIndex: boolean, convInfo: Conv2DInfo,
    backend: MathBackendCPU) {
  const strides = computeStrides(xShape);
  const maxPools =
      backend.pool(xValues, xShape, dtype, strides, convInfo, 'max');
  const maxPositions = backend.maxPoolPositions(
      xValues, xShape, dtype, convInfo, true, includeBatchInIndex);

  return [maxPools.values, maxPositions.values];
}
