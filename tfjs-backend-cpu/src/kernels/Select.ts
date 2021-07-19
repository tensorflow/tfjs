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

import {KernelConfig, KernelFunc, Select, SelectInputs, TensorInfo, TypedArray, upcastType, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

export function select(args: {inputs: SelectInputs, backend: MathBackendCPU}):
    TensorInfo {
  const {inputs, backend} = args;
  const {condition, t, e} = inputs;

  assertNotComplex([condition, t, e], 'select');
  const conditionRank = condition.shape.length;

  const values = backend.data.get(condition.dataId).values as TypedArray;
  const tValues = backend.data.get(t.dataId).values as TypedArray;
  const eValues = backend.data.get(e.dataId).values as TypedArray;
  const resultDtype = upcastType(t.dtype, e.dtype);
  const newValues =
      util.makeZerosTypedArray(util.sizeFromShape(t.shape), resultDtype);

  let index = 0;
  const offset =
      conditionRank === 0 || conditionRank > 1 || t.shape.length === 1 ?
      1 :
      util.sizeFromShape(t.shape.slice(1));

  for (let i = 0; i < values.length; i++) {
    for (let j = 0; j < offset; j++) {
      if (values[i] === 1) {
        newValues[index++] = tValues[i];
      } else {
        newValues[index++] = eValues[i];
      }
    }
  }

  return backend.makeTensorInfo(t.shape, resultDtype, newValues);
}

export const selectConfig: KernelConfig = {
  kernelName: Select,
  backendName: 'cpu',
  kernelFunc: select as {} as KernelFunc
};
