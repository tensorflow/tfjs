/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {DataId, registerKernel} from '../../kernel_registry';
import {BackendValues, DataType, TypedArray} from '../../types';
import {assertNotComplex} from './cpu_util';

export interface CPUStorage {
  readSync(dataId: DataId): BackendValues;
  read(dataId: DataId): Promise<BackendValues>;
  newData(dtype: DataType, values: BackendValues): DataId;
}

registerKernel('Square', 'cpu', ({inputs, storage}) => {
  const {x} = inputs;
  const cpu = storage as CPUStorage;
  assertNotComplex(x, 'square');

  const values = cpu.readSync(x.dataId) as TypedArray;
  const newValues = new Float32Array(values.length);
  for (let i = 0; i < values.length; ++i) {
    const value = values[i];
    newValues[i] = value * value;
  }
  const dataId = cpu.newData(x.dtype, newValues);
  return {dataId, shape: x.shape, dtype: x.dtype};
});
