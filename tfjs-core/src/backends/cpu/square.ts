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

import {registerKernel, TensorInfo} from '../../kernel_registry';
import {CPUStorage} from './cpu_types';
import {assertNotComplex} from './cpu_util';

interface SquareInputs {
  x: TensorInfo;
}

registerKernel('Square', 'cpu', ({inputs, storage, save}) => {
  const {x} = inputs as {} as SquareInputs;
  const cpuStorage = storage as CPUStorage;
  assertNotComplex(x, 'square');

  // Save it for the gradient.
  save([x]);

  const values = cpuStorage.data.get(x.dataId).values as Float32Array;
  const newValues = new Float32Array(values.length);
  for (let i = 0; i < values.length; ++i) {
    const value = values[i];
    newValues[i] = value * value;
  }
  const dataId = cpuStorage.write(newValues, x.shape, x.dtype);
  return {dataId, shape: x.shape, dtype: x.dtype};
});
