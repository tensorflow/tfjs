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

import {KernelConfig, ScatterNd, ScatterNdAttrs, ScatterNdInputs, tensor1d} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const scatterNdConfig: KernelConfig = {
  kernelName: ScatterNd,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {indices, updates} = args.inputs as ScatterNdInputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {shape} = args.attrs as {} as ScatterNdAttrs;

    const opAttrs = [
      createTensorsTypeOpAttr('T', updates.dtype),
      createTensorsTypeOpAttr('Tindices', 'int32')
    ];
    const shapeTensor = tensor1d(shape, 'int32');
    const ret = backend.executeSingleOutput(
        ScatterNd, opAttrs, [indices, updates, shapeTensor]);

    shapeTensor.dispose();

    return ret;
  }
};
