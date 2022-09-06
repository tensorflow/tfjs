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

import {backend_util, GatherV2, GatherV2Attrs, GatherV2Inputs, KernelConfig, scalar, Tensor, TypedArray, util} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const gatherV2Config: KernelConfig = {
  kernelName: GatherV2,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {x, indices} = args.inputs as GatherV2Inputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {axis, batchDims} = args.attrs as {} as GatherV2Attrs;

    // Throw error when any index is out of bound.
    const indicesVals = backend.readSync(indices.dataId) as TypedArray;
    const axisDim = x.shape[axis];
    for (let i = 0; i < indicesVals.length; ++i) {
      const index = indicesVals[i];
      util.assert(
          index <= axisDim - 1 && index >= 0,
          () => `GatherV2: the index value ${index} is not in [0, ${
              axisDim - 1}]`);
    }

    // validate the inputs
    backend_util.segment_util.collectGatherOpShapeInfo(
        x as Tensor, indices as Tensor, axis, batchDims);

    const axisTensor = scalar(axis, 'int32');
    const opAttrs = [
      {name: 'batch_dims', type: backend.binding.TF_ATTR_INT, value: batchDims},
      createTensorsTypeOpAttr('Tparams', x.dtype),
      createTensorsTypeOpAttr('Tindices', indices.dtype),
      createTensorsTypeOpAttr('Taxis', 'int32')
    ];

    const res = backend.executeSingleOutput(
        GatherV2, opAttrs, [x, indices, axisTensor]);
    axisTensor.dispose();
    return res;
  }
};
