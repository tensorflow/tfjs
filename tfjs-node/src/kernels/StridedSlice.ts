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

import {KernelConfig, StridedSlice, StridedSliceAttrs, StridedSliceInputs, tensor1d, tidy} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const stridedSliceConfig: KernelConfig = {
  kernelName: StridedSlice,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {x} = args.inputs as StridedSliceInputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask} =
        args.attrs as unknown as StridedSliceAttrs;

    const attrs = args.attrs as unknown as StridedSliceAttrs;
    // make a copy because it may be modified in-place further down.
    const begin = attrs.begin.slice();
    const end = attrs.end.slice();
    const strides = attrs.strides;

    return tidy(() => {
      const beginTensor = tensor1d(begin, 'int32');
      const endTensor = tensor1d(end, 'int32');
      const stridesTensor = tensor1d(strides, 'int32');

      const opAttrs = [
        createTensorsTypeOpAttr('T', x.dtype),
        createTensorsTypeOpAttr('Index', 'int32'), {
          name: 'begin_mask',
          type: backend.binding.TF_ATTR_INT,
          value: beginMask
        },
        {name: 'end_mask', type: backend.binding.TF_ATTR_INT, value: endMask}, {
          name: 'ellipsis_mask',
          type: backend.binding.TF_ATTR_INT,
          value: ellipsisMask
        },
        {
          name: 'new_axis_mask',
          type: backend.binding.TF_ATTR_INT,
          value: newAxisMask
        },
        {
          name: 'shrink_axis_mask',
          type: backend.binding.TF_ATTR_INT,
          value: shrinkAxisMask
        }
      ];
      return backend.executeSingleOutput(
          StridedSlice, opAttrs, [x, beginTensor, endTensor, stridesTensor]);
    });
  }
};
