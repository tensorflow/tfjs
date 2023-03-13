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

import {KernelConfig, ResizeNearestNeighborGrad, ResizeNearestNeighborGradAttrs, ResizeNearestNeighborGradInputs, tensor1d} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const resizeNearestNeighborGradConfig: KernelConfig = {
  kernelName: ResizeNearestNeighborGrad,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {images, dy} = args.inputs as ResizeNearestNeighborGradInputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {alignCorners} =
        args.attrs as unknown as ResizeNearestNeighborGradAttrs;

    const opAttrs = [
      createTensorsTypeOpAttr('T', images.dtype), {
        name: 'align_corners',
        type: backend.binding.TF_ATTR_BOOL,
        value: alignCorners
      }
    ];
    const [, origHeight, origWidth, ] = images.shape;
    const sizeTensor = tensor1d([origHeight, origWidth], 'int32');
    const res = backend.executeSingleOutput(
        ResizeNearestNeighborGrad, opAttrs, [dy, sizeTensor]);
    sizeTensor.dispose();
    return res;
  }
};
