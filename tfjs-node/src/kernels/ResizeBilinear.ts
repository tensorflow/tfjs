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

import {KernelConfig, ResizeBilinear, ResizeBilinearAttrs, ResizeBilinearInputs, tensor1d} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const resizeBilinearConfig: KernelConfig = {
  kernelName: ResizeBilinear,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {images} = args.inputs as ResizeBilinearInputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {alignCorners, halfPixelCenters, size} =
        args.attrs as {} as ResizeBilinearAttrs;

    const opAttrs = [
      createTensorsTypeOpAttr('T', images.dtype),
      {
        name: 'align_corners',
        type: backend.binding.TF_ATTR_BOOL,
        value: alignCorners
      },
      {
        name: 'half_pixel_centers',
        type: backend.binding.TF_ATTR_BOOL,
        value: halfPixelCenters
      },
    ];
    const [newHeight, newWidth] = size;
    const sizeTensor = tensor1d([newHeight, newWidth], 'int32');
    const ret = backend.executeSingleOutput(
        ResizeBilinear, opAttrs, [images, sizeTensor]);
    sizeTensor.dispose();
    return ret;
  }
};
