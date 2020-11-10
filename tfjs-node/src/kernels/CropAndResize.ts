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

import {CropAndResize, CropAndResizeAttrs, CropAndResizeInputs, KernelConfig, tensor1d} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const cropAndResizeConfig: KernelConfig = {
  kernelName: CropAndResize,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {image, boxes, boxInd} = args.inputs as CropAndResizeInputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {cropSize, method, extrapolationValue} =
        args.attrs as {} as CropAndResizeAttrs;

    const opAttrs = [
      createTensorsTypeOpAttr('T', image.dtype),
      {name: 'method', type: backend.binding.TF_ATTR_STRING, value: method}, {
        name: 'extrapolation_value',
        type: backend.binding.TF_ATTR_FLOAT,
        value: extrapolationValue
      }
    ];
    const cropSizeTensor = tensor1d(cropSize, 'int32');
    const res = backend.executeSingleOutput(
        CropAndResize, opAttrs, [image, boxes, boxInd, cropSizeTensor]);
    cropSizeTensor.dispose();
    return res;
  }
};
