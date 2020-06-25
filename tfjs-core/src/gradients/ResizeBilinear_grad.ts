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
import {ENGINE, ForwardFunc} from '../engine';
import {ResizeBilinear, ResizeBilinearAttrs, ResizeBilinearGrad, ResizeBilinearGradInputs} from '../kernel_names';
import {GradConfig} from '../kernel_registry';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor, Tensor4D} from '../tensor';
import {NamedTensorMap} from '../tensor_types';

export const resizeBilinearGradConfig: GradConfig = {
  kernelName: ResizeBilinear,
  inputsToSave: ['images'],
  gradFunc: (dy: Tensor4D, saved: Tensor[], attrs: NamedAttrMap) => {
    const [images] = saved;

    const backPropKernelFunc: ForwardFunc<Tensor> = (backend) => {
      const {alignCorners} = attrs as {} as ResizeBilinearAttrs;
      return backend.resizeBilinearBackprop(
          dy, images as Tensor4D, alignCorners);
    };

    const inputs: ResizeBilinearGradInputs = {images};
    const imagesDer = () => ENGINE.runKernelFunc(
        backPropKernelFunc, inputs as {} as NamedTensorMap, null /* gradient */,
        ResizeBilinearGrad, attrs);

    return {images: imagesDer};
  }
};
