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
import {ENGINE} from '../engine';
import {Dilation2D, Dilation2DBackpropFilter, Dilation2DBackpropFilterInputs, Dilation2DBackpropInput, Dilation2DBackpropInputInputs} from '../kernel_names';
import {GradConfig} from '../kernel_registry';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor, Tensor3D, Tensor4D} from '../tensor';
import {NamedTensorMap} from '../tensor_types';

export const dilation2dGradConfig: GradConfig = {
  kernelName: Dilation2D,
  inputsToSave: ['x', 'filter'],
  gradFunc: (dy: Tensor4D, saved: Tensor[], attrs: NamedAttrMap) => {
    const [x, filter] = saved as [Tensor4D, Tensor3D];

    const inputInputs: Dilation2DBackpropInputInputs = {x, filter, dy};
    const filterInputs: Dilation2DBackpropFilterInputs = {x, filter, dy};

    return {
      x: () => ENGINE.runKernel(
                   Dilation2DBackpropInput, inputInputs as {} as NamedTensorMap,
                   attrs) as Tensor,
      filter: () => ENGINE.runKernel(
                        Dilation2DBackpropFilter,
                        filterInputs as {} as NamedTensorMap, attrs) as Tensor
    };
  }
};
