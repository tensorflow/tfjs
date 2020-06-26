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
import {Conv3D, Conv3DAttrs} from '../kernel_names';
import {GradConfig, NamedAttrMap} from '../kernel_registry';
import {conv3DBackpropFilter} from '../ops/conv3d_backprop_filter';
import {conv3DBackpropInput} from '../ops/conv3d_backprop_input';
import {tupleValuesAreOne} from '../ops/conv_util';
import {Tensor, Tensor5D} from '../tensor';
import * as util from '../util';

export const conv3DGradConfig: GradConfig = {
  kernelName: Conv3D,
  inputsToSave: ['x', 'filter'],
  gradFunc: (dy: Tensor5D, saved: Tensor[], attrs: NamedAttrMap) => {
    const {dilations, strides, pad} = attrs as {} as Conv3DAttrs;
    util.assert(
        tupleValuesAreOne(dilations),
        () =>
            'Error in gradient of conv3D: dilation rates greater than 1 are ' +
            `not yet supported in gradients. Got dilations '${dilations}'`);

    const [x5D, $filter] = saved;

    return {
      x: () => conv3DBackpropInput(
          (x5D as Tensor5D).shape, dy, $filter as Tensor5D, strides, pad),
      filter: () => conv3DBackpropFilter(
          x5D as Tensor5D, dy, ($filter as Tensor5D).shape, strides, pad)
    };
  }
};
