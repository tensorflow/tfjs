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
import {DepthwiseConv2dNativeBackpropInput, DepthwiseConv2dNativeBackpropInputAttrs, DepthwiseConv2dNativeBackpropInputInputs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor3D, Tensor4D} from '../tensor';
import {NamedTensorMap} from '../tensor_types';

import {op} from './operation';
import {reshape} from './reshape';

function depthwiseConv2dNativeBackpropInput_<T extends Tensor3D|Tensor4D>(
    xShape: [number, number, number, number], dy: T, filter: Tensor4D,
    strides: [number, number]|number, pad: 'valid'|'same'|number,
    dilations: [number, number]|number = [1, 1],
    dimRoundingMode?: 'floor'|'round'|'ceil'): T {
  let dy4D = dy as Tensor4D;
  let reshapedTo4D = false;
  if (dy.rank === 3) {
    reshapedTo4D = true;
    dy4D = reshape(dy, [1, dy.shape[0], dy.shape[1], dy.shape[2]]);
  }

  const inputs: DepthwiseConv2dNativeBackpropInputInputs = {dy: dy4D, filter};
  const attrs: DepthwiseConv2dNativeBackpropInputAttrs =
      {strides, pad, dimRoundingMode, dilations, inputShape: xShape};

  const res =
      // tslint:disable-next-line: no-unnecessary-type-assertion
      ENGINE.runKernel(
          DepthwiseConv2dNativeBackpropInput, inputs as {} as NamedTensorMap,
          attrs as {} as NamedAttrMap) as T;

  if (reshapedTo4D) {
    return reshape(res, [res.shape[1], res.shape[2], res.shape[3]]) as T;
  }
  return res;
}

export const depthwiseConv2dNativeBackpropInput =
    op({depthwiseConv2dNativeBackpropInput_});
