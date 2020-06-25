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
import {DepthwiseConv2dNativeBackpropFilter, DepthwiseConv2dNativeBackpropFilterInputs} from '../kernel_names';
import {Tensor, Tensor3D, Tensor4D} from '../tensor';
import {NamedTensorMap} from '../tensor_types';

import * as conv_util from './conv_util';
import {op} from './operation';
import {reshape} from './reshape';

function depthwiseConv2dNativeBackpropFilter_<T extends Tensor3D|Tensor4D>(
    x: T, dy: T, filterShape: [number, number, number, number],
    convInfo: conv_util.Conv2DInfo): Tensor4D {
  let x4D = x as Tensor4D;
  if (x.rank === 3) {
    x4D = reshape(x, [1, x.shape[0], x.shape[1], x.shape[2]]);
  }
  let dy4D = dy as Tensor4D;
  if (dy4D.rank === 3) {
    dy4D = reshape(dy, [1, dy.shape[0], dy.shape[1], dy.shape[2]]);
  }

  const forward: ForwardFunc<Tensor> = backend =>
      backend.depthwiseConv2DDerFilter(x4D, dy4D, convInfo);

  const inputs: DepthwiseConv2dNativeBackpropFilterInputs = {x: x4D, dy: dy4D};

  return ENGINE.runKernelFunc(
             forward, inputs as {} as NamedTensorMap, null,
             DepthwiseConv2dNativeBackpropFilter) as Tensor4D;
}

export const depthwiseConv2dNativeBackpropFilter =
    op({depthwiseConv2dNativeBackpropFilter_});
