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
import {SpaceToDepth, SpaceToDepthAttrs, SpaceToDepthInputs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor4D} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike4D} from '../types';
import * as util from '../util';

import {op} from './operation';

function spaceToDepth_(
    x: Tensor4D|TensorLike4D, blockSize: number,
    dataFormat: 'NHWC'|'NCHW' = 'NHWC'): Tensor4D {
  const $x = convertToTensor(x, 'x', 'spaceToDepth') as Tensor4D;

  const channelsLast = dataFormat === 'NHWC';

  const inputHeight = channelsLast ? $x.shape[1] : $x.shape[2];
  const inputWidth = channelsLast ? $x.shape[2] : $x.shape[3];

  util.assert(blockSize >= 2, () => `blockSize must be >= 2, got ${blockSize}`);

  util.assert(
      inputHeight % blockSize === 0,
      () => `inputHeight must be divisible by blockSize, got inputHeight=${
          inputHeight}, blockSize=${blockSize}`);

  util.assert(
      inputWidth % blockSize === 0,
      () => `inputWidth must be divisible by blockSize, got inputWidth=${
          inputWidth}, blockSize=${blockSize}`);

  const forward: ForwardFunc<Tensor4D> = backend =>
      backend.spaceToDepth($x, blockSize, dataFormat);

  const inputs: SpaceToDepthInputs = {x: $x};
  const attrs: SpaceToDepthAttrs = {blockSize, dataFormat};

  return ENGINE.runKernelFunc(
      forward, inputs as {} as NamedTensorMap, null /* gradient */,
      SpaceToDepth, attrs as {} as NamedAttrMap);
}

export const spaceToDepth = op({depthToSpace_: spaceToDepth_});
