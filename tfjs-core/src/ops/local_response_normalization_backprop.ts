/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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
import {LRNBackprop, LRNBackpropAttrs, LRNBackpropInputs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor, Tensor4D} from '../tensor';
import {NamedTensorMap} from '../tensor_types';

import {op} from './operation';

function localResponseNormalizationBackprop_<T extends Tensor4D>(
    x: T, y: T, dy: T, depthRadius = 5, bias = 1, alpha = 1, beta = 0.5): T {
  const forward: ForwardFunc<Tensor> = backend =>
      backend.LRNGrad(dy, x, y, depthRadius, bias, alpha, beta);

  const inputs: LRNBackpropInputs = {x, y, dy};

  const attrs: LRNBackpropAttrs = {depthRadius, bias, alpha, beta};

  return ENGINE.runKernelFunc(
             forward, inputs as {} as NamedTensorMap, null /* grad */,
             LRNBackprop, attrs as {} as NamedAttrMap) as T;
}

export const localResponseNormalizationBackprop =
    op({localResponseNormalizationBackprop_});
