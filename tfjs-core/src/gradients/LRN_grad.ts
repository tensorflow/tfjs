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
import {ENGINE} from '../engine';
import {LRN, LRNAttrs} from '../kernel_names';
import {GradConfig, NamedAttrMap} from '../kernel_registry';
import {Tensor, Tensor4D} from '../tensor';

export const lrnGradConfig: GradConfig = {
  kernelName: LRN,
  inputsToSave: ['x'],
  outputsToSave: [true],
  gradFunc: (dy: Tensor4D, saved: Tensor[], attrs: NamedAttrMap) => {
    const [x, y] = saved;
    const {depthRadius, bias, alpha, beta} = attrs as {} as LRNAttrs;

    return {
      x: () => ENGINE.runKernelFunc(
          backend => backend.LRNGrad(
              dy, x as Tensor4D, y as Tensor4D, depthRadius, bias, alpha, beta),
          {})
    };
  }
};
