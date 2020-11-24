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

import {KernelConfig, KernelFunc, OnesLike, OnesLikeInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {fill} from './Fill';

export const onesLike =
    (args: {inputs: OnesLikeInputs, backend: MathBackendWebGL}): TensorInfo => {
      const {inputs, backend} = args;
      const {x} = inputs;

      if (x.dtype === 'string') {
        throw new Error('onesLike is not supported under string dtype');
      } else {
        // TODO(cais, smilkov): Add WebGL shader for onesLike:
        //   https://github.com/tensorflow/tfjs/issues/1293
        return fill(
            {attrs: {shape: x.shape, dtype: x.dtype, value: 1}, backend});
      }
    };

export const onesLikeConfig: KernelConfig = {
  kernelName: OnesLike,
  backendName: 'webgl',
  kernelFunc: onesLike as {} as KernelFunc
};
