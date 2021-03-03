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

import {env, KernelConfig, KernelFunc, LRN, LRNAttrs, LRNInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {LRNProgram} from '../lrn_gpu';
import {LRNPackedProgram} from '../lrn_packed_gpu';

export const lrn =
    (args: {inputs: LRNInputs, backend: MathBackendWebGL, attrs: LRNAttrs}):
        TensorInfo => {
          const {inputs, backend, attrs} = args;
          const {x} = inputs;
          const {depthRadius, bias, alpha, beta} = attrs;

          const program = env().getBool('WEBGL_PACK_NORMALIZATION') ?
              new LRNPackedProgram(x.shape, depthRadius, bias, alpha, beta) :
              new LRNProgram(x.shape, depthRadius, bias, alpha, beta);
          return backend.runWebGLProgram(program, [x], x.dtype);
        };

// tslint:disable-next-line: variable-name
export const LRNConfig: KernelConfig = {
  kernelName: LRN,
  backendName: 'webgl',
  kernelFunc: lrn as {} as KernelFunc
};
