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

import {env, KernelConfig, KernelFunc, PadV2, PadV2Attrs, PadV2Inputs, TensorInfo, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {PadProgram} from '../pad_gpu';
import {PadPackedProgram} from '../pad_packed_gpu';
import {fill} from './Fill';

export const padV2 =
    (args: {inputs: PadV2Inputs, backend: MathBackendWebGL, attrs: PadV2Attrs}):
        TensorInfo => {
          const {inputs, backend, attrs} = args;
          const {x} = inputs;
          const {paddings, constantValue} = attrs;

          if (util.sizeFromShape(x.shape) === 0) {
            // Short-circuit the computation, since x doesn't have value, only
            // the shape is used to compute output shape to pad.
            const outputShape = paddings.map(
                (p, i) =>
                    p[0] /* beforePad */ + x.shape[i] + p[1] /* afterPad */);
            return fill({
              backend,
              attrs: {shape: outputShape, value: constantValue, dtype: x.dtype}
            });
          }

          const program = env().getBool('WEBGL_PACK_ARRAY_OPERATIONS') ?
              new PadPackedProgram(x.shape, paddings, constantValue) :
              new PadProgram(x.shape, paddings, constantValue);
          const customValues = [[constantValue]];
          return backend.runWebGLProgram(program, [x], x.dtype, customValues);
        };

export const padV2Config: KernelConfig = {
  kernelName: PadV2,
  backendName: 'webgl',
  kernelFunc: padV2 as {} as KernelFunc
};
