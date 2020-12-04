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

import {env, KernelConfig, KernelFunc, LeakyRelu, LeakyReluAttrs, LeakyReluInputs, TensorInfo, util} from '@tensorflow/tfjs-core';
import {MathBackendWebGL} from '../backend_webgl';
import {BinaryOpProgram} from '../binaryop_gpu';
import {BinaryOpPackedProgram} from '../binaryop_packed_gpu';

export const LEAKYRELU = `return (a < 0.) ? b * a : a;`;
export const LEAKYRELU_PACKED = `
  vec4 aLessThanZero = vec4(lessThan(a, vec4(0.)));
  return (aLessThanZero * (b * a)) + ((vec4(1.0) - aLessThanZero) * a);
`;

export function leakyRelu(args: {
  inputs: LeakyReluInputs,
  backend: MathBackendWebGL,
  attrs: LeakyReluAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {alpha} = attrs;

  const $alpha = backend.makeTensorInfo(
      [], 'float32',
      util.createScalarValue(alpha as {} as 'float32', 'float32'));

  const program = env().getBool('WEBGL_PACK_BINARY_OPERATIONS') ?
      new BinaryOpPackedProgram(LEAKYRELU_PACKED, x.shape, $alpha.shape) :
      new BinaryOpProgram(LEAKYRELU, x.shape, $alpha.shape);
  const result = backend.runWebGLProgram(program, [x, $alpha], x.dtype);

  backend.disposeIntermediateTensorInfo($alpha);

  return result;
}

export const leakyReluConfig: KernelConfig = {
  kernelName: LeakyRelu,
  backendName: 'webgl',
  kernelFunc: leakyRelu as {} as KernelFunc
};
