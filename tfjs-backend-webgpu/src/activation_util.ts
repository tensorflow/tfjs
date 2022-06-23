/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {backend_util, TensorInfo} from '@tensorflow/tfjs-core';
import {WebGPUBackend} from './backend_webgpu';

import {BinaryOpType, getBinaryOpString} from './binary_op_util';
import {BinaryOpProgram} from './binary_op_webgpu';
import {getUnaryOpString, UnaryOpType} from './unary_op_util';
import {UnaryOpProgram} from './unary_op_webgpu';

export function mapActivationToShaderProgram(
    activation: backend_util.Activation, packed = false): string {
  if (activation === null) {
    return null;
  } else if (activation === 'linear') {
    return getUnaryOpString(UnaryOpType.LINEAR);
  } else if (activation === 'relu') {
    return getUnaryOpString(UnaryOpType.RELU, packed);
  } else if (activation === 'elu') {
    return getUnaryOpString(UnaryOpType.ELU, packed);
  } else if (activation === 'relu6') {
    return getUnaryOpString(UnaryOpType.RELU6, packed);
  } else if (activation === 'prelu') {
    return getBinaryOpString(BinaryOpType.PRELU, packed);
  } else if (activation === 'sigmoid') {
    return getUnaryOpString(UnaryOpType.SIGMOID, packed);
  } else if (activation === 'leakyrelu') {
    return getUnaryOpString(UnaryOpType.LEAKYRELU, packed);
  }
  throw new Error(`Activation ${
      activation} has not been implemented for the WebGPU backend.`);
}

export function executeActivation(
    activation: backend_util.Activation, x: TensorInfo, backend: WebGPUBackend,
    leakyreluAlpha = 0, preluActivationWeights: TensorInfo = null): TensorInfo {
  let opType: UnaryOpType|BinaryOpType = UnaryOpType.RELU;
  if (activation === 'leakyrelu') {
    opType = UnaryOpType.LEAKYRELU;
    const program = new UnaryOpProgram(x.shape, opType as UnaryOpType);
    program.uniforms = 'alpha : f32,';
    const uniformData = [{type: 'float32', data: [leakyreluAlpha]}];
    return backend.runWebGPUProgram(program, [x], x.dtype, uniformData);
  }

  if (activation === 'prelu') {
    opType = BinaryOpType.PRELU;
    const program = new BinaryOpProgram(
        opType as BinaryOpType, x.shape, preluActivationWeights.shape);
    return backend.runWebGPUProgram(
        program, [x, preluActivationWeights], x.dtype);
  }

  if (activation === 'linear') {
    opType = UnaryOpType.LINEAR;
  } else if (activation === 'relu') {
    opType = UnaryOpType.RELU;
  } else if (activation === 'elu') {
    opType = UnaryOpType.ELU;
  } else if (activation === 'relu6') {
    opType = UnaryOpType.RELU6;
  } else if (activation === 'sigmoid') {
    opType = UnaryOpType.SIGMOID;
  }

  const program = new UnaryOpProgram(x.shape, opType as UnaryOpType);
  return backend.runWebGPUProgram(program, [x], x.dtype);
}
