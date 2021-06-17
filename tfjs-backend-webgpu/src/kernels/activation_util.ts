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

import {backend_util} from '@tensorflow/tfjs-core';

import {BinaryOpType, getBinaryOpString} from './binary_op_util';
import {getUnaryOpString, UnaryOpType} from './unary_op_util';

export function mapActivationToShaderProgram(
    activation: backend_util.Activation, packed = false,
    useWgsl = false): string {
  if (activation === null) {
    return null;
  } else if (activation === 'linear') {
    return getUnaryOpString(UnaryOpType.LINEAR);
  } else if (activation === 'relu') {
    return getUnaryOpString(UnaryOpType.RELU, packed, useWgsl);
  } else if (activation === 'elu') {
    return getUnaryOpString(UnaryOpType.ELU, packed);
  } else if (activation === 'relu6') {
    return getUnaryOpString(UnaryOpType.RELU6, packed, useWgsl);
  } else if (activation === 'prelu') {
    return getBinaryOpString(BinaryOpType.PRELU, packed, useWgsl);
  } else if (activation === 'sigmoid') {
    return getUnaryOpString(UnaryOpType.SIGMOID);
  }
  throw new Error(`Activation ${
      activation} has not been implemented for the WebGPU backend.`);
}
