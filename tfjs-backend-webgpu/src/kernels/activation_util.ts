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

import {BinaryOpType, getBinaryOpString} from './binary_ops';
import * as unary_op from './unary_op_webgpu';

export function mapActivationToShaderProgram(
    activation: backend_util.Activation, packed = false): string {
  if (activation === null) {
    return null;
  } else if (activation === 'linear') {
    return unary_op.LINEAR;
  } else if (activation === 'relu') {
    return packed ? unary_op.RELU_VEC4 : unary_op.RELU;
  } else if (activation === 'elu') {
    return packed ? unary_op.ELU_VEC4 : unary_op.ELU;
  } else if (activation === 'relu6') {
    return unary_op.RELU6;
  } else if (activation === 'prelu') {
    return getBinaryOpString(BinaryOpType.PRELU, packed);
  } else if (activation === 'sigmoid') {
    return unary_op.SIGMOID;
  }
  throw new Error(`Activation ${
      activation} has not been implemented for the WebGPU backend.`);
}
