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

import {backend_util} from '@tensorflow/tfjs-core';

import {PRELU, PRELU_PACKED} from '../kernels/Prelu';
import * as unary_op from '../unaryop_gpu';
import * as unary_packed_op from '../unaryop_packed_gpu';

export function mapActivationToShaderProgram(
    activation: backend_util.Activation, packed = false): string {
  if (activation === 'linear') {
    if (packed) {
      return unary_packed_op.LINEAR;
    }
    return unary_op.LINEAR;
  } else if (activation === 'relu') {
    if (packed) {
      return unary_packed_op.RELU;
    }
    return unary_op.RELU;
  } else if (activation === 'elu') {
    if (packed) {
      return unary_packed_op.ELU;
    }
    return unary_op.ELU;
  } else if (activation === 'relu6') {
    if (packed) {
      return unary_packed_op.RELU6;
    }
    return unary_op.RELU6;
  } else if (activation === 'prelu') {
    if (packed) {
      return PRELU_PACKED;
    }
    return PRELU;
  }
  throw new Error(`Activation ${
      activation} has not been implemented for the WebGL backend.`);
}
