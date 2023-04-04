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
import {typeSnippet} from './webgpu_program';

export function activationFnSnippet(
    activation: backend_util.Activation, hasPreluActivationWeights = false,
    packed = false, coordsLength = 3): string {
  if (activation === null) {
    return '';
  }

  let activationOpSnippet = '';
  if (activation === 'linear') {
    activationOpSnippet = getUnaryOpString(UnaryOpType.LINEAR);
  } else if (activation === 'relu') {
    activationOpSnippet = getUnaryOpString(UnaryOpType.RELU, packed);
  } else if (activation === 'elu') {
    activationOpSnippet = getUnaryOpString(UnaryOpType.ELU, packed);
  } else if (activation === 'relu6') {
    activationOpSnippet = getUnaryOpString(UnaryOpType.RELU6, packed);
  } else if (activation === 'prelu') {
    activationOpSnippet = getBinaryOpString(BinaryOpType.PRELU, packed);
  } else if (activation === 'sigmoid') {
    activationOpSnippet = getUnaryOpString(UnaryOpType.SIGMOID, packed);
  } else if (activation === 'leakyrelu') {
    activationOpSnippet = getUnaryOpString(UnaryOpType.LEAKYRELU, packed);
  } else {
    throw new Error(`Activation ${
        activation} has not been implemented for the WebGPU backend.`);
  }
  const elementSize = packed ? 4 : 1;
  const dataType = typeSnippet(elementSize);
  let activationFnSnippet = '';
  if (hasPreluActivationWeights) {
    activationFnSnippet = `
      fn activation(a : ${dataType}, coords : vec${coordsLength}<i32>) -> ${
        dataType} {
        let b = getPreluActivationWeightsByOutputCoords(coords);
        ${activationOpSnippet}
      }`;
  } else {
    activationFnSnippet = `
      fn activation(a : ${dataType}, coords : vec${coordsLength}<i32>) -> ${
        dataType} {
        ${activationOpSnippet}
      }`;
  }
  return activationFnSnippet;
}

export function biasActivationSnippet(
    hasBias: boolean, activation: backend_util.Activation): string {
  return `
      ${hasBias ? 'value = value + getBiasByOutputCoords(coords);' : ''}
      ${activation ? 'value = activation(value, coords);' : ''}
      `;
}
