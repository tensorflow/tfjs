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
import {getShaderBinary} from './Binary';
import {getShaderUnary} from './Unary';

function capitalize(s: string) {
  return s[0].toUpperCase() + s.slice(1);
}

export function mapActivationToShaderProgram(
    activation: backend_util.Activation, packed = false): string {
  if (activation === null) {
    return null;
  } else {
    let key = capitalize(activation);
    if (!['linear', 'sigmoid'].includes(activation) && packed) {
      key += '_vec4';
    }
    if (['elu', 'linear', 'relu', 'relu6', 'sigmoid'].includes(activation)) {
      return getShaderUnary(key);
    } else if (activation === 'prelu') {
      return getShaderBinary(key);
    }
  }
  throw new Error(`Activation ${
      activation} has not been implemented for the WebGPU backend.`);
}
