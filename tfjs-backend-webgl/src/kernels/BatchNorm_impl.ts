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

import {env, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {BatchNormProgram} from '../batchnorm_gpu';
import {BatchNormPackedProgram} from '../batchnorm_packed_gpu';

export function batchNorm(
    x: TensorInfo, mean: TensorInfo, variance: TensorInfo,
    backend: MathBackendWebGL, offset?: TensorInfo, scale?: TensorInfo,
    varianceEpsilon?: number): TensorInfo {
  const inputs = [x, mean, variance];

  let offsetShape = null;
  if (offset != null) {
    offsetShape = offset.shape;
    inputs.push(offset);
  }

  let scaleShape = null;
  if (scale != null) {
    scaleShape = scale.shape;
    inputs.push(scale);
  }

  const program = env().getBool('WEBGL_PACK_NORMALIZATION') ?
      new BatchNormPackedProgram(
          x.shape, mean.shape, variance.shape, offsetShape, scaleShape,
          varianceEpsilon) :
      new BatchNormProgram(
          x.shape, mean.shape, variance.shape, offsetShape, scaleShape,
          varianceEpsilon);
  const output = backend.runWebGLProgram(program, inputs, inputs[0].dtype);
  return output;
}
