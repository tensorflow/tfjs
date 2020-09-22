
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

import {env, FusedBatchNorm, FusedBatchNormAttrs, FusedBatchNormInputs, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {BatchNormProgram} from '../batchnorm_gpu';
import {BatchNormPackedProgram} from '../batchnorm_packed_gpu';

import {reshape} from './Reshape';

export const batchNormKernelFunc: (params: {
  inputs: FusedBatchNormInputs,
  backend: MathBackendWebGL,
  attrs: FusedBatchNormAttrs
}) => TensorInfo | TensorInfo[] = ({inputs, backend, attrs}) => {
  const {x, mean, variance, offset, scale} = inputs;

  util.assert(
      mean.shape.length === variance.shape.length,
      () => 'Batch normalization gradient requires mean and variance to have ' +
          'equal ranks.');
  util.assert(
      offset == null || mean.shape.length === offset.shape.length,
      () => 'Batch normalization gradient requires mean and offset to have ' +
          'equal ranks.');
  util.assert(
      scale == null || mean.shape.length === scale.shape.length,
      () => 'Batch normalization gradient requires mean and scale to have ' +
          'equal ranks.');

  let {varianceEpsilon} = attrs;
  if (varianceEpsilon == null) {
    varianceEpsilon = 0.001;
  }

  const $x: TensorInfo = xAs4D(x, backend);
  const $mean = as1DOr4D(mean, backend);
  const $variance = as1DOr4D(variance, backend);
  const $offset = as1DOr4D(offset, backend);
  const $scale = as1DOr4D(scale, backend);

  const finalInputs = [$x, $mean, $variance];

  let offsetShape = null;
  if ($offset != null) {
    offsetShape = $offset.shape;
    finalInputs.push($offset);
  }

  let scaleShape = null;
  if ($scale != null) {
    scaleShape = $scale.shape;
    finalInputs.push($scale);
  }

  const program = env().getBool('WEBGL_PACK_NORMALIZATION') ?
      new BatchNormPackedProgram(
          $x.shape, $mean.shape, $variance.shape, offsetShape, scaleShape,
          varianceEpsilon) :
      new BatchNormProgram(
          $x.shape, $mean.shape, $variance.shape, offsetShape, scaleShape,
          varianceEpsilon);
  const output =
      backend.runWebGLProgram(program, finalInputs, finalInputs[0].dtype);

  backend.disposeIntermediateTensorInfo($x);
  backend.disposeIntermediateTensorInfo($mean);
  backend.disposeIntermediateTensorInfo($variance);
  if ($offset != null) {
    backend.disposeIntermediateTensorInfo($offset);
  }
  if ($scale != null) {
    backend.disposeIntermediateTensorInfo($scale);
  }

  return output;
};

export const batchNormConfig: KernelConfig = {
  kernelName: FusedBatchNorm,
  backendName: 'webgl',
  kernelFunc: batchNormKernelFunc as {} as KernelFunc,
};

function xAs4D(x: TensorInfo, backend: MathBackendWebGL): TensorInfo {
  const xRank = x.shape.length;
  if (xRank === 0 || xRank === 1) {
    return reshape({
      inputs: {x},
      attrs: {shape: [1, 1, 1, util.sizeFromShape(x.shape)]},
      backend,
    });
  } else if (xRank === 2) {
    return reshape({
      inputs: {x},
      attrs: {shape: [1, 1, x.shape[0], x.shape[1]]},
      backend,
    });
  } else if (xRank === 3) {
    return reshape({
      inputs: {x},
      attrs: {shape: [1, x.shape[0], x.shape[1], x.shape[2]]},
      backend,
    });
  } else {
    backend.incRef(x.dataId);
    return {...x};
  }
}

function as1DOr4D(x: TensorInfo, backend: MathBackendWebGL): TensorInfo {
  if (x == null) {
    return null;
  }
  const xRank = x.shape.length;
  if (xRank === 0) {
    return reshape({
      inputs: {x},
      attrs: {shape: [util.sizeFromShape(x.shape)]},
      backend,
    });
  } else if (xRank === 2) {
    return reshape({
      inputs: {x},
      attrs: {shape: [1, 1, x.shape[0], x.shape[1]]},
      backend,
    });
  } else if (xRank === 3) {
    return reshape({
      inputs: {x},
      attrs: {shape: [1, x.shape[0], x.shape[1], x.shape[2]]},
      backend,
    });
  } else {
    backend.incRef(x.dataId);
    return {...x};
  }
}
