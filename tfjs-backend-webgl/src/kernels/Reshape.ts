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

import {KernelFunc, Reshape, ReshapeAttrs, ReshapeInputs, TensorInfo} from '@tensorflow/tfjs-core';
import {KernelConfig, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {ReshapePackedProgram} from '../reshape_packed_gpu';
import {getBatchDim, getRowsCols, isReshapeFree} from '../webgl_util';

function packedReshape(
    input: TensorInfo, afterShape: number[],
    backend: MathBackendWebGL): TensorInfo {
  const input3DShape =
      [getBatchDim(input.shape),
       ...getRowsCols(input.shape)] as [number, number, number];
  const input3D: TensorInfo = {
    dtype: input.dtype,
    shape: input3DShape,
    dataId: input.dataId
  };
  const afterShapeAs3D =
      [getBatchDim(afterShape),
       ...getRowsCols(afterShape)] as [number, number, number];

  const program = new ReshapePackedProgram(afterShapeAs3D, input3DShape);
  const preventEagerUnpackingOfOutput = true;
  const output = backend.runWebGLProgram(
      program, [input3D], input.dtype, null /* customSetup */,
      preventEagerUnpackingOfOutput);
  return {dataId: output.dataId, shape: afterShape, dtype: output.dtype};
}

export function reshape(args: {
  inputs: ReshapeInputs,
  backend: MathBackendWebGL,
  attrs: ReshapeAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {shape} = attrs;
  const webglBackend = backend;

  const xSize = util.sizeFromShape(x.shape);
  const $shape = util.inferFromImplicitShape(shape, xSize);
  const $xSize = util.sizeFromShape($shape);

  util.assert(
      xSize === $xSize,
      () => `The new shape (${$shape}) has ${$xSize} elements and the old ` +
          `shape (${x.shape}) has ${xSize} elements. The new shape and old ` +
          `shape must have the same number of elements.`);

  const xTexData = webglBackend.texData.get(x.dataId);
  if (xTexData.isPacked && !isReshapeFree(x.shape, $shape) &&
      !(xTexData.texture !== null && isReshapeFree(xTexData.shape, $shape))) {
    return packedReshape(x, $shape, webglBackend);
  }

  webglBackend.incRef(x.dataId);

  return {dataId: x.dataId, shape: $shape, dtype: x.dtype};

  // const out = webglBackend.makeTensorInfo(shape, x.dtype);
  // return out;
}

export const reshapeConfig: KernelConfig = {
  kernelName: Reshape,
  backendName: 'webgl',
  kernelFunc: reshape as {} as KernelFunc
};
