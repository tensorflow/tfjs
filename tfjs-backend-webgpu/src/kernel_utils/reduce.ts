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

import {backend_util, sumOutType, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {maxImplCPU} from './shared';
import {prodImplCPU} from './shared';
import {ReduceProgram} from '../reduce_webgpu';
import {reshape} from '../kernels/Reshape';
import {transpose} from '../kernels/Transpose';

type ReduceTypes = 'max'|'mean'|'min'|'prod'|'sum';

export function reduce(
    x: TensorInfo, axis: number|number[], keepDims: boolean,
    reduceType: ReduceTypes, backend: WebGPUBackend): TensorInfo {
  const xRank = x.shape.length;
  const toDispose = [];

  const origAxes = util.parseAxisParam(axis, x.shape);
  let axes = origAxes;
  const permutedAxes = backend_util.getAxesPermutation(axes, xRank);

  let input = x;
  if (permutedAxes != null) {
    input = transpose({inputs: {x}, attrs: {perm: permutedAxes}, backend});
    axes = backend_util.getInnerMostAxes(axes.length, xRank);
    toDispose.push(input);
  }

  backend_util.assertAxesAreInnerMostDims(reduceType, axes, xRank);

  const [reduceOutShape, reduceShape] =
      backend_util.computeOutAndReduceShapes(input.shape, axes);
  let resOutShape = reduceOutShape;
  if (keepDims) {
    // rather than reshape at the end, set the target shape here.
    resOutShape = backend_util.expandShapeToKeepDim(reduceOutShape, origAxes);
  }

  let res;
  if ((reduceType === 'max' || reduceType === 'prod') &&
      backend.shouldExecuteOnCPU([input])) {
    const xVals = backend.tensorMap.get(input.dataId).values as TypedArray;
    switch (reduceType) {
      case 'max':
        const outValues = maxImplCPU(
            xVals, util.sizeFromShape(reduceShape), resOutShape, x.dtype);
        res = backend.makeTensorInfo(resOutShape, x.dtype, outValues);
        break;
      case 'prod':
        const {outVals, outShape, outDtype} =
            prodImplCPU(input.shape, input.dtype, xVals, axes);
        res = backend.makeTensorInfo(outShape, outDtype, outVals);
        break;
      default:
        throw new Error(
            `${reduceType} CPU implementation is not yet supported.`);
    }
  } else {
    const inSize = util.sizeFromShape(reduceShape);
    const xSize = util.sizeFromShape(input.shape);
    const batchSize = xSize / inSize;

    const reduceInfo = {windowSize: inSize, inSize, batchSize, outSize: 1};
    const dtype = reduceType === 'mean' ? 'float32' : sumOutType(x.dtype);
    const uniformData = [
      {type: 'int32', data: [inSize]},
    ];
    const program = new ReduceProgram(reduceInfo, reduceType);
    const reduced =
        backend.runWebGPUProgram(program, [input], dtype, uniformData);
    toDispose.push(reduced);

    res = reshape({inputs: {x: reduced}, attrs: {shape: resOutShape}, backend});
  }

  toDispose.forEach(t => backend.disposeData(t.dataId));

  return res;
}
