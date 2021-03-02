/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {backend_util, Concat, ConcatAttrs, ConcatInputs, KernelConfig, KernelFunc, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';
import {concatImplCPU} from '../kernel_utils/shared';
import {identity} from './Identity';
import {reshape} from './Reshape';

export function concat(
    args: {inputs: ConcatInputs, backend: BackendWasm, attrs: ConcatAttrs}) {
  const {inputs, backend} = args;

  const axis = util.parseAxisParam(args.attrs.axis, inputs[0].shape)[0];

  let outShape = backend_util.computeOutShape(inputs.map(t => t.shape), axis);

  // Keep only non-empty tensors (ignore tensors with 0 in their shape).
  const $inputs = inputs.filter(t => util.sizeFromShape(t.shape) > 0);
  if ($inputs.length === 1) {
    return identity({inputs: {x: $inputs[0]}, backend});
  }

  const out = backend.makeOutput(outShape, inputs[0].dtype);

  if (util.sizeFromShape(outShape) === 0) {
    return out;
  }

  const shapes = $inputs.map(t => t.shape);
  backend_util.assertParamsConsistent(shapes, axis);

  if ($inputs[0].dtype === 'string') {
    // Any concat of n-dimensional tensors across any axis can be reduced to
    // a concatenation of two-dimensional tensors across the axis 1 by first
    // partitioning the axes of the original tensors into those less than the
    // axis to be concatenated and the rest. Then reshape the tensors
    // into a two-dimensional tensor by collapsing these two sets of axes and
    // concatenate the resulting matrices across the axis 1, finally reshaping
    // the result to have the proper shape.
    const inputs2D = $inputs.map(t => {
      const innerSize = util.sizeFromShape(t.shape.slice(axis));
      const shape = [-1, innerSize];
      return reshape({inputs: {x: t}, backend, attrs: {shape}});
    });

    const inputsValShapes = inputs2D.map(t => {
      return {vals: backend.readSync(t.dataId), shape: t.shape};
    });

    // Concats 2d tensors along axis=1.
    outShape =
        backend_util.computeOutShape(inputs2D.map(t => t.shape), 1 /* axis */);
    const simplyConcat = inputs2D[0].shape[0] === 1;
    const outVals = concatImplCPU(
                        inputsValShapes, outShape, inputs[0].dtype,
                        simplyConcat) as string[];

    const finalOutShape =
        backend_util.computeOutShape($inputs.map(t => t.shape), axis);

    out.shape = finalOutShape;
    const outData = backend.dataIdMap.get(out.dataId);
    outData.stringBytes = backend_util.fromStringArrayToUint8(outVals);

    inputs2D.forEach(t => backend.disposeData(t.dataId));

    return out;
  }

  const batchDim = util.sizeFromShape($inputs[0].shape.slice(0, axis));
  let sumInnerDims = 0;
  const innerDims = $inputs.map(input => {
    const innerDim = util.sizeFromShape(input.shape.slice(axis));
    sumInnerDims += innerDim;
    return innerDim;
  });
  const inVals = $inputs.map(input => backend.typedArrayFromHeap(input));
  const outVals = backend.typedArrayFromHeap(out);
  for (let b = 0; b < batchDim; b++) {
    let outOffset = b * sumInnerDims;
    for (let i = 0; i < inVals.length; i++) {
      const innerDim = innerDims[i];
      const inOffset = b * innerDim;
      const vals = inVals[i].subarray(inOffset, inOffset + innerDim);
      outVals.set(vals, outOffset);
      outOffset += innerDim;
    }
  }
  return out;
}

export const concatConfig: KernelConfig = {
  kernelName: Concat,
  backendName: 'wasm',
  kernelFunc: concat as {} as KernelFunc,
};
